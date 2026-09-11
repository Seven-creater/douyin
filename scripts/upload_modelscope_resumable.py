#!/usr/bin/env python
"""大文件断点续传上传到 ModelScope dataset（并发分片版）。

走 ModelScope dataset 的 OSS 通道（与官方 MsDataset.push 相同的底层）：
  1. HubApi（本地登录态）取 STS 临时凭证 + bucket/前缀
  2. oss2 multipart 分片上传，N 线程并发；本地 checkpoint 记录 upload_id
  3. 重启后以 OSS list_parts 的服务端状态为准续传，不重传已完成分片
  4. 单片失败指数退避重试；STS 过期（403）自动刷新凭证后继续
  5. 完成后 complete_multipart_upload + head_object 校验大小

用法: python -X utf8 scripts/upload_modelscope_resumable.py
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import oss2
from modelscope.hub.api import HubApi

FILE = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\29785\Desktop\douyin\data\raw\guimie\Demon.Slayer.Kimetsu.no.Yaiba.Infinity.Castle.2025.1080p.BluRay.x265.10bit.DTS.5Audio.mkv"
DATASET = sys.argv[2] if len(sys.argv) > 2 else 'guimie'
NAMESPACE = 'sevenchen777'
PART_SIZE = 100 * 1024 * 1024  # 100MB/片，10.2GB ≈ 105 片（OSS 上限 10000 片）
CKPT = FILE + '.upload_ckpt.json'
MAX_PART_ATTEMPTS = 8
WORKERS = 6

lock = threading.Lock()
bucket_holder = {}  # {'bucket': oss2.Bucket}，STS 过期时整体替换


def log(msg):
    print(time.strftime('[%Y-%m-%d %H:%M:%S]'), msg, flush=True)


def save_ckpt(ckpt):
    tmp = CKPT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(ckpt, f)
    os.replace(tmp, CKPT)


def make_bucket():
    """取/刷新 STS 凭证。"""
    cfg = HubApi().get_dataset_access_config(dataset_name=DATASET, namespace=NAMESPACE)
    auth = oss2.StsAuth(cfg['AccessId'], cfg['AccessSecret'], cfg['SecurityToken'])
    # Host 形如 https://{bucket}.{region}.aliyuncs.com，oss2 会自己拼 bucket 虚拟主机前缀，
    # 直接传会把 bucket 加两遍（dataset-hub.dataset-hub.…），需剥掉
    host = cfg['Host'].replace('https://', '').replace('http://', '')
    bucket_name = cfg['Bucket']
    endpoint = host[len(bucket_name) + 1:] if host.startswith(bucket_name + '.') else host
    bucket = oss2.Bucket(auth, 'https://' + endpoint, bucket_name, connect_timeout=120)
    key = cfg['Dir'].rstrip('/') + '/' + os.path.basename(FILE)
    return bucket, key


def is_sts_error(e):
    return e.status == 403 or e.code in ('InvalidAccessKeyId', 'SignatureDoesNotMatch', 'AccessDenied')


def upload_one_part(n, size, etags):
    """上传第 n 片（含重试），返回 (n, etag)。失败到上限则抛出终止整个任务。"""
    offset = (n - 1) * PART_SIZE
    length = min(PART_SIZE, size - offset)
    with open(FILE, 'rb') as f:
        f.seek(offset)
        data = f.read(length)
    for attempt in range(1, MAX_PART_ATTEMPTS + 1):
        try:
            with lock:
                bucket = bucket_holder['bucket']
            r = bucket.upload_part(bucket_holder['key'], bucket_holder['upload_id'], n, data)
            return n, r.etag
        except oss2.exceptions.OssError as e:
            if attempt == MAX_PART_ATTEMPTS:
                raise RuntimeError(f'第 {n} 片连续 {MAX_PART_ATTEMPTS} 次失败: {e.status} {e.code}') from e
            wait = min(2 ** attempt, 60)
            log(f'第 {n} 片失败({e.status} {e.code})，{wait}s 后重试 ({attempt}/{MAX_PART_ATTEMPTS})')
            time.sleep(wait)
            if is_sts_error(e):
                with lock:  # 多线程可能同时触发，加锁后重复刷新一次也无害
                    new_bucket, new_key = make_bucket()
                    bucket_holder['bucket'] = new_bucket
                    bucket_holder['key'] = new_key
    raise RuntimeError(f'第 {n} 片重试逻辑异常退出')


def main():
    size = os.path.getsize(FILE)
    bucket, key = make_bucket()
    bucket_holder.update(bucket=bucket, key=key)
    log(f'目标 oss://{bucket.bucket_name}/{key}  本地 {size:,} bytes ({size / 1024 ** 3:.2f} GB)')

    # 幂等：已完整上传则直接退出
    try:
        head = bucket.head_object(key)
        if head.content_length == size:
            log(f'远端已存在同尺寸文件，无需上传: {head.content_length:,} bytes')
            return
        log(f'远端已存在但尺寸不同({head.content_length:,})，重新上传覆盖')
    except oss2.exceptions.NotFound:
        pass

    # 恢复 checkpoint（key/分片大小变了才作废）
    ckpt = {}
    if os.path.exists(CKPT):
        try:
            with open(CKPT, encoding='utf-8') as f:
                ckpt = json.load(f)
        except (OSError, ValueError):
            ckpt = {}
        if ckpt.get('key') != key or ckpt.get('part_size') != PART_SIZE:
            log('checkpoint 与当前目标不符，忽略')
            ckpt = {}

    upload_id = ckpt.get('upload_id')
    etags = {}  # {part_number(int): etag}，以服务端 list_parts 为准
    if upload_id:
        try:
            for p in oss2.PartIterator(bucket, key, upload_id):
                etags[p.part_number] = p.etag
        except oss2.exceptions.OssError as e:
            log(f'checkpoint 的 upload_id 已失效({e.code})，重新初始化')
            upload_id = None
            etags = {}
    if not upload_id:
        upload_id = bucket.init_multipart_upload(key).upload_id
        etags = {}
    bucket_holder['upload_id'] = upload_id

    total = (size + PART_SIZE - 1) // PART_SIZE
    todo = [n for n in range(1, total + 1) if n not in etags]
    done_bytes0 = min(len(etags) * PART_SIZE, size)
    if etags:
        log(f'断点续传: OSS 上已有 {len(etags)}/{total} 片 (~{done_bytes0 / 1024 ** 3:.2f} GB)，还需 {len(todo)} 片')
    else:
        log(f'全新上传: {total} 片 x 100MB, upload_id={upload_id}')
    ckpt.update(key=key, part_size=PART_SIZE, upload_id=upload_id, parts_done=len(etags))

    t0 = time.time()
    fail_fast = None
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(upload_one_part, n, size, etags): n for n in todo}
        try:
            for fut in as_completed(futures):
                n, etag = fut.result()
                with lock:
                    etags[n] = etag
                    ckpt['parts_done'] = len(etags)
                    save_ckpt(ckpt)
                    done = min(len(etags) * PART_SIZE, size)
                    dt = time.time() - t0
                    speed = (done - done_bytes0) / 1024 ** 2 / max(dt, 1e-6)
                    eta = (size - done) / (speed * 1024 ** 2) if speed > 0.5 else float('inf')
                    log(f'part {n:3d}/{total} ok  累计 {len(etags)}/{total}  '
                        f'{done / 1024 ** 3:.2f}/{size / 1024 ** 3:.2f} GB  {speed:.1f} MB/s  ETA {eta / 60:.0f} min')
        except Exception as e:  # 任一片重试耗尽 → 记录并取消其余
            fail_fast = e
            log(f'致命错误: {e}，取消剩余分片（checkpoint 已保存，重跑可续传）')
            for f_ in futures:
                f_.cancel()

    if fail_fast:
        sys.exit(1)

    parts = [oss2.models.PartInfo(n, etags[n]) for n in sorted(etags)]
    if len(parts) != total:
        log(f'致命: 分片不齐 {len(parts)}/{total}，不提交（重跑可续传）')
        sys.exit(1)
    bucket.complete_multipart_upload(key, upload_id, parts)

    head = bucket.head_object(key)
    if head.content_length != size:
        log(f'致命: 完成后远端尺寸 {head.content_length:,} != 本地 {size:,}')
        sys.exit(1)
    os.remove(CKPT)
    log(f'上传完成 ✓  oss://{bucket.bucket_name}/{key}  {head.content_length:,} bytes')
    log(f'数据集页面: https://www.modelscope.cn/datasets/{NAMESPACE}/{DATASET}/files')


if __name__ == '__main__':
    main()
