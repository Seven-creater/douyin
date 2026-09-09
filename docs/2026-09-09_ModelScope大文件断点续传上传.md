# ModelScope 大文件断点续传上传（鬼灭素材 10.2GB）

## 背景与问题

要把本地 10.23GB 的 mkv（鬼灭之刃·无限城 1080p 蓝光压片，素材库源片）传到 ModelScope
私有 dataset，供服务器侧管线取用。要求**断点续传**（家宽上行约 1.5 MB/s，全程约 2 小时，
中途断网/进程挂掉不能从头再来）。

modelscope SDK 1.35.1 的官方路径 `modelscope upload --repo-type dataset` 不能满足：
读 `HubApi._upload_blob`（api.py:2507）源码，它对大文件是**单次流式 HTTP PUT** 到
LFS 预签名 URL，无分片、无 offset。断了重跑 = 从第 0 字节重来。唯一的"秒传"是
`_validate_blob` 按 sha256 判"已完整上传"去重——只对**完整传完**的文件生效。

## 候选方案

| 方案 | 思路 | 预期优劣 |
|---|---|---|
| A. 官方 CLI `modelscope upload` | 简单一行命令 | 无真断点；断了从头传 10GB |
| B. git clone + git-lfs push | 走 git 通道 | ModelScope LFS 是否支持 tus 续传未知；配置繁琐 |
| C. dataset 的 OSS 通道 + oss2 multipart 自研续传 | `get_dataset_access_config` 拿 STS 临时凭证直传 dataset-hub bucket，multipart 分片 + 本地 checkpoint 记 upload_id，重启后以服务端 `list_parts` 对账续传 | 真·断点续传；要自己写 ~150 行，处理 STS 过期/域名坑 |

## 实测/对比

- **A（读码确认，未实测传 10GB）**：`_upload_blob` 源码层面排除——单 PUT 流式，无断点语义。
- **C（实测）**：
  - 10:30 首跑单线程：每片 100MB 约 66-75s ≈ **1.5 MB/s 单流**；
  - 改 6 线程并发后 4 分钟**一片未完** → 并发不提速，**瓶颈是家宽上行总带宽**（~12 Mbps），不是单连接限制；
  - 传到 2/105 片时人为杀进程重跑：正确识别服务端已有 2 片、跳过续传 ✓；
  - 中途 `ststoken` 接口出现一次"无权访问该数据集"（数据集本身 Visibility=1 正常、
    get_dataset 可读、cookies/token 均有效）——**平台侧瞬时抖动**，数分钟后同调用自愈。

## 诊断依据

- "并发救不了"的判定：单流 1.5 MB/s；6 流跑 4 分钟 parts_done 从 2→2（若总带宽充足，
  6 流应至少完成 ~5 片）。排除"单连接被限速"假设、坐实"上行总带宽瓶颈"。
- "官方无断点"的判定：直接读 SDK 源码 `_upload_blob`（api.py:2507-2602），
  全路径只有 `session.put(url, data=read_in_chunks(...))` 一个 PUT，无分片 API 调用。
- "ststoken 拒绝是瞬时抖动"：同凭证同参数 5 分钟前后一败一成，且期间数据集元数据/权限无变更。

## 选择与理由

选 **C**：`scripts/upload_modelscope_resumable.py`。关键设计：

1. checkpoint 只存 `{key, part_size, upload_id}`，**完成分片以 OSS `list_parts` 服务端状态为准**
   （本地 etag 列表可能落后于服务端）；upload_id 属于 bucket 而非 STS token，换凭证不失效。
2. 单片失败指数退避重试 8 次；403/InvalidAccessKeyId 触发 STS 凭证整体刷新后继续。
3. 完成后 `complete_multipart_upload` + `head_object` 比对字节数，等值才算成功。

**带宽瓶颈的应对（用户拍板）**：上行 1.5 MB/s 下 10.23GB ≈ 2 小时。本机有 RTX 5060
（NVENC）→ NVENC HEVC CQ24 硬编 **12.9 倍速、12 分钟**压到 **3.63GB**（35%），音轨 5 条
（日 DTS5.1/日 FLAC/国语/台配/粤语，约 3GB）只留日语主音轨 AAC 256k，字幕/字体附件全弃。
上传时间 2h → ~45min。代价：一次有损转码 + 云端无原片备份（用户明确选"只传压缩版"）。

## 验证

- 断点续传：杀进程重跑，日志出现"断点续传: OSS 上已有 2/105 片"且进度从 2 片起算 ✓；
- 压缩版时长：ffprobe 输出 9295.828s 与源 9295.828s 完全一致 ✓，exit 0；
- 完整性校验（head_object 字节比对 + dataset 页面可见）：压缩版上传**进行中，待回填**；
- 未测路径：跨天续传（STS 12h 过期后 refresh 分支只在代码层面 review，未实际触发）；
  403 抖动重试路径未在 ststoken 获取处覆盖（首次 make_bucket 失败直接退出，需重跑脚本）。

## 结果

（待回填：压缩版上传完成时间、总耗时、远端字节数比对、服务器侧下载是否可见）

## 从结果学到什么

1. ModelScope 传大文件别用 `modelscope upload`——读 `_upload_blob` 源码就知道它没有断点语义；
   正确通道是 `get_dataset_access_config` → OSS multipart（官方 MsDataset.push 底层同款）。
2. 该通道 `Host` 字段自带 bucket 前缀（`dataset-hub.oss-cn-hangzhou...`），直接喂给
   `oss2.Bucket` 会拼成 `dataset-hub.dataset-hub....` DNS 解析失败——**先剥 bucket 再传 endpoint**。
3. 并发度救不了上行总带宽：先测单流速率，再决定是加线程还是压文件；1.5 MB/s 上行下，
   "NVENC 12 分钟压缩 + 半小时上传"完胜"2 小时裸传"，且 5 音轨砍 1 是体积大头。
4. 平台 ststoken 偶发 403 是抖动不是权限问题：先 `get_dataset` 验元数据再重试，别急着改权限。
