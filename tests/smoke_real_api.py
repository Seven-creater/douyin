"""真实 Wellbyte API 冒烟脚本（手动执行，花 credits；pytest 不会自动收集）。

用法（仓库根目录）：
    python -m tests.smoke_real_api --sub-types 1001   # 单榜，先花 2 credits 检查真实结构
    python -m tests.smoke_real_api                    # 全部 5 榜（共 10 credits）
    python -m tests.smoke_real_api --refetch          # 忽略当日已存在 raw，强制重拉
    python -m tests.smoke_real_api --bad-key          # 用假 key 验证失败路径（不计费）

输出：raw 落盘路径 + 各榜视频数 + objs[0] 真实字段名清单（写 parser 的依据）。
绝不打印 API key。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from src.config import ensure_utf8_stdio, load_config, require_api_key, setup_logging
from src.trend.wellbyte_client import WellbyteClient


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="Wellbyte 真实 API 冒烟")
    ap.add_argument("--sub-types", default="", help="逗号分隔，如 1001,1002；默认全部 5 榜")
    ap.add_argument("--refetch", action="store_true", help="当日 raw 已存在也强制重拉")
    ap.add_argument("--bad-key", action="store_true", help="用假 key 验证失败路径（不计费）")
    args = ap.parse_args(argv)

    cfg = load_config()
    setup_logging(cfg.paths.logs_dir, cfg.logging_level)
    api_key = "dg_live_invalid_key_for_failure_path_test" if args.bad_key else require_api_key()

    client = WellbyteClient(
        api_key,
        base_url=cfg.wellbyte.base_url,
        endpoint=cfg.wellbyte.endpoint,
        timeout=cfg.wellbyte.timeout_seconds,
        max_attempts=cfg.wellbyte.retry_max_attempts,
        backoff_seconds=cfg.wellbyte.retry_backoff_seconds,
    )

    sub_types = [int(s) for s in args.sub_types.split(",") if s.strip()] or cfg.wellbyte.sub_types
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"目标榜单: {sub_types}  日期目录文件名前缀: {today}")

    failed = 0
    for st in sub_types:
        out = cfg.paths.raw_dir / f"{today}_{st}.json"
        oc = client.fetch_and_save(st, cfg.wellbyte.request_params, out, overwrite=args.refetch)
        name = cfg.wellbyte.sub_type_names.get(st, "?")
        print(f"\n=== sub_type={st}（{name}）===")
        print(f"  status={oc.status} videos={oc.video_count} http={oc.http_status} api_code={oc.api_code}")
        if oc.raw_path:
            print(f"  raw: {oc.raw_path}")
        if oc.error:
            print(f"  error: {oc.error}")
            failed += 1
        if oc.status == "fetched":
            payload = json.loads(out.read_text(encoding="utf-8"))
            objs = (payload.get("data") or {}).get("objs") or []
            if objs:
                print(f"  objs[0] 字段名（共 {len(objs[0])} 个）: {sorted(objs[0].keys())}")
            print(f"  data.page: {(payload.get('data') or {}).get('page')}")
            print(f"  meta: {payload.get('meta')}")

    print(f"\n结果: {len(sub_types) - failed}/{len(sub_types)} 成功")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
