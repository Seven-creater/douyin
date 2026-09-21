# -*- coding: utf-8 -*-
"""人审批准 CLI（L3 → ASSET LOCK）。

明早用法（服务器 repo 根目录）：
  python -m src.agentic_video.asset_studio.approve \
      --run-dir data/agentic_runs/v4_run_m2a --decision approve \
      [--reviewer name] [--note "..."]

写入 human_approval:C0（绑定 candidate_sha + 各视图 sha）并执行
approve_character_asset → asset:C0 COMMIT。decision=reject 只落记录，
不产生 final。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--asset", default="C0")
    parser.add_argument("--decision", choices=["approve", "reject"],
                        required=True)
    parser.add_argument("--reviewer", default="human")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    from src.agentic_video.workspace import Workspace
    from src.agentic_video.provenance import APPROVAL_POLICY_VERSION
    from src.agentic_video.skills.registry import SkillBlocked
    ws = Workspace(Path(args.run_dir))
    candidate = ws.read_artifact(f"asset:{args.asset}_candidate")
    if not candidate:
        sys.exit(f"no committed candidate at {args.run_dir}")
    approval = {
        "asset_id": args.asset,
        "candidate_id": f"asset:{args.asset}_candidate",
        "candidate_sha": ws.get_sha(f"asset:{args.asset}_candidate"),
        "views": {view: entry.get("sha") for view, entry in
                  (candidate.get("views_4k") or {}).items()},
        "identity_sheet_sha": (candidate.get("sheet") or {}).get("sha"),
        "parent_shas": [
            {"artifact_id": f"asset:{args.asset}_candidate",
             "sha": ws.get_sha(f"asset:{args.asset}_candidate")},
            *[{"artifact_id": f"view:{view}", "sha": entry.get("sha")}
              for view, entry in (candidate.get("views_4k") or {}).items()],
            {"artifact_id": "identity_sheet",
             "sha": (candidate.get("sheet") or {}).get("sha")},
        ],
        "approval_policy_version": APPROVAL_POLICY_VERSION,
        "decision": args.decision,
        "reviewer": args.reviewer,
        "note": args.note}
    ws.write_draft(f"human_approval:{args.asset}", approval)
    ws.record_dependency_snapshot(f"human_approval:{args.asset}")
    ws.commit(f"human_approval:{args.asset}")
    print(f"human_approval:{args.asset} committed "
          f"(decision={args.decision})")

    if args.decision != "approve":
        print("rejected — candidate stays machine_validated, no final "
              "asset committed")
        return

    from src.agentic_video.asset_studio.skills import build_m2a_registry
    registry = build_m2a_registry(t2i=None, multiview=None, upscale=None,
                                  asset_id=args.asset)
    try:
        result = registry.execute(
            {"skill": "approve_character_asset"}, ws)
    except SkillBlocked as blocked:
        sys.exit(f"APPROVE BLOCKED: {blocked}")
    from src.agentic_video.validators import run_test
    report = run_test("test_asset_approved", ws, runner=None)
    if report["passed"]:
        ws.record_dependency_snapshot(f"asset:{args.asset}")
        ws.commit(f"asset:{args.asset}")
        print(f"asset:{args.asset} COMMITTED (human approved)")
        print(json.dumps(result, ensure_ascii=False))
    else:
        sys.exit(f"test_asset_approved failed: {report}")


if __name__ == "__main__":
    main()
