# -*- coding: utf-8 -*-
"""v4 Agent Harness Run A：自然运行——Omni 从零自主完成 creative_dna→
screenplay→(repair)→asset_graph，Goal = asset_graph:committed。

不告诉 Agent 下一步做什么——只给 Production Map + Goal + runnable
Skills + 最近失败，让 Omni 自己选。全程落 agent_trace.jsonl。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

if __name__ == "__main__":
    RUN_DIR = Path("data/agentic_runs/v4_run_a")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    from src.config import load_config
    from src.perception.omni_pool import OmniProcessPool
    from src.agentic_video.workspace import Workspace
    from src.agentic_video.skills import build_m1_registry
    from src.agentic_video.agent_v4 import agent_loop

    cfg = load_config(None)
    ws = Workspace(RUN_DIR)

    # 已有 creative_dna（ragen_director 产物）作为种子
    dna_path = Path("data/agentic_runs/ragen_20260919/transfer_contract.json")
    if dna_path.is_file():
        dna = json.loads(dna_path.read_text(encoding="utf-8"))
        # enrich with rhythm + section roles from contract
        ws.write_draft("creative_dna", dna)

    with OmniProcessPool("0,1", cfg.perception.get("omni") or {},
                         response_timeout_s=600.0) as pool:
        # Step 0: deterministic validator for creative_dna
        from src.agentic_video.validators import run_test
        report = run_test("test_creative_dna", ws, runner=pool)
        ws.record_trace(0, {"skill": "bootstrap"}, {"artifact": "creative_dna"},
                        report)
        if report["passed"]:
            ws.record_dependency_snapshot("creative_dna")
            ws.commit("creative_dna")
            print("step 0: creative_dna COMMITTED (deterministic)")
        else:
            sys.exit(f"creative_dna failed: {report}")

        # Agent loop from screenplay onward
        registry = build_m1_registry(runner=pool)
        result = agent_loop(ws, registry, controller_runner=pool,
                            max_steps=15, budget="cheap_text")

    print("=== RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print("\n=== TRACE ===")
    trace_lines = (RUN_DIR / "agent_trace.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    for line in trace_lines:
        entry = json.loads(line)
        skill = entry.get("action", {}).get("skill", "?")
        target = entry.get("action", {}).get("target", "")
        reason = str(entry.get("action", {}).get("reason", ""))[:80]
        passed = entry.get("tests", {}).get("passed", "?")
        print(f"  step {entry.get('step', '?'):2d} | {skill:25s} "
              f"| target={target:5s} | test={'PASS' if passed else 'FAIL'}"
              f" | {reason}")
    print("\n=== MAP ===")
    print(ws.build_map())
