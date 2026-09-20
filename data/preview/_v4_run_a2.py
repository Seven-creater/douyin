# -*- coding: utf-8 -*-
"""v4 Agent Harness Run A-v2：共享 schema + no-progress + repair_asset_graph。

修正三个 Run A 发现的问题后重跑，Goal 仍 = asset_graph:committed。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

if __name__ == "__main__":
    RUN_DIR = Path("data/agentic_runs/v4_run_a2")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    from src.config import load_config
    from src.perception.omni_pool import OmniProcessPool
    from src.agentic_video.workspace import Workspace
    from src.agentic_video.skills.asset_schema import (
        ASSET_GRAPH_FORMAT_SPEC, validate_asset_graph)
    from src.agentic_video.no_progress import NoProgressDetector

    cfg = load_config(None)
    ws = Workspace(RUN_DIR)

    # 种子 creative_dna
    dna_path = Path(
        "data/agentic_runs/ragen_20260919/transfer_contract.json")
    if dna_path.is_file():
        dna = json.loads(dna_path.read_text(encoding="utf-8"))
        ws.write_draft("creative_dna", dna)

    # 构建 registry（注入共享 schema）
    from src.agentic_video.skills import build_m1_registry

    # 修补 extract_assets prompt：加入共享 schema
    import src.agentic_video.skills as skills_mod
    original_extract_prompt = skills_mod.EXTRACT_ASSETS_PROMPT
    skills_mod.EXTRACT_ASSETS_PROMPT = (
        original_extract_prompt + "\n\n输出必须遵循此 schema：\n"
        + ASSET_GRAPH_FORMAT_SPEC)

    # 修补 validator：先 deterministic schema 校验，再 Omni 语义校验
    from src.agentic_video.validators import run_test as orig_run_test

    def patched_run_test(validator_name, workspace, runner):
        if validator_name == "test_asset_graph":
            graph = workspace.read_artifact("asset_graph")
            if not graph:
                return {"passed": False,
                        "failures": [{"check": "exists",
                                      "detail": "empty"}],
                        "validators_run": [validator_name]}
            # Deterministic schema check first
            failures = validate_asset_graph(graph)
            if failures:
                return {"passed": False, "failures": failures,
                        "validators_run": [validator_name]}
            # Schema passed → semantic check (coverage etc.) via Omni
            return orig_run_test(validator_name, workspace, runner)
        return orig_run_test(validator_name, workspace, runner)

    import src.agentic_video.validators as validators_mod
    validators_mod.run_test = patched_run_test

    with OmniProcessPool("0,1", cfg.perception.get("omni") or {},
                         response_timeout_s=600.0) as pool:
        registry = build_m1_registry(runner=pool)

        # Bootstrap creative_dna
        from src.agentic_video.validators import run_test
        report = run_test("test_creative_dna", ws, runner=pool)
        ws.record_trace(0, {"skill": "bootstrap"},
                        {"artifact": "creative_dna"}, report)
        if report["passed"]:
            ws.record_dependency_snapshot("creative_dna")
            ws.commit("creative_dna")
            print("step 0: creative_dna COMMITTED")
        else:
            sys.exit(f"creative_dna failed: {report}")

        # Agent loop with no-progress detection
        from src.agentic_video.agent_v4 import agent_loop
        detector = NoProgressDetector(limit=3)

        # Wrap agent_loop to inject no-progress check
        step = 0
        last_failure = None
        max_steps = 20

        while not ws.goal_satisfied() and step < max_steps:
            step += 1
            # No-progress check
            progress = detector.step(ws, last_failure)
            if progress["stalled"]:
                print(f"step {step}: NO_PROGRESS detected "
                      f"({progress['consecutive_stagnant_steps']} stagnant)")
                ws.record_trace(step,
                                {"skill": "NO_PROGRESS", "event": True},
                                {"stalled": True}, {"passed": False})
                break

            # Run one agent step (via agent_loop with max_steps=1)
            result = agent_loop(ws, registry, controller_runner=pool,
                                max_steps=1, budget="cheap_text")
            if not result.get("goal_satisfied"):
                # Get latest trace entry for failure tracking
                recent = ws.recent_trace(1)
                if recent:
                    entry = recent[0]
                    last_failure = (None if entry.get("tests", {}).get(
                        "passed") else entry.get("tests"))
            else:
                last_failure = None

        print("=== RESULT ===")
        print(json.dumps({
            "steps_taken": step,
            "goal_satisfied": ws.goal_satisfied(),
            "asset_graph_status": ws.get_status("asset_graph"),
        }, ensure_ascii=False, indent=1))

    print("\n=== TRACE ===")
    trace_lines = (RUN_DIR / "agent_trace.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    for line in trace_lines:
        entry = json.loads(line)
        skill = entry.get("action", {}).get("skill", "?")
        target = entry.get("action", {}).get("target", "")
        reason = str(entry.get("action", {}).get("reason", ""))[:90]
        passed = entry.get("tests", {}).get("passed", "?")
        failures = entry.get("tests", {}).get("failures") or []
        fdetail = failures[0].get("detail", "")[:60] if failures else ""
        print(f"  step {entry.get('step', '?'):2d} | {skill:25s} "
              f"| {'PASS' if passed else 'FAIL'} | {reason} | {fdetail}")
    print("\n=== MAP ===")
    print(ws.build_map())
