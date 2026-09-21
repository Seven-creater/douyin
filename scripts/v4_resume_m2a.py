"""Resume an existing M2-A workspace; preserve all versions and stop at candidate.

Run from the repository root using omni_src Python on the Linux GPU server.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import fcntl

    from src.config import load_config
    from src.perception.omni_pool import OmniProcessPool
    from src.agentic_video.agent_v4 import agent_loop
    from src.agentic_video.asset_studio.acceptance import write_acceptance_report
    from src.agentic_video.asset_studio.backends import (
        AssetImageEditBackend, AssetImageT2IBackend, PILUpscaleBackend,
        QwenEditIterativeMultiView)
    from src.agentic_video.asset_studio.skills import build_m2a_registry
    from src.agentic_video.asset_studio.workspace_setup import prepare_m2a_workspace
    from src.agentic_video.no_progress import NoProgressDetector
    from src.agentic_video.workspace import Workspace
    from src.agentic_video.provenance import assert_production_sources_current

    run_dir = Path("data/agentic_runs/v4_run_m2a")
    if not (run_dir / "workspace.json").is_file():
        raise RuntimeError("Expected existing M2-A workspace; refusing to reseed")
    with (run_dir / "resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        usage = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
        ], text=True)
        memory = {int(row.split(",")[0]): int(row.split(",")[1])
                  for row in usage.strip().splitlines()}
        # Reserved vLLM GPUs 2/3 are never used or modified.
        for gpu in (0, 1, 4, 5):
            if memory.get(gpu, 999999) > 500:
                raise RuntimeError(f"GPU {gpu} is occupied; resume not started")
        ws = Workspace(run_dir)
        # The archived youth-fencer chain predates exact lineage metadata and
        # is intentionally barred from starting further expensive work.
        assert_production_sources_current(
            ws, ("creative_dna", "screenplay", "asset_graph"),
            revocation_path=Path("config/agentic_revocations.json"))
        prepare_m2a_workspace(ws, "C0", night_goal=True)
        print(ws.build_map(), flush=True)
        print("Acceptance:", write_acceptance_report(ws), flush=True)
        if ws.goal_satisfied():
            print("Candidate already committed; awaiting human review", flush=True)
            return
        worker_python = "/data02/usr/wangqihao/miniconda3/envs/h3/bin/python"
        t2i = AssetImageT2IBackend(
            gpu="4", python_bin=worker_python,
            stderr_log_path=str(run_dir / "resume_t2i.log"))
        edit = AssetImageEditBackend(
            gpu="5", python_bin=worker_python,
            stderr_log_path=str(run_dir / "resume_edit.log"))
        cfg = load_config(None)
        try:
            registry = build_m2a_registry(
                t2i=t2i, multiview=QwenEditIterativeMultiView(edit),
                upscale=PILUpscaleBackend())
            with OmniProcessPool("0,1", cfg.perception.get("omni") or {},
                                 response_timeout_s=900.0) as pool:
                result = agent_loop(
                    ws, registry, controller_runner=pool, max_steps=15,
                    budget="gpu", no_progress_detector=NoProgressDetector(3))
                print("=== RESULT ===", flush=True)
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        finally:
            t2i.close()
            edit.close()
            print("Acceptance:", write_acceptance_report(ws), flush=True)


if __name__ == "__main__":
    main()
