# -*- coding: utf-8 -*-
"""v4 Agent Controller：Omni 自主选 Skill 的循环（非 if/else 工作流）。

p0626 §一/§十二：workspace 提供状态，registry 提供 affordances，
Omni Controller 根据当前状态+可用 skills+最近失败推理下一步。
Harness enforcement（前置/预算/参数）在 registry.validate_action。
"""
from __future__ import annotations

import json
from typing import Any

from src.agentic_video.skills import CHOOSE_ACTION_PROMPT, SkillRegistry
from src.agentic_video.skills.registry import SkillBlocked
from src.agentic_video.workspace import Workspace


class AgentBlocked(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


def agent_loop(workspace: Workspace, registry: SkillRegistry, *,
               controller_runner, max_steps: int = 30,
               budget: str = "cheap_text",
               no_progress_detector=None) -> dict[str, Any]:
    """主循环：inspect → Omni 选 skill → execute → test → commit/repair。

    controller_runner: Omni 池（ask 文本模式）——用于 choose_action 和
    validators（独立 prompt，Actor ≠ Test）。
    no_progress_detector: 可选 NoProgressDetector——世界状态+失败均
    连续零增量时 break（Run A-v2 集成教训：检测必须在循环内部）。
    """
    step = 0
    last_failure: dict[str, Any] | None = None
    stop_reason = "max_steps"

    while not workspace.goal_satisfied() and step < max_steps:
        step += 1

        if no_progress_detector is not None:
            progress = no_progress_detector.step(workspace, last_failure)
            if progress["stalled"]:
                stagnant = progress["consecutive_stagnant_steps"]
                workspace.record_trace(
                    step,
                    {"skill": "NO_PROGRESS", "event": stagnant},
                    {"artifact": "-"},
                    {"passed": False,
                     "failures": [{
                         "check": "no_progress",
                         "detail": f"stagnant steps: {stagnant}"}],
                     "restricted_actions":
                         no_progress_detector.restricted_actions()})
                stop_reason = "no_progress"
                break

        state_map = workspace.build_map()
        runnable = registry.runnable_skills(workspace, budget)
        recent = workspace.recent_trace(3)

        if not runnable:
            raise AgentBlocked(
                "no_runnable_skills",
                f"step={step}, map={state_map[:200]}")

        # Omni 自主选动作（非硬编码分支）
        from src.agentic_video.reference_program_v9 import _parse_one_object
        payload = json.dumps(
            {"goal": workspace.state["goal"],
             "workspace_map": state_map,
             "runnable_skills": runnable,
             "recent_failure": last_failure,
             "recent_trace": [
                 {"step": t.get("step"),
                  "action": t.get("action", {}).get("skill"),
                  "tests_passed": t.get("tests", {}).get("passed")}
                 for t in recent]},
            ensure_ascii=False, separators=(",", ":"))
        answer = controller_runner.ask(
            CHOOSE_ACTION_PROMPT + payload, max_new_tokens=512,
            stop_after_json_object=True)
        raw = getattr(answer, "text", str(answer))
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        action = _parse_one_object(text, stage="agent_controller")

        # Harness enforcement
        registry.validate_action(action, workspace)

        # 执行
        kwargs = {}
        if last_failure:
            kwargs["test_failures"] = last_failure.get("failures") or []
            kwargs["target"] = action.get("target") or ""
        result = None
        try:
            result = registry.execute(action, workspace, **kwargs)
        except SkillBlocked as blocked:
            # 依赖门拦截（BLOCKED ≠ 崩溃）：落 trace，循环继续
            result = {"artifact": None, "action": "blocked",
                      "blocked_reason": str(blocked)}
        artifact_name = str(result.get("artifact") or "")

        # 测试（Actor ≠ Test）——validator 归属于本次 skill
        # （M2 多 skill 共享输出 artifact，按 artifact 反查会猜错）
        if result.get("action") == "blocked":
            test_report = {"passed": False,
                           "failures": [{"check": "dependency_gate",
                                         "detail": str(
                                             result.get("blocked_reason")
                                         )[:200]}],
                           "validators_run": []}
        else:
            test_report = _run_validator_for_action(
                action, registry, workspace, controller_runner)

        # 落档 trace
        workspace.record_trace(step, action, result, test_report)

        if test_report.get("passed"):
            workspace.record_dependency_snapshot(artifact_name)
            workspace.commit(artifact_name)
            last_failure = None
        else:
            workspace.set_status(artifact_name, "draft")
            last_failure = test_report

    if workspace.goal_satisfied():
        stop_reason = "goal_satisfied"
    return {
        "steps_taken": step,
        "goal_satisfied": workspace.goal_satisfied(),
        "stop_reason": stop_reason,
        "goal": workspace.state["goal"],
        "final_status": workspace.get_status(
            workspace.state["goal"]["target_artifact"]),
    }


def _run_validator_for_action(action: dict[str, Any], registry: SkillRegistry,
                              workspace: Workspace, runner) -> dict[str, Any]:
    """运行本次 action 所属 skill 的 validator（独立上下文，Actor ≠ Test）。

    M1 语义不变（write/repair_screenplay 本就同一 validator）；
    M2 多 skill 共享输出 artifact 时按 artifact 反查会永远猜中第一个。
    """
    from src.agentic_video.validators import run_test
    skill_name = str((action or {}).get("skill") or "")
    spec = registry.skills.get(skill_name)
    validator_names = spec.validators if spec else []
    if not validator_names:
        return {"passed": True, "validators_run": [],
                "detail": "no validator"}
    return run_test(validator_names[0], workspace, runner)
