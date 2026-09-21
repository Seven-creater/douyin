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
from src.agentic_video.skills.registry import (
    SkillBlocked, SkillExecutionError)
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
    failure_artifact = ""
    # Resume the latest unresolved validation, including across read-only steps.
    for entry in reversed(workspace.recent_trace(1000)):
        previous = entry.get("result") or {}
        name = str(previous.get("artifact") or "")
        tests = entry.get("tests") or {}
        active = (workspace.state["artifacts"].get(name) or {}).get(
            "active_version")
        if (name and active and previous.get("version") == active
                and tests.get("passed") is False
                and workspace.effective_status(name) != "committed"):
            last_failure, failure_artifact = tests, name
            break
    recovery_used = False
    stop_reason = "max_steps"

    while not workspace.goal_satisfied() and step < max_steps:
        step += 1

        runnable = registry.runnable_skills(workspace, budget)
        recovery_mode = False
        if no_progress_detector is not None:
            progress = no_progress_detector.step(workspace, last_failure)
            if progress["stalled"]:
                stagnant = progress["consecutive_stagnant_steps"]
                allowed = no_progress_detector.restricted_actions(
                    runnable, failure_artifact)
                repairs = [row for row in runnable if row["name"] in allowed]
                if recovery_used or not last_failure or not repairs:
                    workspace.record_trace(
                        step, {"skill": "NO_PROGRESS", "event": stagnant},
                        {"artifact": "-"},
                        {"passed": False,
                         "failures": [{"check": "no_progress",
                                       "detail": f"stagnant steps: {stagnant}"}],
                         "restricted_actions": allowed})
                    stop_reason = "no_progress"
                    break
                # One bounded recovery opportunity; validate_action enforces it.
                runnable = repairs
                recovery_used = recovery_mode = True

        state_map = workspace.build_map()
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
             "failure_artifact": failure_artifact,
             "recovery_mode": recovery_mode,
             "recent_trace": [
                 {"step": t.get("step"),
                  "action": t.get("action", {}).get("skill"),
                  "tests_passed": t.get("tests", {}).get("passed"),
                  "observation": t.get("result", {})}
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

        # Harness enforcement（P0-4：预算/可运行集硬门）
        runnable_names = {row["name"] for row in runnable}

        # 执行（P0-8：blocked 与真实运行时异常都转 Observation）
        kwargs = {}
        if last_failure:
            kwargs["test_failures"] = last_failure.get("failures") or []
            kwargs["target"] = action.get("target") or ""
        result = None
        try:
            registry.validate_action(action, workspace, budget=budget,
                                      allowed_skill_names=runnable_names)
            result = registry.execute(action, workspace, **kwargs)
        except SkillBlocked as blocked:
            # 依赖门/预算门拦截（BLOCKED ≠ 崩溃）：落 trace，循环继续
            result = {"artifact": None, "action": "blocked",
                      "blocked_reason": str(blocked)}
        except SkillExecutionError as exc:
            result = {"artifact": None, "action": "execution_failed",
                      "error_type": type(exc).__name__,
                      "error": str(exc)[:400]}
        except Exception as exc:  # noqa: BLE001 — 环境错误必须是 Observation
            result = {"artifact": None, "action": "execution_failed",
                      "error_type": type(exc).__name__,
                      "error": str(exc)[:400]}
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
        elif result.get("action") == "execution_failed":
            # P0-8：工具运行时失败 ≠ 验收失败——分开标记供 Agent 诊断
            test_report = {
                "passed": False, "failure_class": "tool_runtime",
                "failures": [{
                    "check": "skill_execution",
                    "skill": str(action.get("skill") or ""),
                    "error_type": result.get("error_type"),
                    "detail": str(result.get("error"))[:300]}],
                "validators_run": []}
        else:
            test_report = _run_validator_for_action(
                action, registry, workspace, controller_runner)

        # 落档 trace
        if recovery_mode:
            test_report["recovery_mode"] = True
        workspace.record_trace(step, action, result, test_report)

        if test_report.get("passed"):
            # 无 validator 的动作（inspect 等只读诊断）不得触发 commit
            # ——M3-A dry-run 教训：inspect auto-pass 把垃圾 draft
            # commit 了，goal 假 satisfied
            if test_report.get("validators_run"):
                workspace.record_dependency_snapshot(artifact_name)
                workspace.commit(artifact_name)
                if artifact_name == failure_artifact:
                    last_failure = None
                    failure_artifact = ""
        else:
            if artifact_name:
                workspace.set_status(artifact_name, "draft")
            # A rejected action does not erase the unresolved artifact failure.
            if artifact_name or last_failure is None:
                last_failure = test_report
                failure_artifact = artifact_name

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
