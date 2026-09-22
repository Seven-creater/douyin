# -*- coding: utf-8 -*-
"""v4 Skill Registry：typed contracts + cost + validators + 前置条件。

p0626 §五：Skill 有 input/output types、preconditions、validators、
cost_class、destructive flag——Agent 可据此推理"下一步哪个最划算"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.agentic_video.workspace import Workspace


class SkillBlocked(RuntimeError):
    """技能执行被依赖门拦截（如 storyboard 资产未 COMMIT）。

    Blocked 不是崩溃：Agent 落 trace 后继续循环（选别的动作/停）。
    """

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}:{detail}")
        self.reason_code = reason_code
        self.detail = detail


class SkillExecutionError(RuntimeError):
    """真实工具运行时异常（CUDA OOM / TypeError / worker 死亡等）。

    agent_loop 捕获后转成 tool_runtime 失败进 trace——环境错误必须
    是 Observation（可诊断），而不是杀掉整个进程。
    """


@dataclass
class SkillSpec:
    name: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)  # "creative_dna:committed"
    validators: list[str] = field(default_factory=list)     # "test_screenplay"
    cost_class: str = "cheap_text"  # "cheap_text" | "expensive_gpu"
    destructive: bool = False
    idempotent: bool = True
    execute_fn: Callable[..., Any] | None = None  # set by subclass
    # R2-D additive contract metadata. Defaults preserve every legacy skill.
    skill_version: str = "legacy"
    input_schema_versions: list[str] = field(default_factory=list)
    output_schema_version: str | None = None
    permission_profile: str = "workspace_legacy"
    max_calls: int | None = None
    max_repair_attempts: int = 0
    package_sha: str | None = None
    prompt_or_instruction_sha: str | None = None

    def can_run(self, workspace: Workspace) -> tuple[bool, str]:
        """检查前置条件是否满足。返回 (can_run, reason_if_not)。

        rsplit：artifact 名本身可含冒号（M2 的 "asset:C0_master:committed"
        → artifact="asset:C0_master", required="committed"）。
        用 effective_status（P0-6）：上游 stale 时下游 skill 不可运行。
        """
        for pre in self.preconditions:
            artifact, _, required = pre.rpartition(":")
            if not artifact:
                artifact, required = pre, "committed"
            status = workspace.effective_status(artifact)
            if required == "committed" and status != "committed":
                return False, f"precondition {pre} unmet ({status})"
            if required == "exists" and status == "not_started":
                return False, f"precondition {pre} unmet (not_started)"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "skill_version": self.skill_version,
                "cost_class": self.cost_class,
                "preconditions": self.preconditions,
                "validators": self.validators,
                "outputs": self.outputs,
                "input_schema_versions": self.input_schema_versions,
                "output_schema_version": self.output_schema_version,
                "permission_profile": self.permission_profile,
                "max_calls": self.max_calls,
                "max_repair_attempts": self.max_repair_attempts,
                "package_sha": self.package_sha,
                "prompt_or_instruction_sha": self.prompt_or_instruction_sha}


class SkillRegistry:
    """管理全部可用 skills；提供 runnable_skills() 供 Agent 选择。"""

    def __init__(self) -> None:
        self.skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        self.skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec:
        if name not in self.skills:
            raise KeyError(f"skill not found: {name}")
        return self.skills[name]

    def runnable_skills(self, workspace: Workspace,
                        budget: str = "cheap_text"
                        ) -> list[dict[str, Any]]:
        """返回当前可运行的 skills（前置条件满足 + 预算允许）。"""
        result = []
        for spec in self.skills.values():
            if spec.cost_class == "expensive_gpu" and budget == "cheap_text":
                continue
            can, reason = spec.can_run(workspace)
            if can:
                result.append(spec.to_dict())
        return result

    def validate_action(self, action: dict[str, Any],
                        workspace: Workspace, *,
                        budget: str | None = None,
                        allowed_skill_names: set[str] | None = None,
                        allowed_permission_profiles: set[str] | None = None,
                        calls_made: int | None = None,
                        ) -> None:
        """Harness enforcement（P0-4：预算/可运行集是硬门非提示）。

        - skill 存在
        - 在当前 runnable 集内（Omni 幻觉的 skill 一律拒）
        - cost_class 与 budget 兼容
        - 前置条件满足（effective_status）
        违规抛 SkillBlocked（进 trace，不杀进程）。
        """
        skill_name = str(action.get("skill") or "")
        if skill_name not in self.skills:
            raise SkillBlocked("unknown_skill", skill_name)
        spec = self.skills[skill_name]
        if allowed_skill_names is not None and \
                skill_name not in allowed_skill_names:
            raise SkillBlocked(
                "action_not_runnable",
                f"{skill_name} is not in the current runnable set "
                f"(state/budget)")
        if budget == "cheap_text" and spec.cost_class == "expensive_gpu":
            raise SkillBlocked(
                "budget_gate",
                f"{skill_name} is expensive_gpu, budget={budget}")
        if (allowed_permission_profiles is not None
                and spec.permission_profile not in allowed_permission_profiles):
            raise SkillBlocked(
                "permission_profile",
                f"{skill_name} requires {spec.permission_profile}")
        if (calls_made is not None and spec.max_calls is not None
                and calls_made >= spec.max_calls):
            raise SkillBlocked(
                "call_budget",
                f"{skill_name} used {calls_made}/{spec.max_calls} calls")
        can, reason = spec.can_run(workspace)
        if not can:
            raise SkillBlocked("precondition", reason)

    def execute(self, action: dict[str, Any],
                workspace: Workspace, **kwargs) -> dict[str, Any]:
        """执行 skill 并返回结果。"""
        spec = self.get(str(action.get("skill")))
        if spec.execute_fn is None:
            raise ValueError(f"skill {spec.name} has no execute_fn")
        return spec.execute_fn(workspace, **kwargs)
