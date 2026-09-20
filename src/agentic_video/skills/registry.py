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

    def can_run(self, workspace: Workspace) -> tuple[bool, str]:
        """检查前置条件是否满足。返回 (can_run, reason_if_not)。

        rsplit：artifact 名本身可含冒号（M2 的 "asset:C0_master:committed"
        → artifact="asset:C0_master", required="committed"）。
        """
        for pre in self.preconditions:
            artifact, _, required = pre.rpartition(":")
            if not artifact:
                artifact, required = pre, "committed"
            status = workspace.get_status(artifact)
            if required == "committed" and status != "committed":
                return False, f"precondition {pre} unmet ({status})"
            if required == "exists" and status == "not_started":
                return False, f"precondition {pre} unmet (not_started)"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "cost_class": self.cost_class,
                "preconditions": self.preconditions,
                "validators": self.validators,
                "outputs": self.outputs}


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
                        workspace: Workspace) -> None:
        """Harness enforcement：skill 存在、前置满足、参数合法。"""
        skill_name = str(action.get("skill") or "")
        if skill_name not in self.skills:
            raise ValueError(f"unknown skill: {skill_name}")
        spec = self.skills[skill_name]
        can, reason = spec.can_run(workspace)
        if not can:
            raise ValueError(f"precondition failed: {reason}")

    def execute(self, action: dict[str, Any],
                workspace: Workspace, **kwargs) -> dict[str, Any]:
        """执行 skill 并返回结果。"""
        spec = self.get(str(action.get("skill")))
        if spec.execute_fn is None:
            raise ValueError(f"skill {spec.name} has no execute_fn")
        return spec.execute_fn(workspace, **kwargs)
