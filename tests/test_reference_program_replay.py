# -*- coding: utf-8 -*-
"""P0.4 真机输出重放门：历史 run 的真实模型输出必须在新代码下全过校验。

背景（用户批评"每次都要重跑"）：真机跑曾暴露一类只在真实模型输出下出现的
契约违例（operation_type 枚举混用、continuity_basis 漏证据、边界功能误选），
而 FakeRunner 测试只锚定期望行为。本文件把已完成 run 的真实产物固化为重放
夹具：**新代码上服务器前必须先在这里全过**，不再盲跑。

夹具来源：data/replay/p6_v9_*/（p03f2 / p04a / p04b 三轮真机产物，
由 scripts 侧 tar+scp 拉回；无夹具环境自动 skip）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agentic_video import narrative_boundary
from src.agentic_video.reference_program_v9 import (
    COMPOSITION_MODES, _parse_one_object, compile_material_requirements,
    validate_reference_programs)

REPLAY_ROOT = Path(__file__).resolve().parents[1] / "data" / "replay"
_RUN_DIRS = sorted(REPLAY_ROOT.glob("p6_v9_*_20260918")) if REPLAY_ROOT.is_dir() else []
RUNS = [path.name for path in _RUN_DIRS]


def _load(run: str, name: str) -> dict:
    return json.loads(
        (REPLAY_ROOT / run / name).read_text(encoding="utf-8"))


def _replay_validation(run: str, tmp_path: Path) -> dict:
    """历史产物 + 当前代码（程序侧矫正重放）→ 必须零错误。"""
    ledger = _load(run, "reference_evidence.json")
    observations = _load(run, "section_observations.json")
    reconciliation = _load(run, "boundary_reconciliation.json")
    conflicts = _load(run, "semantic_conflicts.json")
    montage = _load(run, "montage_shot_observations.json")
    normalization = _load(run, "shot_normalization.json")
    content = _load(run, "reference_content_program.json")
    edit = json.loads(json.dumps(_load(run, "reference_edit_program.json")))
    # 与 build_reference_edit_program 一致的确定性矫正（历史 run 可能缺新修）
    for operation in edit.get("operations") or []:
        if str(operation.get("operation_type") or "") in COMPOSITION_MODES:
            operation["operation_type"] = "hard_cut"
    # 与 build_reference_content_program 一致的证据回填
    known = {str(row.get("evidence_id")) for row in
             content.get("reference_observations") or []}
    for section in content.get("sections") or []:
        section_ids = [str(item) for item in section.get("evidence_ids") or []
                       if str(item) in known]
        for dimension, support in (section.get("continuity_basis") or {}).items():
            level = (section.get("continuity") or {}).get(dimension)
            if (level in {"required", "preferred"} and
                    not (support.get("evidence_ids") or []) and section_ids):
                support["evidence_ids"] = list(section_ids)
    # 真实管线里确定性修正发生在 hash 绑定之前；重放历史 run 时镜像这一点，
    # 否则绑定校验会把"修好的 content"误判为篡改。
    from src.agentic_video.manifest import json_hash
    if edit.get("content_program_sha256") is not None:
        edit["content_program_sha256"] = json_hash(content)
    requirements = compile_material_requirements(
        content, edit, tmp_path, normalization=normalization)
    return validate_reference_programs(
        content, edit, requirements, ledger, observations,
        reconciliation=reconciliation, conflicts=conflicts,
        montage=montage, normalization=normalization)


@pytest.mark.skipif(not RUNS, reason="no replay artifacts under data/replay")
@pytest.mark.parametrize("run", RUNS)
def test_historical_real_outputs_pass_with_current_code(
        run: str, tmp_path: Path) -> None:
    result = _replay_validation(run, tmp_path)
    assert result["passed"], (run, result["errors"])
    assert all(result["narrative_usability"].values()), (
        run, result["narrative_usability"])


@pytest.mark.skipif(not RUNS, reason="no replay artifacts under data/replay")
@pytest.mark.parametrize("run", RUNS)
def test_boundary_raw_verdicts_are_valid_and_decided_deterministically(
        run: str) -> None:
    """历史边界原始回答：schema 合法；same_event=true 的闭环一票否决必须成立。"""
    raw_files = sorted(
        (REPLAY_ROOT / run / "raw_responses").glob("boundary_*.txt"))
    assert raw_files, run
    saw_narrative_schema = False
    for raw_file in raw_files:
        verdict = _parse_one_object(
            raw_file.read_text(encoding="utf-8"), stage="boundary_reconciliation")
        if "before_function" not in verdict:
            continue  # 旧层级 schema 时代的 run（p03f2）没有功能枚举
        saw_narrative_schema = True
        narrative_boundary.validate_narrative_verdict(verdict)
        decision = narrative_boundary.decide_narrative_boundary(
            verdict.get("before_function"), verdict.get("after_function"),
            verdict.get("same_event"))
        if verdict.get("same_event") is True:
            # 完整事件闭环内（如 13.0 的赛后庆祝）不得成为边界
            assert decision is False, (run, raw_file.name, verdict)
    # p04 之后的 run 必须真的产出过功能枚举判定（防止夹具退化成空转）
    if run not in {"p6_v9_p03f2_20260918"}:
        assert saw_narrative_schema, run
