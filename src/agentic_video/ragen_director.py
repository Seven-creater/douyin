# -*- coding: utf-8 -*-
"""ReGen Module 1：Reference → Transfer Contract（导演抽象，p0523 七项修正版）。

从冻结 P0 产物**逆向恢复创作程序**（Program Induction——本项目辨识度所在）：
- Narrative Causal Grammar（修正 1）：不是 theme+roles 标签，而是可执行因果
  结构——counter_evidence 必须直接反驳 initial_belief（domain 匹配），
  ending 不得取消修正后的认知。防"四 role 凑齐但故事不通"。
- Editing Functional Grammar（修正 4）：role→剪辑功能绑定（为什么这样剪），
  不只形式参数（whip-pan 是手段，"压缩完整事件为最小充分证据"才是语法）。
- Rhythm Grammar（修正 7）：P0 节拍/切点数据确定性推出相对节奏结构——
  迁移节奏不迁移音乐。
- Instantiation Slots + display_format（修正 5）：人物/职业/比赛等=自由创造
  区；成片画布 9:16 竖版（撤横屏默认），reference 具体实现只进 annex。

防泄露：DIRECTOR_CONTRACT_PROMPT 为中性模板（零参考词，进 spoiler 测试）；
导演层合法阅读参考理解（它的职责就是理解参考），剥离发生在下游 creative
模式的输入门。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash

TRANSFER_CONTRACT_VERSION = "transfer_contract_v1"

DIRECTOR_CONTRACT_PROMPT = """你是导演学者。输入是一部参考短片的**结构化理解
产物**（每段的叙事功能、可迁移结构、认知变化、剪辑形态、节奏数据）。你的任务
不是描述它讲了什么，而是逆向恢复它的**创作程序**——为什么这样讲故事、为什么
这样剪。只输出一个 JSON 对象：
{"theme": "一句话：这条片子用什么因果机制打动观众（不是内容概要）",
 "narrative_invariants": {
   "initial_belief": {"type": "underestimation",
     "belief_predicate": "the protagonist is not capable of X",
     "capability_dimension": "X（抽象维度名，2-4 词）",
     "source_of_underestimation": "free_instantiation_slot"},
   "counter_evidence": {"required_relation": "direct_negation",
     "must_target_same_capability_dimension": true,
     "must_be_visually_observable": true, "must_be_decisive": true},
   "reinforcement": {"must_support_revised_belief": true,
     "may_use_other_events": true},
   "ending": {"function": "humanizing_contrast",
     "must_not_reverse_revised_belief": true}},
 "editing_grammar": [
   {"role": "...", "editing_intent": "...", "content_pattern": ["..."],
    "shot_density": "low|medium|medium_high|very_high",
    "transition_policy": "...", "ordering": "causal|...",
    "timing_function": "..."}],
 "instantiation_slots": ["可自由替换的具体化维度，如主角身份/能力领域/场合"]}
要求：
1. narrative_invariants 是**谓词级因果结构**（不是标签清单）：
   initial_belief 必须写成 belief_predicate（"the protagonist is not
   capable of X" 形式）+ capability_dimension（X = 抽象能力维度名，如
   "physical competence"/"professional skill"，2-4 词）+
   source_of_underestimation（填 "free_instantiation_slot"——低估的
   来源是自由槽位：外表/年龄/资历/身份/身体条件都只是实例）。
   counter_evidence 必须 required_relation="direct_negation" 且
   must_target_same_capability_dimension=true。ending 只人格化不翻案。
2. **抽象禁令**：capability_dimension 与 theme 不得携带任何具体实例色彩
   （身体条件/残疾/具体运动/具体职业全禁止）——参考片用什么实例实现
   这条因果链与契约无关，那是 instantiation_slots 的事。
3. editing_grammar 每条回答"这一段为什么用这种剪法"（editing_intent）；
   content_pattern 用**通用证据类别**（如 different_context_evidence /
   decisive_action / visible_result / humanizing_detail），不得罗列
   参考片后段具体出现的内容类型。
4. 所有描述必须可迁移：换成任何题材/人物，同一套约束仍然成立且可用。
输入："""


def _bucket_density(cuts_per_s: float) -> str:
    if cuts_per_s < 0.15:
        return "low"
    if cuts_per_s < 0.4:
        return "medium"
    if cuts_per_s < 0.8:
        return "medium_high"
    return "very_high"


class ReGenBlocked(RuntimeError):
    def __init__(self, stage: str, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{stage}:{reason_code}:{detail}")
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail


def compute_rhythm_grammar(content: dict[str, Any],
                           ledger: dict[str, Any]) -> dict[str, Any]:
    """修正 7：从 P0 切点/时长数据确定性推出相对节奏结构（不迁移音乐）。"""
    sections = content.get("sections") or []
    total = sum(float(row["interval"][1]) - float(row["interval"][0])
                for row in sections) or 1.0
    cut_pts = [float(row.get("pts_s") or 0.0) for row in
               (ledger.get("scene_detection") or {}).get("cut_candidates") or []]
    ratios, densities, spans = [], [], []
    for row in sections:
        start, end = float(row["interval"][0]), float(row["interval"][1])
        span = max(end - start, 0.01)
        ratios.append(round(span / total, 3))
        cuts_inside = sum(1 for pts in cut_pts if start < pts < end)
        densities.append(_bucket_density(cuts_inside / span))
        spans.append(round(span, 2))
    # p0524 修正 7：节奏是**先验区间**不是精确硬约束（复制秒数=复刻路线）
    roles = [str(row.get("content_function")) for row in sections]
    rhythm_prior = {
        role: {"target": ratio,
               "range": [round(max(0.0, ratio - 0.055), 3),
                         round(ratio + 0.065, 3)]}
        for role, ratio in zip(roles, ratios)}
    return {
        "section_duration_s": spans,
        "section_duration_ratios": ratios,
        "rhythm_prior": rhythm_prior,
        "relative_cut_density": densities,
        "punchline_release": bool(
            sections and roles[-1] in
            {"evidence_expansion", "punchline_payoff"}),
        "beat_sync_strength": "preferred",
        "policy": "迁移相对节奏结构（先验区间）；新 BGM 的 beat timeline "
                  "映射到相同模式，不复制精确秒数",
    }


def build_director_payload(content: dict[str, Any],
                           edit_program: dict[str, Any]) -> list[dict[str, Any]]:
    """每段的结构化理解（导演合法阅读参考理解；剥离发生在下游 creative 门）。"""
    patterns = {str(row.get("section_id")): row
                for row in edit_program.get("editorial_patterns") or []}
    payload = []
    for row in content.get("sections") or []:
        pattern = patterns.get(str(row.get("section_id"))) or {}
        payload.append({
            "role": row.get("content_function"),
            "transferable_structure": row.get("transferable_structure"),
            "cognition_change": row.get("cognition_change"),
            "editing_form": {
                "composition_mode": pattern.get("composition_mode"),
                "semantic_phases": pattern.get("semantic_phases"),
                "snippet_count_range": pattern.get("snippet_count_range"),
                "ordering_constraint": pattern.get("ordering_constraint")},
            "section_text_role": row.get("hook_text_quote") and "hook"
            or (row.get("final_text_quote") and "punchline" or None),
        })
    return payload


# 抽象禁令（p0524）：维度名/theme 不得携带参考实例色彩——上一轮三候选全
# 残障题材的根因即"身体完整性"渗进了本该抽象的 capability_dimension。
DIMENSION_INSTANCE_BAN = ("身体", "残疾", "残障", "肢体", "听力", "视力",
                          "轮椅", "失聪", "失明", "body integrity", "disab",
                          "hand", "arm", "limb", "deaf", "blind")


def validate_contract(contract: dict[str, Any],
                      section_roles: list[str]) -> None:
    invariants = contract.get("narrative_invariants") or {}
    for key in ("initial_belief", "counter_evidence", "reinforcement",
                "ending"):
        if not isinstance(invariants.get(key), dict) or not invariants[key]:
            raise ReGenBlocked("director", "invariant_missing", key)
    belief = invariants["initial_belief"]
    for key in ("belief_predicate", "capability_dimension"):
        if not str(belief.get(key) or "").strip():
            raise ReGenBlocked("director", "invariant_field_missing", key)
    dimension = str(belief.get("capability_dimension") or "").lower()
    theme_text = str(contract.get("theme") or "").lower()
    for banned in DIMENSION_INSTANCE_BAN:
        if banned in dimension or banned in theme_text:
            raise ReGenBlocked(
                "director", "dimension_not_abstract",
                f"'{banned}' leaked an instantiation into the transferable "
                f"dimension/theme; body/appearance specifics belong to "
                f"instantiation slots, not contract invariants")
    counter = invariants["counter_evidence"]
    if (counter.get("required_relation") != "direct_negation" or
            counter.get("must_target_same_capability_dimension") is not True or
            counter.get("must_be_visually_observable") is not True or
            counter.get("must_be_decisive") is not True):
        raise ReGenBlocked("director", "invariant_logic_weak", "counter_evidence")
    if invariants["ending"].get("must_not_reverse_revised_belief") is not True:
        raise ReGenBlocked("director", "invariant_logic_weak", "ending")
    if not str(contract.get("theme") or "").strip():
        raise ReGenBlocked("director", "theme_missing")
    grammar = contract.get("editing_grammar") or []
    covered = {str(row.get("role")) for row in grammar if isinstance(row, dict)}
    missing = {role for role in section_roles if role not in covered}
    if missing:
        raise ReGenBlocked("director", "editing_grammar_role_missing",
                           ",".join(sorted(missing)))
    for row in grammar:
        if not str(row.get("editing_intent") or "").strip() or \
                not isinstance(row.get("content_pattern"), list) or \
                not row.get("content_pattern"):
            raise ReGenBlocked("director", "editing_grammar_field_missing",
                               str(row.get("role")))
    if not contract.get("instantiation_slots"):
        raise ReGenBlocked("director", "instantiation_slots_missing")


def build_transfer_contract(p04e_dir: Path, output_dir: Path, *,
                            runner, force: bool = False
                            ) -> dict[str, Any]:
    """冻结 P0 → transfer_contract.json（可下发）+ reference_annex.json（封存）。"""
    p04e_dir = Path(p04e_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "transfer_contract.json"
    content = json.loads(
        (p04e_dir / "reference_content_program.json").read_text(encoding="utf-8"))
    edit_program = json.loads(
        (p04e_dir / "reference_edit_program.json").read_text(encoding="utf-8"))
    ledger = json.loads(
        (p04e_dir / "reference_evidence.json").read_text(encoding="utf-8"))
    payload = build_director_payload(content, edit_program)
    section_roles = [str(row.get("role")) for row in payload]
    input_hash = json_hash({"payload": payload,
                            "prompt": DIRECTOR_CONTRACT_PROMPT})
    if contract_path.is_file() and not force:
        cached = json.loads(contract_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            return cached
    answer = runner.ask(
        DIRECTOR_CONTRACT_PROMPT + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")),
        max_new_tokens=2048, stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    (output_dir / "director_raw.txt").write_text(raw, encoding="utf-8")
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    from src.agentic_video.reference_program_v9 import _parse_one_object
    abstract = _parse_one_object(text, stage="ragen_director")
    validate_contract(abstract, section_roles)
    rhythm = compute_rhythm_grammar(content, ledger)
    contract = {
        "schema_version": TRANSFER_CONTRACT_VERSION,
        "input_sha256": input_hash,
        "theme": abstract["theme"],
        "narrative_invariants": abstract["narrative_invariants"],
        "editing_grammar": abstract["editing_grammar"],
        "instantiation_slots": abstract["instantiation_slots"],
        "rhythm_grammar": rhythm,
        "display_format": {
            "target_canvas": "9:16",
            "active_picture_policy": "fit_or_crop_preserving_subject",
            "reference_active_ratio": "3:4",
            "note": "竖版短视频画布（p0523 修正 5：撤横屏默认）"},
        "section_roles": section_roles,
    }
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=1), encoding="utf-8")
    annex = {
        "schema_version": "reference_annex_v1",
        "policy": "reference_specific 封存：不进入 creative 模式输入",
        "core_expression": content.get("core_expression"),
        "sections": [
            {"section_id": row.get("section_id"),
             "reference_specific_fact": row.get("reference_specific_fact"),
             "hook_text_quote": row.get("hook_text_quote"),
             "final_text_quote": row.get("final_text_quote")}
            for row in content.get("sections") or []],
    }
    (output_dir / "reference_annex.json").write_text(
        json.dumps(annex, ensure_ascii=False, indent=1), encoding="utf-8")
    return contract
