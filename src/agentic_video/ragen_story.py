# -*- coding: utf-8 -*-
"""ReGen Module 2：Story Instantiation（双模式，p0523 修正 2/3）。

- **Creative Mode**：输入只有 Transfer Contract（代码剥离一切
  reference_specific，过严格输入门后才发）——Omni 自由创造 3 个全新故事。
  这是真正的抽象测试。
- **Asset-aware Mode**：输入 = Contract + 现有 dailies 的观察语义——为已有
  素材创造成立的新故事（take_001 走此路，S1/S3 补拍）。不冒充 clean 抽象。

门语义（修正 3）：**输入泄漏严格、输出泄漏从宽**——输出只拦逐字 quote /
reference_specific_fact 长串照抄；"比赛/运动/跆拳道"这类模型可独立想到的
词不算泄漏（强迫绕开参考反而毁掉创作测试）。

因果检查器（修正 1）：counter_evidence 必须与 initial_belief 同维度正反
命题、可视可观察；ending 不得取消修正后认知；reinforcement 逐条支持。
四 role 凑齐但 S2 不反驳 S1 的故事在这里被拦。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.ragen_director import (
    ReGenBlocked, TRANSFER_CONTRACT_VERSION)

STORY_PLANS_VERSION = "story_plans_v1"

# 输入门（严格）：payload 里不得出现 reference 特定词
INPUT_LEAK_TERMS = ("没有双手", "失去双手", "全国冠军", "taekwondo",
                    "handless", "no hands", "national champion")
# 输出门（从宽，修正 3）：只拦逐字/长串照抄，不拦可独立想到的领域词
_OUTPUT_COPY_MIN_CHARS = 15

STORY_INSTANTIATION_PROMPT = """你是故事创作者。输入是一份**叙事迁移契约**
（一条参考短片逆向归纳出的创作程序：主题、叙事因果约束、剪辑语法、可替换
槽位）。你的任务：只依据这份契约，自由创造 3 个**全新的故事**，每个故事都
用不同的方式满足同一套叙事因果约束。只输出一个 JSON 对象：
{"stories": [{
  "story_id": "story_a",
  "logline": "一句话故事",
  "initial_belief": {"statement": "观众/他人一开始对主人公的预期",
                     "dimension": "被低估的能力维度"},
  "counter_evidence": {"statement": "直接反驳该预期的可视事件",
                       "contradicts_dimension": "必须等于 initial_belief.dimension",
                       "visually_observable": true},
  "reinforcement": [{"statement": "继续支持修正后认知的证据",
                     "supports_corrected_belief": true}],
  "ending": {"statement": "收尾小限制/小反差", "function": "人格化幽默收束",
             "cancels_corrected_belief": false},
  "sections": [{"role": "契约中的段角色", "event": "这一段发生什么",
                "emotion": "观众此刻的感受"}],
  "emotion_arc": ["...", "...", "..."]}]}
要求：
1. 三个故事在人物身份、能力领域、场合、结尾反差上**互不相同**（填不同的
   槽位组合）；领域选择完全自由（运动/职业/生活技能/创作皆可）。
2. 每个故事的 counter_evidence 必须与 initial_belief 在同一维度上构成
   正面反驳（不能预期 A 却证明 B）。
3. sections 按契约的段角色顺序组织，每个角色恰好一段。
4. 全部内容必须可直接拍成视频（可视事件，不依赖旁白解释）。
输入契约："""

ASSET_STORY_PROMPT = """你是故事创作者。输入是一份**叙事迁移契约**和一段
**已有视频素材的观察语义**。你的任务：创造一个新故事，使已有素材正好充当
其中的核心反证段（counter_evidence），并补齐其余段落的事件描述。只输出一
个 JSON 对象（同 creative 模式的单故事 schema，story_id 固定
"asset_aware"，sections 中 counter_evidence 段的 event 字段写
"USE_EXISTING_FOOTAGE"）。要求：
1. 故事不得破坏契约的叙事因果约束（S1 建立的预期必须正好被已有素材反驳：
   已有素材展示的能力维度要填进 initial_belief.dimension）。
2. 补齐的段落（setup/reinforcement/ending 等）给出一句话事件描述即可，
   后续由生成模块展开。
输入契约与素材语义："""


def strip_to_transferable(contract: dict[str, Any]) -> dict[str, Any]:
    """输入门：只保留可下发字段；serialized payload 过严格禁词检查。"""
    whitelist = ("schema_version", "theme", "narrative_invariants",
                 "editing_grammar", "instantiation_slots", "rhythm_grammar",
                 "display_format", "section_roles")
    payload = {key: contract[key] for key in whitelist if key in contract}
    serialized = json.dumps(payload, ensure_ascii=False)
    for term in INPUT_LEAK_TERMS:
        if term in serialized:
            raise ReGenBlocked("story_input_gate", "input_leak_term", term)
    return payload


def _output_violations(story_text: str, annex: dict[str, Any]) -> list[str]:
    """输出门（从宽）：逐字 quote 或 reference_specific_fact 长串照抄。"""
    violations = []
    for row in annex.get("sections") or []:
        for key in ("hook_text_quote", "final_text_quote"):
            quote = str(row.get(key) or "").strip()
            if len(quote) >= 6 and quote in story_text:
                violations.append(f"verbatim_quote:{key}")
        fact = str(row.get("reference_specific_fact") or "").strip()
        # 长串特异组合照抄（滑窗 15+ 字符命中即视为复制参考事实链）
        for start in range(0, max(len(fact) - _OUTPUT_COPY_MIN_CHARS, 0) + 1):
            chunk = fact[start:start + _OUTPUT_COPY_MIN_CHARS]
            if chunk and chunk in story_text:
                violations.append("reference_fact_copy")
                break
    return violations


def validate_story_causal(story: dict[str, Any],
                          contract: dict[str, Any]) -> list[str]:
    """因果检查器（修正 1）：结构性验证，返回违规清单（空=过）。"""
    problems = []
    belief = story.get("initial_belief") or {}
    counter = story.get("counter_evidence") or {}
    dimension = str(belief.get("dimension") or "").strip()
    if not dimension:
        problems.append("initial_belief.dimension_missing")
    if str(counter.get("contradicts_dimension") or "").strip() != dimension:
        problems.append("counter_evidence_dimension_mismatch")
    if counter.get("visually_observable") is not True:
        problems.append("counter_evidence_not_visual")
    for index, row in enumerate(story.get("reinforcement") or []):
        if row.get("supports_corrected_belief") is not True:
            problems.append(f"reinforcement_{index}_not_supporting")
    ending = story.get("ending") or {}
    if ending.get("cancels_corrected_belief") is not False:
        problems.append("ending_cancels_corrected_belief")
    roles = [str(row.get("role")) for row in story.get("sections") or []]
    required = [str(role) for role in contract.get("section_roles") or []]
    if roles != required:
        problems.append(f"section_roles_mismatch:{roles}!={required}")
    return problems


def _parse_answer(raw: str) -> dict[str, Any]:
    from src.agentic_video.reference_program_v9 import _parse_one_object
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return _parse_one_object(text, stage="ragen_story")


def instantiate_stories(contract: dict[str, Any], output_dir: Path, *,
                        runner, annex: dict[str, Any],
                        mode: str = "creative",
                        dailies: list[dict[str, Any]] | None = None,
                        force: bool = False) -> dict[str, Any]:
    """creative：3 个全新故事；asset_aware：为已有素材造一个故事。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "story_plans.json"
    payload = strip_to_transferable(contract)
    prompt = STORY_INSTANTIATION_PROMPT
    if mode == "asset_aware":
        prompt = ASSET_STORY_PROMPT
        payload = {"contract": payload,
                   "available_dailies": dailies or []}
    input_hash = json_hash({"payload": payload, "mode": mode,
                            "prompt": prompt})
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            return cached
    answer = runner.ask(
        prompt + json.dumps(payload, ensure_ascii=False,
                            separators=(",", ":")),
        max_new_tokens=3072, stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    (output_dir / f"story_{mode}_raw.txt").write_text(raw, encoding="utf-8")
    value = _parse_answer(raw)
    stories = value.get("stories") or ([value] if mode == "asset_aware" else [])
    if not stories:
        raise ReGenBlocked("story", "no_stories_returned", mode)
    audited = []
    for story in stories:
        story_text = json.dumps(story, ensure_ascii=False)
        problems = validate_story_causal(story, contract)
        violations = _output_violations(story_text, annex)
        audited.append({"story": story, "causal_problems": problems,
                        "copy_violations": violations,
                        "passed": not problems and not violations})
    passing = [row for row in audited if row["passed"]]
    if mode == "creative" and not passing:
        raise ReGenBlocked(
            "story", "no_causally_valid_story",
            json.dumps([row["causal_problems"] + row["copy_violations"]
                        for row in audited], ensure_ascii=False)[:500])
    result = {"schema_version": STORY_PLANS_VERSION, "mode": mode,
              "input_sha256": input_hash, "stories": audited}
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
