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
  "initial_belief": {"claim": "他人/观众的具体判断（含低估来源）",
                     "capability": "本故事的**具体**能力领域（如编程/烹饪/攀岩/急救——是契约抽象维度的一个实例，不是抽象维度本身）",
                     "source_of_underestimation": "低估来源（外表/年龄/资历/身份/经验等）"},
  "counter_evidence": {"demonstrated_capability": "必须等于 initial_belief.capability",
                       "event": "直接证明该能力的可视事件",
                       "outcome": "决定性结果"},
  "reinforcement": [{"event": "继续支持修正后认知的证据",
                     "supports_corrected_belief": true}],
  "ending": {"statement": "收尾小限制/小反差", "function": "人格化幽默收束",
             "cancels_corrected_belief": false},
  "sections": [{"role": "契约中的段角色", "event": "这一段发生什么",
                "emotion": "观众此刻的感受"}],
  "emotion_arc": ["...", "...", "..."]}]}
要求：
1. **多样性硬约束**：三个故事必须在 source_of_underestimation（低估来源）、
   capability（**具体**能力领域——三个故事必须选三个不同领域，禁止都填
   契约里的抽象维度原文）、社会语境三方面都不同——不是换皮同一机制。
2. 每个故事的 counter_evidence.demonstrated_capability 必须与
   initial_belief.capability 是同一维度（预期 A 就证明 A，不能预期 A 证明 B）。
3. sections 按契约的段角色顺序组织，每个角色恰好一段。
4. 全部内容必须可直接拍成视频（可视事件，不依赖旁白解释）。
输入契约："""

ASSET_STORY_PROMPT = """你是故事创作者。输入是一份**叙事迁移契约**和一段
**已有视频素材的客观观察语义**（不含任何原故事文本）。已知该素材将充当
故事的核心反证段（counter_evidence/S2）。你的任务：创造 **3 个互不相同**
的完整故事外壳，每个外壳为这段素材设计一个新的 S1 前提（被这段素材直接
推翻）和一个新的 S3（继续支持修正后的认知并以轻松方式收尾）。只输出一个
JSON 对象：
{"stories": [{
  "story_id": "wrapper_a",
  "logline": "一句话故事",
  "initial_belief": {"claim": "他人对主人公的具体判断（含低估来源）",
                     "capability": "被低估的具体能力领域",
                     "source_of_underestimation": "低估来源"},
  "counter_evidence": {"demonstrated_capability": "素材展示的具体能力领域",
                       "event": "USE_EXISTING_FOOTAGE",
                       "outcome": "素材的决定性结果"},
  "reinforcement": [{"event": "S3 的补充证据事件",
                     "supports_corrected_belief": true}],
  "ending": {"statement": "S3 收尾小反差", "function": "人格化幽默收束",
             "cancels_corrected_belief": false},
  "sections": [
    {"role": "situation_setup", "event": "S1 可视事件（4-6 秒能看清）",
     "emotion": "观众此刻的感受"},
    {"role": "counter_evidence", "event": "USE_EXISTING_FOOTAGE",
     "emotion": "惊讶"},
    {"role": "evidence_expansion", "event": "S3 可视事件（4-6 秒能看清）",
     "emotion": "认可与轻松"}],
  "emotion_arc": ["...", "...", "..."]}]}
注意：每个故事是完整 JSON 对象（含 initial_belief/counter_evidence 等
字段），不是按 S1/S2/S3 拆开的段落字典。其中：
  demonstrated_capability 填素材实际展示的具体能力领域
- 三个外壳的 source_of_underestimation 与 S3 的证据类型必须互不相同
- S1/S3 的事件必须是**可拍成视频的可视事件**（4-6 秒能看清）
要求：
1. 每个外壳的 S1 前提必须正好被已有素材**直接推翻**（同一能力维度）；
   judge 会检查这一点。
2. 不得破坏契约的叙事因果约束；ending 只人格化不翻案。
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
    """因果检查器（p0524 升级）：谓词级结构验证，返回违规清单（空=过）。

    字段相等是必要不充分条件——语义级"E 是否直接否定 P"由
    judge_direct_negation 的文本 judge 补足（廉价 ask，不看视频）。
    """
    problems = []
    belief = story.get("initial_belief") or {}
    counter = story.get("counter_evidence") or {}
    capability = str(belief.get("capability") or "").strip()
    if not capability or not str(belief.get("claim") or "").strip():
        problems.append("initial_belief_predicate_missing")
    if not str(belief.get("source_of_underestimation") or "").strip():
        problems.append("source_of_underestimation_missing")
    if str(counter.get("demonstrated_capability") or "").strip() != capability:
        problems.append("counter_evidence_capability_mismatch")
    if not str(counter.get("event") or "").strip() or not str(
            counter.get("outcome") or "").strip():
        problems.append("counter_evidence_not_visual_decisive")
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


NEGATION_JUDGE_PROMPT = """只做逻辑判定。前提 P 是他人对主人公的判断；
证据事件 E 是视频中可见发生的事。问题：若 E 为真且清晰可见，E 是否
**直接否定** P（同一能力维度上的正面反驳）？只输出一个 JSON 对象：
{"directly_contradicts": "yes|no|partially",
 "capability_match": true,
 "reason": "一句话"}
判定 yes 需：E 展示的正是 P 声称缺乏的同一能力。预期 A 证明 B = no。
前提与证据："""


def judge_direct_negation(story: dict[str, Any], *, runner) -> str:
    """文本 judge（p0524 修正 2）：E 是否直接否定 P——结构门管不住的
    语义级反驳判定，一次廉价 ask，不看视频。"""
    belief = story.get("initial_belief") or {}
    counter = story.get("counter_evidence") or {}
    payload = json.dumps(
        {"P": {"claim": belief.get("claim"),
               "capability": belief.get("capability")},
         "E": {"event": counter.get("event"),
               "demonstrated_capability":
                   counter.get("demonstrated_capability"),
               "outcome": counter.get("outcome")}},
        ensure_ascii=False, separators=(",", ":"))
    answer = runner.ask(NEGATION_JUDGE_PROMPT + payload,
                        max_new_tokens=512,
                        stop_after_json_object=True)
    raw = getattr(answer, "text", str(answer))
    from src.agentic_video.reference_program_v9 import _parse_one_object
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    value = _parse_one_object(text, stage="ragen_story_judge")
    verdict = str(value.get("directly_contradicts") or "")
    if verdict not in {"yes", "no", "partially"}:
        raise ReGenBlocked("story_judge", "judge_verdict_invalid", verdict)
    return verdict


def validate_diversity(stories: list[dict[str, Any]]) -> list[str]:
    """p0524 修正 3：候选须在低估来源/能力维度上真正不同。"""
    problems = []
    if len(stories) < 3:
        return problems

    def _values(key):
        return [str((s.get("initial_belief") or {}).get(key) or "").strip()
                .lower() for s in stories]

    for key, label in (("source_of_underestimation", "underestimation_source"),
                       ("capability", "capability_domain")):
        values = _values(key)
        if len(set(values)) < 3:
            problems.append(f"candidates_homogeneous_{label}:{values}")
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
    stories = value.get("stories") or [value]
    if not stories:
        raise ReGenBlocked("story", "no_stories_returned", mode)
    diversity_problems = (validate_diversity(stories)
                          if mode == "creative" else [])
    audited = []
    for story in stories:
        story_text = json.dumps(story, ensure_ascii=False)
        problems = validate_story_causal(story, contract)
        violations = _output_violations(story_text, annex)
        negation = None
        if not problems:
            negation = judge_direct_negation(story, runner=runner)
            if negation != "yes":
                problems.append(f"negation_judge_{negation}")
        audited.append({"story": story, "causal_problems": problems,
                        "copy_violations": violations,
                        "negation_judge": negation,
                        "passed": not problems and not violations})
    passing = [row for row in audited if row["passed"]]
    if mode == "creative" and not passing:
        raise ReGenBlocked(
            "story", "no_causally_valid_story",
            json.dumps([row["causal_problems"] + row["copy_violations"]
                        for row in audited], ensure_ascii=False)[:500])
    selected = None
    if mode == "asset_aware" and passing:
        selected = passing[0]["story"]["story_id"]
        for row in audited:
            row["selected"] = row["story"]["story_id"] == selected
    result = {"schema_version": STORY_PLANS_VERSION, "mode": mode,
              "input_sha256": input_hash,
              "diversity_problems": diversity_problems,
              "selected_story_id": selected,
              "stories": audited}
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
