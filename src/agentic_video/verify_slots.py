"""槽级证据验证器（V1 P3）：看实际进成片的区间，回答结构化验证问题。

不是问"是否满足需求"（容易得到宽泛肯定），而是逐条核对 must_have：
结论(通过/不通过/不确定) + 各条件是否满足 + 证据在片段的什么时间 +
是否需要前后文 + 具体失败原因。看的是**实际准备放进成片的区间**；
为理解上下文额外看的部分帮助判断，但"观众看不到的源片情节"不算成片
已表达。需求"保护同伴"时看见挥刀不能直接通过——须定位受威胁对象、
介入动作，或可靠对白/前后文支持保护关系。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.template.schema import extract_json_block

VERIFICATION_PROMPT = """你是素材证据验证员。观看影片 {video} 的 {start:g}~{end:g} 秒区间
（这正是将被放进成片的片段，观众只能看到这段）。对照下列叙事需求逐条验证。

叙事需求：{need}
必须满足：{must_have}
明确排除：{must_not}
证据模式：{evidence_mode}

只输出 JSON：
{{"verdict":"pass|fail|uncertain",
"conditions":[{{"condition":"必须满足项原文","met":true,"evidence_interval":[片段内秒数,片段内秒数]}}],
"missing":["未满足的条件"],
"failure_reason":"不通过时的具体原因：缺什么证据、断在哪",
"needs_context":false,
"what_is_visible":"片段里实际可见的内容一句话"}}

判定纪律：
- 只依据片段内可见/可听内容；片段外剧情不能作为通过理由。
- 每条 met=true 都必须给 evidence_interval；给不出就 met=false。
- 无法判断（画面太暗/太快/被遮挡）→ verdict=uncertain，不要猜。
"""

VERIFICATION_PROMPT_VERSION = "verify_v1"


def _slot_window(slot: dict, *, pad_s: float = 0.0) -> tuple[float, float]:
    source = slot.get("source") or {}
    start = max(0.0, float(source.get("start_s") or 0) - pad_s)
    end = float(source.get("end_s") or 0) + pad_s
    return start, end


def parse_verification(raw: str) -> dict | None:
    block = extract_json_block(raw)
    try:
        payload = json.loads(block) if block else None
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or payload.get("verdict") not in {
            "pass", "fail", "uncertain"}:
        return None
    conditions = []
    for item in payload.get("conditions") or []:
        if isinstance(item, dict) and item.get("condition"):
            conditions.append({
                "condition": str(item["condition"]),
                "met": bool(item.get("met")),
                "evidence_interval": item.get("evidence_interval")
                if isinstance(item.get("evidence_interval"), list) else None,
            })
    # V4 E1 规整层（外审六节反例实锤：pass + met=false + missing 非空原样
    # 保留；met=true 无 evidence_interval 也通过）——由代码保证不变式，
    # 不信模型的自我汇报：
    #   ①met=true 必须带合法 evidence_interval（二元数值、start<end、非负），
    #     否则该条件 met=false；
    #   ②verdict=pass 但存在 met=false 或 missing 非空 → 降级 fail；
    #   ③fail 但 missing 与 failure_reason 皆空 → 补 unspecified（fail 仍 fail，
    #     但可行动）。
    for condition in conditions:
        interval = condition.get("evidence_interval")
        valid = (isinstance(interval, list) and len(interval) == 2
                 and all(isinstance(v, (int, float)) for v in interval)
                 and 0 <= float(interval[0]) < float(interval[1]))
        if condition["met"] and not valid:
            condition["met"] = False
            condition["evidence_interval"] = None
    unmet = [c for c in conditions if not c["met"]]
    missing = [str(value) for value in payload.get("missing") or []]
    verdict = payload["verdict"]
    failure_reason = str(payload.get("failure_reason") or "")
    if verdict == "pass" and (unmet or missing):
        verdict = "fail"
        failure_reason = (failure_reason + " | " if failure_reason else "") \
            + "normalized: pass with unmet conditions"
    if verdict == "fail" and not missing and not failure_reason:
        failure_reason = "unspecified"
    return {
        "verdict": verdict,
        "conditions": conditions,
        "missing": missing,
        "failure_reason": failure_reason,
        "needs_context": bool(payload.get("needs_context")),
        "what_is_visible": str(payload.get("what_is_visible") or "")[:120],
    }


def verify_slots(cfg, story_plan: dict, *, runner, slot_idxs=None,
                 context_pad_s: float = 5.0, clip_root=None) -> dict:
    """对 supported 槽逐个看实际区间验证。needs_context 时有界扩展重看一次。

    返回 {"results": [...], "failed_slots": [...], "uncertain_slots": [...]}，
    由 pipeline 决定是否触发 re_search（每轮限一次验证 pass）。
    clip_root：切片缓存目录（watch 带 start/end 必须给 clip_dir，否则
    cut_clip 里 None/"clip.mp4" 直接 TypeError——2026-09-11 C 档夜间首跑实锤）。
    """
    import tempfile

    results = []
    clip_root = Path(clip_root) if clip_root else Path(
        tempfile.mkdtemp(prefix="verify_slots_"))
    for slot in story_plan.get("slots") or []:
        idx = int(slot.get("slot_idx", 0))
        if slot_idxs is not None and idx not in set(slot_idxs):
            continue
        if slot.get("status") != "supported":
            continue
        source = slot.get("source") or {}
        video = Path(str(source.get("video") or ""))
        if not video.exists():
            results.append({"slot_idx": idx, "verdict": "uncertain",
                            "failure_reason": "source video missing"})
            continue
        spec = slot.get("need_spec") or {}
        start, end = _slot_window(slot)

        def _ask(start_s: float, end_s: float) -> dict | None:
            prompt = VERIFICATION_PROMPT.format(
                video=video.name, start=start_s, end=end_s,
                need=spec.get("need") or slot.get("role"),
                must_have="；".join(spec.get("must_have") or []),
                must_not="；".join(spec.get("must_not") or []),
                evidence_mode=spec.get("evidence_mode") or "visual")
            clip_dir = clip_root / f"slot_{idx:02d}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            answer = runner.watch(video, prompt, start_s=start_s, end_s=end_s,
                                  clip_dir=clip_dir, max_new_tokens=1024,
                                  duration_s=end_s - start_s)
            return parse_verification(answer.text)

        verdict = _ask(start, end)
        if verdict is None:
            verdict = {"verdict": "uncertain", "conditions": [], "missing": [],
                       "failure_reason": "verification parse failed",
                       "needs_context": False, "what_is_visible": ""}
        else:
            # V4 E1：证据区间必须是片段内秒数——模型报绝对时间/越界秒数时
            # 该条件打回 met=false（解析层不知片段时长，只能在这里判）
            clip_duration = end - start
            demoted = False
            for condition in verdict.get("conditions") or []:
                interval = condition.get("evidence_interval")
                if condition.get("met") and isinstance(interval, list) \
                        and len(interval) == 2:
                    if (float(interval[0]) >= clip_duration
                            or float(interval[1]) > clip_duration + 1.0):
                        condition["met"] = False
                        condition["evidence_interval"] = None
                        demoted = True
            if demoted:
                if verdict.get("verdict") == "pass":
                    verdict["verdict"] = "fail"
                verdict["failure_reason"] = (
                    str(verdict.get("failure_reason") or "")
                    + " | normalized: evidence outside clip")
            if verdict.get("needs_context"):
                # 有界上下文扩展：帮助判断，但判定仍以原区间为准
                widened = _ask(max(0.0, start - context_pad_s), end + context_pad_s)
                if widened is not None:
                    verdict["context_widened"] = True
                    verdict = widened
        verdict["slot_idx"] = idx
        results.append(verdict)
    return {
        "results": results,
        "failed_slots": [row["slot_idx"] for row in results
                         if row["verdict"] == "fail"],
        "uncertain_slots": [row["slot_idx"] for row in results
                            if row["verdict"] == "uncertain"],
        "prompt_version": VERIFICATION_PROMPT_VERSION,
    }


DETERMINISTIC_CHECK_VERSION = "det_check_v2"   # V4：外观别名退出等价+bindings 生效+cutaway 分类学


LOCALIZE_PROMPT = """你是素材证据定位员。观看影片 {video} 的 {start:g}~{end:g} 秒区间
（这是一个完整候选窗口，远长于成片槽位）。找出**实际承载下列叙事需求**的
连续片段——需求靠什么表达（可见动作/可听对白），片段就在哪里。

叙事需求：{need}

只输出 JSON：
{{"found": true, "start": 窗口内起始秒数, "end": 窗口内结束秒数,
"evidence": "该片段内可见/可听内容的一句话（写实，不外推）", "confidence": 0.0}}

判定纪律：
- 区间不得小于 2 秒，不得超出窗口范围。
- 只依据窗口内可见/可听内容定位；窗口内没有承载该需求的片段 → found=false，
  禁止"差不多在前半段"式的猜测定位。
"""

LOCALIZE_PROMPT_VERSION = "localize_v1"


def parse_localization(raw: str) -> dict | None:
    block = extract_json_block(raw)
    try:
        payload = json.loads(block) if block else None
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        return None
    try:
        confidence = min(1.0, max(0.0, float(payload.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    interval = payload.get("interval")
    if interval is None and payload.get("start") is not None:
        interval = [payload.get("start"), payload.get("end")]
    if not (isinstance(interval, list) and len(interval) == 2):
        interval = None
    return {
        "found": bool(payload.get("found")),
        "interval": [float(value) for value in interval]
        if interval and all(isinstance(v, (int, float)) for v in interval) else None,
        "evidence": str(payload.get("evidence") or "")[:160],
        "confidence": confidence,
    }


def localize_coarse_slots(cfg, story_plan: dict, *, runner,
                          output_dir=None) -> list[dict]:
    """V4 B4（外审三轮必改②）：coarse 槽的 localize-or-reject。

    无对白锚且远超预算（>2×budget）的候选窗口，禁止"从起点截前 N 秒"——
    让 Omni 在完整窗口内定位实际承载需求的片段；解析失败/未找到 → 槽降
    unsupported（必选槽 → plan_incomplete → delivery blocked），绝不盲切。
    返回逐槽日志（localize_log 落档供验收核对 -ss）。
    """
    import tempfile

    log: list[dict] = []
    clip_root = Path(tempfile.mkdtemp(prefix="localize_"))
    for slot in story_plan.get("slots") or []:
        idx = int(slot.get("slot_idx", 0))
        source = slot.get("source") or {}
        if slot.get("status") not in {"supported", "uncertain"} \
                or source.get("anchor") != "coarse":
            continue
        spec = slot.get("need_spec") or {}
        budget = (float(slot["target_interval"][1])
                  - float(slot["target_interval"][0]))
        win_start, win_end = float(source.get("start_s") or 0), float(source.get("end_s") or 0)
        video = Path(str(source.get("video") or ""))
        entry = {"slot_idx": idx, "window": [win_start, win_end], "budget_s": budget}
        if not video.exists():
            slot["status"], slot["reason"] = "unsupported", "coarse_unlocalizable(video_missing)"
            entry["outcome"] = "rejected"
            log.append(entry)
            continue
        prompt = LOCALIZE_PROMPT.format(video=video.name, start=win_start, end=win_end,
                                        need=spec.get("need") or slot.get("role"))
        clip_dir = clip_root / f"slot_{idx:02d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        try:
            answer = runner.watch(video, prompt, start_s=win_start, end_s=win_end,
                                  clip_dir=clip_dir, max_new_tokens=768,
                                  duration_s=win_end - win_start)
            parsed = parse_localization(answer.text)
        except Exception:                                     # noqa: BLE001 - 定位失败=拒绝
            parsed = None
        if parsed is None or not parsed["found"] or parsed["interval"] is None:
            slot["status"] = "unsupported"
            slot["reason"] = "coarse_unlocalizable(not_found)"
            entry["outcome"] = "rejected"
            entry["parsed"] = parsed
            log.append(entry)
            continue
        rel_start, rel_end = parsed["interval"]
        abs_start = min(win_end, max(win_start, win_start + rel_start))
        abs_end = min(win_end, abs_start + budget)
        if abs_end - abs_start < min(budget, 2.0) - 1e-6:
            slot["status"], slot["reason"] = "unsupported", "coarse_unlocalizable(too_short)"
            entry["outcome"] = "rejected"
            log.append(entry)
            continue
        source["start_s"], source["end_s"] = round(abs_start, 3), round(abs_end, 3)
        source["anchor"] = "localized"
        source["caption"] = parsed["evidence"] or source.get("caption") or ""
        entry.update({"outcome": "localized", "interval": [abs_start, abs_end],
                      "evidence": parsed["evidence"], "confidence": parsed["confidence"]})
        log.append(entry)
    if log and output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "localize_log.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


def deterministic_story_check(story_plan: dict, *, registry: dict | None = None) -> dict:
    """V3 P3：渲染前零模型全局检查——Local Correctness ≠ Narrative Coherence。

    lxh_p4_C2 病灶：三槽三个主角各自过槽级验证，成片却不知主角是谁。
    三指标（确定性，不依赖 critic）：Protagonist Switch Count（单主角必须 0）/
    Required-slot Protagonist Presence（100%）/ Unexplained Entity Transition
    Rate（0）。Slot0=小黑、Slot1=无限 在这里直接 FAIL，一分钱 Omni 不花。
    """
    from src.agentic_video.narrative_form import protagonist_required
    from src.library.entity_registry import load_entity_registry, row_identity_keys

    if registry is None:
        registry = load_entity_registry()
    contract = story_plan.get("entity_contract") or {}
    protagonist = contract.get("protagonist")
    live = [slot for slot in story_plan.get("slots") or []
            if slot.get("status") in {"supported", "uncertain"}
            and (slot.get("source") or {}).get("video")]
    presence: list[tuple[int, bool]] = []
    for slot in live:
        if not protagonist_required(slot.get("need_spec") or {}):
            continue
        keys = row_identity_keys(slot.get("source") or {}, registry)
        presence.append((int(slot["slot_idx"]),
                         (not protagonist) or protagonist in keys))
    missing = [idx for idx, ok in presence if not ok]
    # 主角中途缺席再回归 = 一次切换（present→absent→present）
    switch_count = 0
    seen_present = seen_absent = False
    for _idx, ok in presence:
        if ok:
            if seen_absent:
                switch_count += 1
            seen_present, seen_absent = True, False
        elif seen_present:
            seen_absent = True
    # V4 A4 转移分类学：cutaway = 自由槽带声明理由的切走（不门控，显式上报
    # 供人工/盲看复核）；unexplained 只门控主角必选槽的断链——身份键收紧后
    # 若不分类，每个合法配角镜头都会被判违规（外审三轮：配角镜头可以存在，
    # 但不能假装连续，也不能什么都不声明就切走）。
    cutaways = [{"slot_idx": int(slot["slot_idx"]),
                 "cutaway_function": (slot.get("need_spec") or {})
                 .get("cutaway_function")}
                for slot in live if slot.get("transition_reason") == "cutaway"]
    unexplained = [int(slot["slot_idx"]) for slot in live
                   if slot.get("transition_reason") == "unexplained"]
    reversals: list[list[int]] = []          # 信息项：同片时间倒流（倒叙须有理由）
    for left, right in zip(live, live[1:]):
        left_src, right_src = left.get("source") or {}, right.get("source") or {}
        if left_src.get("video") == right_src.get("video") \
                and float(right_src.get("start_s") or 0) \
                < float(left_src.get("start_s") or 0) - 1e-6:
            reversals.append([int(left["slot_idx"]), int(right["slot_idx"])])
    metrics = {
        "protagonist": protagonist,
        "protagonist_switch_count": switch_count,
        "required_protagonist_presence":
            round((len(presence) - len(missing)) / len(presence), 4)
            if presence else None,
        "unexplained_entity_transition_rate":
            round(len(unexplained) / max(1, len(live) - 1), 4),
        "cutaway_slots": cutaways,
        "time_reversals": reversals,
    }
    violations = []
    if switch_count:
        violations.append(f"protagonist_switch_count={switch_count}")
    if missing:
        violations.append(f"required_slots_missing_protagonist={missing}")
    if unexplained:
        violations.append(f"unexplained_entity_transitions={unexplained}")
    return {"version": DETERMINISTIC_CHECK_VERSION, "passed": not violations,
            "metrics": metrics, "violations": violations,
            "re_search_slots": sorted(set(missing) | set(unexplained))}


BLIND_VIDEO_PROMPT = """你是第一次观看这条短视频的观众，没有任何背景资料。
只根据画面和可听声音回答，不猜画面外剧情。只输出 JSON：
{"main_character":"这条视频的主要人物是谁（按可见外观描述）",
"consistent_protagonist":true,
"switch_points":[{"at_s":0.0,"what_changed":"无法解释的人物切换"}],
"story_in_one_sentence":"这条视频讲了什么",
"event_relations":"前后段事件的关系：延续/并列/无关，逐段说明"}
判定纪律：前后主要人物换成另一个人且没有转场理由，consistent_protagonist
写 false 并在 switch_points 给出大致时间；不确定也写 false 并说明原因。
画面内字幕/烧录文字属于视频内容，可作判断依据。"""

BLIND_VIDEO_PROMPT_VERSION = "blind_v2"   # V4：盲看输入改为终版（含烧录文案+终混音）


def blind_video_check(video_path, *, runner) -> dict:
    """V3 P3：渲染后盲看（零上下文）——只看成片回答主体一致性/故事性。

    不给 Story Plan / 文案 / 参考语境（V1 红线3：看过计划的模型自报答对
    不可信——lxh_p4_C2 的 comprehension 5/5 与 coherence 0 同文件自相矛盾）。
    """
    from src.perception.common import prepare_watch_copy

    answer = runner.watch(prepare_watch_copy(video_path), BLIND_VIDEO_PROMPT,
                          max_new_tokens=1024)
    block = extract_json_block(answer.text)
    try:
        payload = json.loads(block) if block else None
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        return {"parsed": False, "consistent_protagonist": None,
                "main_character": "", "switch_points": [],
                "story_in_one_sentence": "", "event_relations": "",
                "raw_head": str(answer.text)[:400],
                "prompt_version": BLIND_VIDEO_PROMPT_VERSION}
    consistent = payload.get("consistent_protagonist")
    return {
        "parsed": True,
        "main_character": str(payload.get("main_character") or ""),
        "consistent_protagonist": consistent if isinstance(consistent, bool) else None,
        "switch_points": [item for item in payload.get("switch_points") or []
                          if isinstance(item, dict)],
        "story_in_one_sentence": str(payload.get("story_in_one_sentence") or ""),
        "event_relations": str(payload.get("event_relations") or ""),
        "prompt_version": BLIND_VIDEO_PROMPT_VERSION,
    }
