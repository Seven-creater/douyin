"""Evidence-first discovery of meaningful popular Douyin references."""
from __future__ import annotations

import json
from datetime import date as date_type, timedelta
from pathlib import Path
from typing import Iterable

from src.template.schema import extract_json_block

CONTENT_CATEGORIES = ("real_story", "screen_story", "growth_story")
CATEGORY_LABELS = {
    "real_story": "真实人物/救助/暖心",
    "screen_story": "影视/动漫台词叙事",
    "growth_story": "成长/励志/人物经历",
}

_PURE_EDIT_WORDS = ("卡点技术流", "运镜教学", "转场教学", "纯卡点", "踩点教程",
                    "剪映教程", "舞蹈挑战", "变装卡点", "教学", "教程", "挑战", "抽象")
_AD_WORDS = ("直播间", "下单", "同款链接", "优惠券", "招生", "培训报名")
_CATEGORY_WORDS = {
    "real_story": ("救助", "拯救", "暖心", "家人", "家庭", "宝宝", "小猫", "小狗",
                   "应急", "迫降", "真实故事", "善意"),
    "screen_story": ("台词混剪", "电影", "影视", "动漫", "剧情", "角色", "名场面",
                     "电视剧", "综艺", "短剧"),
    "growth_story": ("成长", "励志", "坚持", "梦想", "经历", "从小", "多年后",
                     "自律", "人生", "蜕变", "青春"),
}


def infer_category(title: str) -> str | None:
    title = (title or "").lower()
    scored = [(sum(word.lower() in title for word in words), category)
              for category, words in _CATEGORY_WORDS.items()]
    score, category = max(scored, key=lambda item: (item[0], -CONTENT_CATEGORIES.index(item[1])))
    return category if score else None


def metadata_triage(record: dict) -> dict:
    """Cheap, transparent gate. It only removes obvious non-video/non-story candidates."""
    title = str(record.get("title") or "")
    extras = record.get("extras") or {}
    duration_s = float(extras.get("duration_ms") or 0) / 1000.0
    reasons = []
    if extras.get("media_type") != 4:
        reasons.append("not_video")
    if duration_s < 12 or duration_s > 180:
        reasons.append("duration_outside_12_180s")
    if any(word in title for word in _PURE_EDIT_WORDS):
        reasons.append("pure_edit_or_tutorial")
    if any(word in title for word in _AD_WORDS):
        reasons.append("commercial_or_promotion")
    return {
        **record,
        "duration_s": round(duration_s, 3),
        "category_hint": infer_category(title),
        "eligible": not reasons,
        "reasons": reasons,
    }


def has_temporal_story_coverage(row: dict) -> bool:
    """Require located evidence to span a beginning and an actual ending."""
    try:
        duration = float(row.get("duration_s") or 0)
    except (TypeError, ValueError):
        duration = 0
    if duration <= 0:
        return True
    intervals = []
    for event in row.get("events") or []:
        try:
            start, end = float(event.get("start_s")), float(event.get("end_s"))
        except (AttributeError, TypeError, ValueError):
            continue
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return False
    first = min(start for start, _ in intervals)
    last = max(end for _, end in intervals)
    return first <= duration * 0.30 and last >= duration * 0.70 \
        and last - first >= duration * 0.50


def passes_content_gate(row: dict) -> bool:
    return bool(
        row.get("eligible", True)
        and row.get("category") in CONTENT_CATEGORIES
        and int(row.get("event_count") or 0) >= 3
        and row.get("has_goal_or_causality") is True
        and row.get("has_resolution") is True
        and float(row.get("evidence_coverage") or 0) >= 0.80
        and float(row.get("story_clarity") or 0) >= 0.70
        and float(row.get("confidence") or 0) >= 0.65
        and row.get("pure_sensory") is False
        and has_temporal_story_coverage(row)
    )


def _popularity_key(row: dict) -> tuple:
    trend = row.get("trend") or {}
    lists = trend.get("lists") or row.get("appeared_in_lists") or []
    ranks = trend.get("rank") or row.get("ranks") or {}
    stats = row.get("stats") or {}
    best_rank = min((int(rank) for rank in ranks.values()), default=10**6)
    return (-len(lists), best_rank, -int(stats.get("likes") or 0), str(row.get("aweme_id")))


def _audition_key(row: dict) -> tuple:
    """Prefer a fresh signed download URL, then use the normal popularity rank."""
    dates = [str(value) for value in row.get("observed_dates") or []]
    latest = max(dates, default="0000-00-00")
    freshness = -int(latest.replace("-", "")) if latest[:4].isdigit() else 0
    return (freshness, *_popularity_key(row))


def select_balanced(audits: Iterable[dict], *, per_category: int = 4
                    ) -> tuple[list[dict], dict[str, int]]:
    """Apply the hard content gate, then popularity-rank only within each category."""
    accepted = [dict(row) for row in audits if passes_content_gate(row)]
    selected = []
    gaps = {}
    for category in CONTENT_CATEGORIES:
        rows = sorted((row for row in accepted if row.get("category") == category),
                      key=_popularity_key)
        picked = rows[:per_category]
        selected.extend(picked)
        gaps[category] = max(0, per_category - len(picked))
    return selected, gaps


def build_audition_shortlist(triaged: list[dict], *, per_category: int,
                             unclassified: int = 12) -> list[dict]:
    """Keep category leads plus popular unknowns for semantic audition.

    Title keywords are only a cheap prioritisation signal. Unknown titles must still
    reach semantic audition so the metadata gate cannot silently become a content
    classifier.
    """
    shortlist = []
    for category in CONTENT_CATEGORIES:
        rows = [row for row in triaged
                if row.get("eligible") and row.get("category_hint") == category]
        shortlist.extend(sorted(rows, key=_audition_key)[:max(per_category, 6)])
    unknown = [row for row in triaged
               if row.get("eligible") and row.get("category_hint") is None]
    shortlist.extend(sorted(unknown, key=_audition_key)[:max(0, unclassified)])
    result, seen = [], set()
    for row in shortlist:
        aweme_id = str(row["aweme_id"])
        if aweme_id not in seen:
            seen.add(aweme_id)
            result.append(row)
    return result


SEMANTIC_AUDIT_PROMPT = """你是短视频内容审查 Agent。请结合视频音画与下方确定性材料，判断它是否包含
观众可以复述的完整内容，而不是只靠音乐、快速切镜或特效制造刺激。

只输出一个 JSON 对象：
{"category":"real_story|screen_story|growth_story|other",
 "summary":"谁、遇到什么、做了什么、结果如何",
 "event_count":3,
 "events":[{"start_s":0.0,"end_s":1.0,"description":"可见事件","evidence_sources":["frame"]}],
 "has_goal_or_causality":true,"causal_summary":"原因→行动→结果",
 "has_resolution":true,"message":"表达的意义",
 "evidence_coverage":0.0,"story_clarity":0.0,"confidence":0.0,
 "pure_sensory":false,"rejection_reasons":[]}

event_count 只计算能定位时间且有材料支持的事件。没有证据时必须填 false/低分，不得根据标题补剧情。
事件时间必须使用原视频坐标，并覆盖开头、发展和真正结尾；不得把前几秒拆成三个事件冒充完整故事。

【确定性材料】
{context}
"""


def parse_semantic_audit(raw: str, record: dict) -> dict:
    block = extract_json_block(raw)
    parsed = {}
    if block:
        try:
            value = json.loads(block)
            parsed = value if isinstance(value, dict) else {}
        except ValueError:
            parsed = {}
    category = parsed.get("category")
    if category not in CONTENT_CATEGORIES:
        category = record.get("category_hint")
    row = {**record, **parsed, "category": category}
    # Preserve model-provided rejection reasons, but never allow an audit
    # inconsistency to pass the content gate.  The reported event count is a
    # claim; the evidence-backed list below is the only count we trust.
    rejection_reasons = [str(reason) for reason in (row.get("rejection_reasons") or [])]
    evidence_events = []
    for event in parsed.get("events") or []:
        if not isinstance(event, dict):
            continue
        try:
            start_s, end_s = float(event.get("start_s")), float(event.get("end_s"))
        except (TypeError, ValueError):
            continue
        if end_s <= start_s or not event.get("evidence_sources"):
            continue
        evidence_events.append({**event, "start_s": start_s, "end_s": end_s})
    row["events"] = evidence_events
    reported_count = parsed.get("event_count")
    row["event_count"] = len(evidence_events)
    count_mismatch = False
    if reported_count is not None:
        try:
            if int(reported_count) != len(evidence_events):
                count_mismatch = True
                rejection_reasons.append("event_count_mismatch")
        except (TypeError, ValueError):
            count_mismatch = True
            rejection_reasons.append("event_count_invalid")
    row["rejection_reasons"] = list(dict.fromkeys(rejection_reasons))
    if not has_temporal_story_coverage(row):
        row["rejection_reasons"] = list(dict.fromkeys(
            [*row["rejection_reasons"], "insufficient_temporal_story_coverage"]))
    row["eligible"] = (bool(record.get("eligible", True)) and not count_mismatch
                        and passes_content_gate({**row, "eligible": True}))
    if not block:
        row["rejection_reasons"] = list(dict.fromkeys(
            [*row["rejection_reasons"], "semantic_audit_parse_failed"]))
        row["eligible"] = False
    return row


def revalidate_semantic_audit(audit: dict, triage: dict) -> dict:
    """Apply the current deterministic gates to a cached model audition."""
    row = {**audit}
    for key in ("duration_s", "category_hint", "reasons"):
        if key in triage:
            row[key] = triage[key]
    row["metadata_eligible"] = bool(triage.get("eligible", True))
    rejection_reasons = [str(reason) for reason in row.get("rejection_reasons") or []]
    if not has_temporal_story_coverage(row):
        rejection_reasons.append("insufficient_temporal_story_coverage")
    row["rejection_reasons"] = list(dict.fromkeys(rejection_reasons))
    row["eligible"] = (row["metadata_eligible"]
                       and passes_content_gate({**row, "eligible": True}))
    return row


def write_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_selection_report(path: Path, selected: list[dict], gaps: dict[str, int],
                           *, candidate_count: int, audited_count: int) -> Path:
    lines = ["# 有内容的热门参考视频筛选", "",
             f"- 候选：{candidate_count}", f"- 完成语义审看：{audited_count}",
             f"- 通过硬门槛：{len(selected)}", "",
             "## 分类结果", ""]
    for category in CONTENT_CATEGORIES:
        lines.append(f"### {CATEGORY_LABELS[category]}")
        lines.append("")
        rows = [row for row in selected if row.get("category") == category]
        if not rows:
            lines.append(f"- 暂无合格视频；缺口 {gaps.get(category, 0)} 条。")
        for row in rows:
            lines.append(f"- `{row.get('aweme_id')}` {row.get('title') or '无标题'} — "
                         f"事件 {row.get('event_count')}，证据覆盖 "
                         f"{float(row.get('evidence_coverage') or 0):.0%}，"
                         f"清晰度 {float(row.get('story_clarity') or 0):.2f}")
        lines.append("")
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return Path(path)


def load_rolling_candidates(cfg, *, end_date: str, days: int) -> tuple[list[dict], dict[str, object]]:
    from src.trend.deduplicate import merge_videos, to_jsonl_record
    from src.trend.parser import parse_file
    from src.trend.ranking import rank_videos

    end = date_type.fromisoformat(end_date)
    records = []
    dates_used = []
    freshest_by_id = {}
    observed_dates: dict[str, set[str]] = {}
    for offset in range(max(1, days)):
        day = (end - timedelta(days=offset)).isoformat()
        paths = sorted(cfg.paths.raw_dir.glob(f"{day}_*.json"))
        if paths:
            dates_used.append(day)
        for path in paths:
            parsed = parse_file(path, date_window=int(
                cfg.wellbyte.request_params.get("date_window", 24)))
            for record in parsed:
                freshest_by_id.setdefault(record.aweme_id, record)
                observed_dates.setdefault(record.aweme_id, set()).add(day)
            records.extend(parsed)
    merged = merge_videos(records)
    # Ranking aggregates all seven days, while expiring CDN URLs must always
    # come from the newest observation of a work.
    for item in merged:
        fresh = freshest_by_id[item.aweme_id]
        item.record.download_url = fresh.download_url
        item.record.raw = fresh.raw
    ranked = rank_videos(merged, top_n=len(merged))
    rows = []
    for item in ranked:
        row = to_jsonl_record(item.merged, final_rank=item.final_rank)
        row["observed_dates"] = sorted(observed_dates.get(item.merged.aweme_id, set()),
                                       reverse=True)
        row["download_url_observed_date"] = row["observed_dates"][0]
        rows.append(row)
    return rows, {"dates_used": dates_used, "merged": merged,
                  "merged_by_id": {item.aweme_id: item for item in merged}}


def _audition_context(cfg, aweme_id: str, record: dict) -> str:
    from src.agentic_video.narrative_agent import build_narrative_material

    return (json.dumps({"title": record.get("title"), "stats": record.get("stats"),
                        "category_hint": record.get("category_hint")},
                       ensure_ascii=False)
            + "\n" + build_narrative_material(cfg, aweme_id))[:12000]


def run_discovery(cfg, *, end_date: str, days: int, per_category: int,
                  output_dir: Path, refetch: bool = False, collect: bool = True,
                  download: bool = True, audition: bool = True, force: bool = False,
                  runner=None) -> dict:
    """Collect, cheaply triage, download, and semantically audition a balanced pool."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if collect:
        from src.pipeline.collect_trends import run_trend_stage
        run_trend_stage(cfg, sub_types=cfg.wellbyte.sub_types, date=end_date,
                        refetch=refetch)
    records, state = load_rolling_candidates(cfg, end_date=end_date, days=days)
    write_jsonl(output_dir / "candidates.jsonl", records)
    triaged = [metadata_triage(record) for record in records]
    write_jsonl(output_dir / "triage.jsonl", triaged)

    shortlist = build_audition_shortlist(triaged, per_category=per_category)
    download_stats = None
    stale_missing = []
    if download:
        from src.pipeline.run_downloads import run_download_stage
        merged_by_id = state["merged_by_id"]
        downloadable = []
        for row in shortlist:
            local_video = cfg.paths.videos_dir / row["aweme_id"] / "video.mp4"
            if local_video.exists() or row.get("download_url_observed_date") == end_date:
                downloadable.append(row)
            else:
                stale_missing.append(row["aweme_id"])
        download_stats = run_download_stage(
            cfg, [merged_by_id[row["aweme_id"]] for row in downloadable
                  if row["aweme_id"] in merged_by_id])

    audits = []
    audition_dir = output_dir / "semantic_auditions"
    audition_dir.mkdir(parents=True, exist_ok=True)
    available = [row for row in shortlist
                 if (cfg.paths.videos_dir / row["aweme_id"] / "video.mp4").exists()]
    if audition and available:
        from src.perception.run_all import run_all
        run_all(cfg, ids=[row["aweme_id"] for row in available], force=force)
        if runner is None:
            from src.perception.omni_runner import OmniRunner
            runner = OmniRunner(cfg.perception.get("omni") or {})
        for record in available:
            path = audition_dir / f"{record['aweme_id']}.json"
            if path.exists() and not force:
                audits.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            video = cfg.paths.videos_dir / record["aweme_id"] / "video.mp4"
            context = _audition_context(cfg, record["aweme_id"], record)
            prompt = SEMANTIC_AUDIT_PROMPT.replace("{context}", context)
            answer = runner.watch(video, prompt, max_new_tokens=2048)
            audit = parse_semantic_audit(answer.text, record)
            audit["elapsed_s"] = answer.elapsed_s
            path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            audits.append(audit)
    else:
        for path in sorted(audition_dir.glob("*.json")):
            audits.append(json.loads(path.read_text(encoding="utf-8")))
    triage_by_id = {str(row["aweme_id"]): row for row in triaged}
    audits = [revalidate_semantic_audit(
        audit, triage_by_id.get(str(audit.get("aweme_id")), {"eligible": True}))
        for audit in audits]
    for audit in audits:
        path = audition_dir / f"{audit.get('aweme_id')}.json"
        path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    selected, gaps = select_balanced(audits, per_category=per_category)
    write_jsonl(output_dir / "selected.jsonl", selected)
    write_selection_report(output_dir / "selection_report.md", selected, gaps,
                           candidate_count=len(records), audited_count=len(audits))
    manifest = {
        "end_date": end_date, "days": days, "dates_used": state["dates_used"],
        "candidate_count": len(records), "shortlist_count": len(shortlist),
        "audited_count": len(audits), "selected_count": len(selected), "gaps": gaps,
        "download": download_stats, "stale_missing_skipped": stale_missing,
    }
    (output_dir / "discovery_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
