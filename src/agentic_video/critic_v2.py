"""Structured Recipe-v2 critic, patch guard, and bounded stopping policy."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from src.agentic_video.recipe_v2 import recipe_hash, validate_recipe_v2
from src.template.prompts import dedupe_json_repetition
from src.template.schema import extract_json_block

ASPECTS = ("recipe_fidelity", "theme_relevance", "rhythm_continuity", "render_integrity")
PATCH_OPS = ("add", "replace", "remove")

CRITIC_PROMPT = """你是可审计的视频剪辑 Critic。请观看成片，并对照 Recipe、素材计划和检索结果。
分别评估：剪辑程序遵循、主题相关性、节拍/视觉连续性、渲染完整性。

只输出一个 JSON 对象：
{{"score": 0.0,
 "aspects": {{"recipe_fidelity": 0.0, "theme_relevance": 0.0,
              "rhythm_continuity": 0.0, "render_integrity": 0.0}},
 "issues": [{{"aspect": "render_integrity", "severity": "high", "t_s": 0.0,
             "evidence": "具体可见现象", "operation_id": null}}],
 "patches": [{{"op": "replace", "path": "/operations/0/params/rate", "value": 1.5,
              "reason": "补丁与可见证据之间的关系"}}],
 "verdict": "一句话结论"}}

score 和四个方面均为 0~1。只能建议结构化 Recipe 补丁，不能要求人工操作，不能直接改视频。
没有足够证据时 patches 为空。补丁路径只能指向 operations 中的 params/status/confidence/interval，
或 assets 中的 source_range/uri。不要删除证据。

注意：参考 Recipe 约束的是剪辑结构、节拍和图层关系，不要求复制参考视频中的人物、地点或文字。
当用户主题要求替换素材时，参考片主体与成片主体不同是预期行为；主题相关性应依据用户主题、
素材计划和检索到的素材描述判断，不能仅因参考片是城市而成片是鬼灭，或反之，就判定失败。

【Recipe】{recipe}
【素材计划】{asset_plan}
【检索结果】{retrieval}
"""


def parse_critique(raw: str) -> dict | None:
    block = extract_json_block(dedupe_json_repetition(raw, first_key='"score"'))
    if not block:
        return None
    try:
        critique = json.loads(block)
    except ValueError:
        return None
    if not isinstance(critique, dict) or not isinstance(critique.get("issues"), list):
        return None
    try:
        critique["score"] = min(1.0, max(0.0, float(critique.get("score", 0))))
    except (TypeError, ValueError):
        return None
    aspects = critique.get("aspects")
    if not isinstance(aspects, dict):
        return None
    normalized = {}
    for aspect in ASPECTS:
        try:
            normalized[aspect] = min(1.0, max(0.0, float(aspects.get(aspect, 0))))
        except (TypeError, ValueError):
            normalized[aspect] = 0.0
    critique["aspects"] = normalized
    for issue in critique["issues"]:
        if isinstance(issue, dict) and issue.get("aspect") not in ASPECTS:
            issue["aspect"] = "render_integrity"
    critique["patches"] = [patch for patch in critique.get("patches") or []
                           if isinstance(patch, dict)]
    return critique


def validate_patch(recipe: dict, patch: dict) -> list[str]:
    errors = []
    if patch.get("op") not in PATCH_OPS:
        errors.append("patch op invalid")
    path = patch.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        return errors + ["patch path invalid"]
    parts = path.strip("/").split("/")
    allowed = False
    if len(parts) >= 3 and parts[0] == "operations" and parts[1].isdigit():
        index = int(parts[1])
        allowed = index < len(recipe.get("operations") or []) and parts[2] in {
            "params", "status", "confidence", "interval"}
        if parts[2] == "params" and len(parts) < 4:
            allowed = False
    elif len(parts) == 3 and parts[0] == "assets" and parts[1].isdigit():
        index = int(parts[1])
        allowed = index < len(recipe.get("assets") or []) and parts[2] in {
            "source_range", "uri"}
    if not allowed:
        errors.append("patch path is outside critic allowlist")
    if patch.get("op") in {"add", "replace"} and "value" not in patch:
        errors.append("patch value missing")
    return errors


def _path_parts(path: str) -> list[str]:
    return [part.replace("~1", "/").replace("~0", "~")
            for part in path.strip("/").split("/")]


def apply_recipe_patches(recipe: dict, patches: list[dict]) -> tuple[dict, list[dict]]:
    """Apply allowlisted JSON-style patches; validate after every candidate patch."""
    current = copy.deepcopy(recipe)
    audit = []
    for patch in patches:
        errors = validate_patch(current, patch)
        if errors:
            audit.append({"patch": patch, "status": "rejected", "errors": errors})
            continue
        candidate = copy.deepcopy(current)
        parts = _path_parts(patch["path"])
        target = candidate
        try:
            for part in parts[:-1]:
                target = target[int(part)] if isinstance(target, list) else target[part]
            key = parts[-1]
            if patch["op"] == "remove":
                if isinstance(target, list):
                    target.pop(int(key))
                else:
                    target.pop(key)
            elif isinstance(target, list):
                target[int(key)] = patch["value"]
            else:
                target[key] = patch["value"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            audit.append({"patch": patch, "status": "rejected", "errors": [str(exc)]})
            continue
        validation = validate_recipe_v2(candidate)
        if validation:
            audit.append({"patch": patch, "status": "rejected", "errors": validation})
            continue
        current = candidate
        audit.append({"patch": patch, "status": "applied"})
    return current, audit


def should_stop(history: list[dict], next_recipe: dict, *, max_rounds: int = 2,
                min_improvement: float = 0.02) -> tuple[bool, str]:
    next_hash = recipe_hash(next_recipe)
    if any(row.get("recipe_hash") == next_hash for row in history):
        return True, "recipe_repeated"
    if len(history) >= 2:
        previous = history[-2].get("score")
        current = history[-1].get("score")
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)) \
                and current - previous < min_improvement:
            return True, "insufficient_improvement"
    if len(history) >= max_rounds:
        return True, "max_rounds"
    return False, "continue"


def recipe_skeleton(recipe: dict) -> str:
    """V3 P4：critic 只看 Recipe 操作骨架（type/interval/status）——旧版全文
    12k 字里带参考片自己的文字层/人物描述，edit critic 幻觉出成片里不存在
    的"跆拳道比赛、手写文字、户外风景照"（C2 实锤参考污染）。"""
    rows = []
    for op in (recipe.get("operations") or [])[:80]:
        rows.append({"type": op.get("type"),
                     "interval": op.get("interval"),
                     "status": op.get("status")})
    return json.dumps({"reference": recipe.get("reference"),
                       "operations": rows}, ensure_ascii=False)[:3000]


def reference_terms(narrative: dict | None, recipe: dict | None) -> list[str]:
    """V3 P4：参考片专属词（intent 主题/主角 + 参考文字层文本）——critic 证据
    引用这些词而目标库里没有，就是参考污染幻觉的特征。长 CJK 段切二元组
    （"跆拳道服"→跆拳/拳道），证据里"跆拳道比赛"才对得上。"""
    terms: set[str] = set()

    def _absorb(text: str) -> None:
        for run in _content_tokens(text):
            if len(run) == 2:
                terms.add(run)
            elif len(run) > 2:
                terms.update(run[i:i + 2] for i in range(len(run) - 1))

    intent = (narrative or {}).get("intent") or {}
    for field in ("protagonist", "topic", "message"):
        _absorb(str(intent.get(field) or ""))
    for op in ((recipe or {}).get("operations") or []):
        _absorb(str(((op.get("params") or {}).get("text")) or ""))
    return sorted(terms)[:60]


def _content_tokens(text: str) -> list[str]:
    """按标点/空白切出 CJK 连续段（粗粒度内容词）。"""
    tokens, current = [], []
    for char in str(text or ""):
        if "一" <= char <= "鿿":
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def filter_critic_issues(critique: dict, library_entity_names: list[str],
                         reference_tokens: list[str] | None = None) -> dict:
    """V3 P4 幻觉过滤：issue 证据引用**参考片专属词**（且不引用目标素材实体）
    → 参考污染幻觉，弃用。C2 实锤：edit critic 描述成片里不存在的
    "跆拳道比赛、手写文字、户外风景照"（全是参考片内容）。"""
    names = [str(name) for name in library_entity_names if str(name).strip()]
    refs = [str(term) for term in reference_tokens or [] if len(str(term)) >= 2]
    if not refs:
        return critique
    kept, dropped = [], []
    for issue in critique.get("issues") or []:
        evidence = str((issue or {}).get("evidence") or "")
        mentions_library = any(name in evidence for name in names)
        mentions_reference = any(term in evidence for term in refs)
        if mentions_reference and not mentions_library:
            dropped.append({**issue, "dropped_reason": "reference_contamination"})
        else:
            kept.append(issue)
    critique["issues"] = kept
    critique["hallucinated_issues"] = dropped
    return critique


def run_structured_critic(video: Path, recipe: dict, asset_plan: dict,
                          retrieval: list[dict], *, runner,
                          narrative: dict | None = None) -> dict:
    prompt = CRITIC_PROMPT.format(
        recipe=recipe_skeleton(recipe),
        asset_plan=json.dumps(asset_plan, ensure_ascii=False)[:5000],
        retrieval=json.dumps(retrieval, ensure_ascii=False)[:5000])
    # V3 晨修：成片审看用 720p 副本——1080p 直喂在 critic 阶段 OOM（B/C2 实锤）
    from src.perception.common import prepare_watch_copy

    answer = runner.watch(prepare_watch_copy(video), prompt, max_new_tokens=2048)
    critique = parse_critique(answer.text)
    if critique is None:
        critique = {"score": 0.0, "aspects": {aspect: 0.0 for aspect in ASPECTS},
                    "issues": [{"aspect": "render_integrity", "severity": "high",
                                "t_s": 0.0, "evidence": "critic JSON parse failed",
                                "operation_id": None}], "patches": [],
                    "verdict": "critic parse failed", "raw_head": answer.text[:300]}
    critique["elapsed_s"] = answer.elapsed_s
    # 幻觉过滤（V3 P4）：证据引用参考片专属词且不引用目标素材 → 弃用
    entity_names = []
    for row in retrieval or []:
        picked = row.get("picked") or {}
        entity_names.extend(str(v) for v in (picked.get("entity_names") or []))
    return filter_critic_issues(critique, entity_names,
                                reference_tokens=reference_terms(narrative, recipe))


def build_fault_suite(recipes: list[dict], *, count: int = 60) -> list[dict]:
    """Create deterministic, machine-labeled faults for critic evaluation."""
    if count < 0:
        raise ValueError("count must be non-negative")
    faults = []
    kinds = ("interval_shift", "parameter_change", "low_confidence", "uncertain_status")
    candidates = [recipe for recipe in recipes if recipe.get("operations")]
    if count and not candidates:
        raise ValueError("at least one recipe with operations is required")
    for index in range(count):
        truth = candidates[index % len(candidates)]
        faulty = copy.deepcopy(truth)
        op_index = index % len(faulty["operations"])
        op = faulty["operations"][op_index]
        kind = kinds[index % len(kinds)]
        if kind == "interval_shift":
            duration = float(faulty["reference"]["duration_s"])
            start, end = op["interval"]
            shift = min(0.4, max(0.05, duration * 0.05))
            shift = shift if end + shift <= duration else -shift
            op["interval"] = [round(start + shift, 4), round(end + shift, 4)]
            fault_path = f"/operations/{op_index}/interval"
        elif kind == "parameter_change":
            key = sorted(op["params"])[0]
            value = op["params"][key]
            if isinstance(value, bool):
                op["params"][key] = not value
            elif isinstance(value, (int, float)):
                op["params"][key] = value + 0.25
            elif isinstance(value, list):
                op["params"][key] = list(reversed(value))
                if op["params"][key] == value and value:
                    op["params"][key][0] = (float(value[0]) + 0.1
                                             if isinstance(value[0], (int, float))
                                             else str(value[0]) + "_fault")
            else:
                op["params"][key] = str(value) + "_fault"
            fault_path = f"/operations/{op_index}/params/{key}"
        elif kind == "low_confidence":
            op["confidence"] = 0.1
            fault_path = f"/operations/{op_index}/confidence"
        else:
            op["status"] = "uncertain"
            fault_path = f"/operations/{op_index}/status"
        if validate_recipe_v2(faulty):
            raise AssertionError(f"generated invalid fault: {kind}")
        faults.append({"case_id": f"fault_{index:03d}", "kind": kind,
                       "operation_id": op["id"], "fault_path": fault_path,
                       "truth": truth, "faulty": faulty})
    return faults


def write_fault_suite(faults: list[dict], output_dir: Path) -> Path:
    """Persist a replayable fault manifest without validating the intentionally
    corrupted recipes.  Each case keeps both the clean truth and faulty input,
    making model-critic runs resumable and auditable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fault in faults:
        case_id = str(fault["case_id"])
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        truth_path = case_dir / "truth.recipe.json"
        faulty_path = case_dir / "faulty.recipe.json"
        truth_path.write_text(json.dumps(fault["truth"], ensure_ascii=False, indent=2),
                              encoding="utf-8")
        faulty_path.write_text(json.dumps(fault["faulty"], ensure_ascii=False, indent=2),
                               encoding="utf-8")
        rows.append({"case_id": case_id, "kind": fault["kind"],
                     "operation_id": fault["operation_id"],
                     "fault_path": fault["fault_path"],
                     "truth": str(truth_path), "faulty": str(faulty_path)})
    manifest = output_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
    return manifest


def score_fault_critiques(faults: list[dict], critiques: list[dict], *,
                          clean_false_positives: int = 0, clean_count: int = 0) -> dict:
    if len(faults) != len(critiques):
        raise ValueError("fault/critique counts differ")
    detected = repaired = 0
    for fault, critique in zip(faults, critiques):
        if any(issue.get("operation_id") == fault["operation_id"]
               for issue in critique.get("issues") or [] if isinstance(issue, dict)):
            detected += 1
        patched, _audit = apply_recipe_patches(
            fault["faulty"], [row for row in critique.get("patches") or []
                              if isinstance(row, dict)])
        if patched == fault["truth"]:
            repaired += 1
    detection_rate = detected / max(1, len(faults))
    repair_rate = repaired / max(1, len(faults))
    false_positive_rate = clean_false_positives / max(1, clean_count)
    return {"fault_count": len(faults), "detected": detected,
            "detection_rate": round(detection_rate, 4),
            "repaired": repaired, "repair_rate": round(repair_rate, 4),
            "clean_count": clean_count,
            "clean_false_positive_rate": round(false_positive_rate, 4),
            "passes_thresholds": detection_rate >= 0.70 and repair_rate >= 0.70
            and clean_count > 0 and false_positive_rate <= 0.05}
