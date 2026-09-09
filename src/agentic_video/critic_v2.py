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


def run_structured_critic(video: Path, recipe: dict, asset_plan: dict,
                          retrieval: list[dict], *, runner) -> dict:
    prompt = CRITIC_PROMPT.format(
        recipe=json.dumps(recipe, ensure_ascii=False)[:12000],
        asset_plan=json.dumps(asset_plan, ensure_ascii=False)[:5000],
        retrieval=json.dumps(retrieval, ensure_ascii=False)[:5000])
    answer = runner.watch(video, prompt, max_new_tokens=2048)
    critique = parse_critique(answer.text)
    if critique is None:
        critique = {"score": 0.0, "aspects": {aspect: 0.0 for aspect in ASPECTS},
                    "issues": [{"aspect": "render_integrity", "severity": "high",
                                "t_s": 0.0, "evidence": "critic JSON parse failed",
                                "operation_id": None}], "patches": [],
                    "verdict": "critic parse failed", "raw_head": answer.text[:300]}
    critique["elapsed_s"] = answer.elapsed_s
    return critique


def build_fault_suite(recipes: list[dict], *, count: int = 60) -> list[dict]:
    """Create deterministic, machine-labeled faults for critic evaluation."""
    faults = []
    kinds = ("interval_shift", "missing_evidence", "bad_confidence", "bad_status")
    candidates = [recipe for recipe in recipes if recipe.get("operations")]
    for index in range(count):
        truth = candidates[index % len(candidates)]
        faulty = copy.deepcopy(truth)
        op_index = index % len(faulty["operations"])
        op = faulty["operations"][op_index]
        kind = kinds[index % len(kinds)]
        if kind == "interval_shift":
            duration = float(faulty["reference"]["duration_s"])
            start, end = op["interval"]
            shift = min(0.4, max(0.05, duration - end))
            op["interval"] = [round(start + shift, 4), round(end + shift, 4)]
        elif kind == "missing_evidence":
            op["evidence"] = []
        elif kind == "bad_confidence":
            op["confidence"] = 1.5
        else:
            op["status"] = "silently_degraded"
        faults.append({"case_id": f"fault_{index:03d}", "kind": kind,
                       "operation_id": op["id"], "truth": truth, "faulty": faulty})
    return faults


def score_fault_critiques(faults: list[dict], critiques: list[dict], *,
                          clean_false_positives: int = 0, clean_count: int = 0) -> dict:
    if len(faults) != len(critiques):
        raise ValueError("fault/critique counts differ")
    detected = 0
    for fault, critique in zip(faults, critiques):
        if any(issue.get("operation_id") == fault["operation_id"]
               for issue in critique.get("issues") or [] if isinstance(issue, dict)):
            detected += 1
    detection_rate = detected / max(1, len(faults))
    false_positive_rate = clean_false_positives / max(1, clean_count)
    return {"fault_count": len(faults), "detected": detected,
            "detection_rate": round(detection_rate, 4),
            "clean_false_positive_rate": round(false_positive_rate, 4),
            "passes_thresholds": detection_rate >= 0.70 and false_positive_rate <= 0.05}
