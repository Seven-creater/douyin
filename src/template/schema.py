"""Template JSON 抽取 / 校验 / 降级解析（全 stdlib，不用 pydantic）。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.template.prompts import ROLE_ENUM

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

TOP_KEYS = ("trend_summary", "core_meme", "timeline", "audio", "fixed_elements",
            "replaceable_elements", "generation_plan")


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)     # 硬错误（拒收）
    warnings: list[str] = field(default_factory=list)   # 软警告（通过但落盘）
    template: dict = field(default_factory=dict)


# ---------- 抽取 ----------

def extract_json_block(text: str) -> str | None:
    """剥 think/围栏 → 从首个 { 或 [ 起配对扫描到外层容器闭合（对象或数组皆可）。"""
    cleaned = _THINK_RE.sub("", text).strip()
    m = _FENCE_RE.search(cleaned)
    if m:
        cleaned = m.group(1).strip()
    start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0), default=-1)
    if start < 0:
        return None
    opener = cleaned[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(cleaned[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                if ch == closer:
                    return cleaned[start : i + 1]
                return None  # 外层类型不匹配（截断/损坏）
    return None


# ---------- 校验 ----------

def _check_number_range(v, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


def validate_template(obj, *, duration_s: float, shot_boundaries: list[float],
                      beat_points_known: list[float]) -> ValidationResult:
    res = ValidationResult(ok=False)
    if not isinstance(obj, dict):
        res.errors.append("顶层不是 JSON 对象")
        return res
    for k in TOP_KEYS:
        if k not in obj:
            res.errors.append(f"缺顶层键 {k}")
    if res.errors:
        return res
    if not isinstance(obj["trend_summary"], str) or not obj["trend_summary"].strip():
        res.errors.append("trend_summary 非字符串或为空")
    if not isinstance(obj["core_meme"], str) or not obj["core_meme"].strip():
        res.errors.append("core_meme 非字符串或为空")
    for k in ("fixed_elements", "replaceable_elements", "generation_plan"):
        if not isinstance(obj[k], list) or not all(isinstance(x, str) for x in obj[k]):
            res.errors.append(f"{k} 非字符串数组")

    # timeline
    tl = obj["timeline"]
    if not isinstance(tl, list) or not tl:
        res.errors.append("timeline 为空或非数组")
    else:
        last_end = -1.0
        for i, seg in enumerate(tl):
            if not isinstance(seg, dict):
                res.errors.append(f"timeline[{i}] 非对象")
                continue
            start, end = seg.get("start"), seg.get("end")
            if not _check_number_range(start, 0, duration_s + 0.5):
                res.errors.append(f"timeline[{i}].start={start} 非法（应 0~{duration_s:.1f}）")
                continue
            if not _check_number_range(end, 0, duration_s + 0.5) or end <= start:
                res.errors.append(f"timeline[{i}].end={end} 非法（应 > start 且 ≤ 时长）")
                continue
            if start < last_end - 0.01:
                res.errors.append(f"timeline[{i}] 与前一段重叠")
            last_end = end
            role = seg.get("role")
            if role not in ROLE_ENUM:
                res.errors.append(f"timeline[{i}].role={role!r} 不在枚举内")
            # 软：snap-to-grid
            if shot_boundaries:
                nearest = min(shot_boundaries, key=lambda b: abs(b - start)) \
                    if isinstance(start, (int, float)) else start
                if isinstance(start, (int, float)) and abs(nearest - start) > 0.5:
                    res.warnings.append(f"timeline[{i}].start={start} 离最近镜头边界 {nearest} 超 0.5s")

    # audio
    audio = obj["audio"]
    if not isinstance(audio, dict):
        res.errors.append("audio 非对象")
    else:
        if not (audio.get("bgm") is None or isinstance(audio.get("bgm"), str)):
            res.errors.append("audio.bgm 非字符串或 null")
        beats = audio.get("beat_points")
        if not isinstance(beats, list):
            res.errors.append("audio.beat_points 非数组")
        else:
            if beats and not beat_points_known:
                res.errors.append("beat_points 非空但材料未提供节拍（禁止自造数字）")
            else:
                for b in beats:
                    if not any(abs(b - k) <= 0.05 for k in beat_points_known):
                        res.errors.append(f"beat_point {b} 不在已知节拍列表内（自造数字）")
                        break
        if not isinstance(audio.get("speech"), list):
            res.errors.append("audio.speech 非数组")

    res.ok = not res.errors
    if res.ok:
        res.template = normalize_template(obj)
    return res


def normalize_template(obj: dict) -> dict:
    """键排序、时间 round(1)、role 未知的保持原样由校验拦截。"""
    out = {
        "trend_summary": (obj.get("trend_summary") or "").strip(),
        "core_meme": (obj.get("core_meme") or "").strip(),
        "timeline": [
            {
                "start": round(float(s.get("start", 0)), 1),
                "end": round(float(s.get("end", 0)), 1),
                "role": s.get("role") or "other",
                "visual": (s.get("visual") or "").strip() or None,
                "speech": (s.get("speech") or "").strip() or None,
                "text": (s.get("text") or "").strip() or None,
            }
            for s in obj.get("timeline") or [] if isinstance(s, dict)
        ],
        "audio": {
            "bgm": (obj.get("audio") or {}).get("bgm") or None,
            "beat_points": [round(float(b), 2) for b in (obj.get("audio") or {}).get("beat_points") or []],
            "speech": (obj.get("audio") or {}).get("speech") or [],
        },
        "fixed_elements": list(obj.get("fixed_elements") or []),
        "replaceable_elements": list(obj.get("replaceable_elements") or []),
        "generation_plan": list(obj.get("generation_plan") or []),
    }
    return out


# ---------- 降级解析 ----------

def parse_template_lenient(text: str) -> tuple[dict | None, str]:
    """整体解析失败后逐键 raw_decode。返回 (template|None, mode)。"""
    block = extract_json_block(text)
    if block:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj, "json"
        except ValueError:
            pass
    cleaned = _THINK_RE.sub("", text)
    obj: dict = {}
    decoder = json.JSONDecoder()
    got_any = False
    for key in TOP_KEYS:
        idx = cleaned.find(f'"{key}"')
        if idx < 0:
            continue
        colon = cleaned.find(":", idx + len(key) + 2)
        if colon < 0:
            continue
        pos = colon + 1
        while pos < len(cleaned) and cleaned[pos] in " \t\r\n":
            pos += 1
        try:
            val, _ = decoder.raw_decode(cleaned, pos)
            obj[key] = val
            got_any = True
        except ValueError:
            continue
    if not got_any:
        return None, "fail"
    return obj, "partial"


def parse_and_validate(text: str, *, duration_s: float, shot_boundaries: list[float],
                       beat_points_known: list[float]) -> tuple[ValidationResult, str]:
    """组合：抽取+解析+校验。mode: json/partial/fail。partial 结果不再走硬校验拒收（缺键补默认）。"""
    block = extract_json_block(text)
    if block:
        try:
            obj = json.loads(block)
            res = validate_template(obj, duration_s=duration_s,
                                    shot_boundaries=shot_boundaries,
                                    beat_points_known=beat_points_known)
            if res.ok:
                return res, "json"
            return res, "json_invalid"   # 有 JSON 但校验失败 → 调用方可 repair 重试
        except ValueError as exc:
            pass
    obj, mode = parse_template_lenient(text)
    if obj is None:
        return ValidationResult(ok=False, errors=["无法解析出任何 JSON"]), "fail"
    # partial：尽量补全默认值，标记为非完整模式
    obj = {**{k: ([] if k in ("fixed_elements", "replaceable_elements", "generation_plan") else
              ({} if k == "audio" else "") ) for k in TOP_KEYS}, **obj}
    res = validate_template(obj, duration_s=duration_s, shot_boundaries=shot_boundaries,
                            beat_points_known=beat_points_known)
    return res, "partial" if res.ok else "partial_invalid"
