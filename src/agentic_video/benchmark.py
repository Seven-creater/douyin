"""Controlled ground-truth suite for edit-program induction."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

from src.agentic_video.recipe_v2 import (LAYER_OP_TYPES, POINT_OP_TYPES,
                                         new_recipe, sha256_file,
                                         validate_recipe_v2, write_recipe)

FAMILY_OPS = {
    "cut_timing": ("hard_cut", "match_cut"),
    "speed": ("speed_ramp",),
    "crop": ("crop_reframe", "zoom_punch"),
    "flash_color": ("luma_flash", "color_adjust", "opacity_blend"),
    "text": ("text_overlay",),
    "text_animation": ("text_layer_animation",),
    "cutout": ("subject_cutout_composite", "tracked_mask_fill"),
    "mask_transition": ("mask_wipe", "foreground_occlusion_transition"),
}


def _track_for(op_type: str) -> tuple[str, str, int]:
    if op_type.startswith("text"):
        return "text_main", "text", 20
    if op_type in {"tracked_mask_fill", "mask_wipe"}:
        return "mask_main", "mask", 10
    if op_type in {"subject_cutout_composite", "foreground_occlusion_transition",
                   "opacity_blend"}:
        return "overlay_main", "overlay", 10
    return "video_main", "video", 0


def _operation(op_type: str, index: int, rng: random.Random, duration: float) -> dict:
    start = round(rng.uniform(0.45, duration - 1.0), 3)
    if op_type in POINT_OP_TYPES:
        end = start
    else:
        end = round(min(duration, start + rng.uniform(0.35, 0.8)), 3)
    track_id, _, _ = _track_for(op_type)
    params: dict = {}
    if op_type == "speed_ramp":
        params = {"rate": round(rng.choice((0.5, 1.5, 2.0)), 2)}
    elif op_type in {"crop_reframe", "zoom_punch"}:
        params = {"scale": round(rng.uniform(1.12, 1.45), 2),
                  "center": [round(rng.uniform(0.4, 0.6), 2),
                             round(rng.uniform(0.4, 0.6), 2)]}
    elif op_type in {"text_overlay", "text_layer_animation"}:
        params = {"text": f"EDIT {index}", "animation": "slide_up"}
    elif op_type in {"tracked_mask_fill", "subject_cutout_composite"}:
        params = {"target": "moving_subject", "fill": "stripe"}
    elif op_type in {"mask_wipe", "foreground_occlusion_transition"}:
        params = {"direction": rng.choice(("left", "right"))}
    elif op_type == "luma_flash":
        params = {"strength": 1.0, "duration_frames": 2}
    elif op_type == "color_adjust":
        params = {"hue_shift": rng.choice((30, 60, 120))}
    elif op_type == "opacity_blend":
        params = {"opacity": round(rng.uniform(0.35, 0.7), 2)}
    elif op_type in {"hard_cut", "match_cut"}:
        params = {"switch_palette": True}
    return {
        "id": f"op_{index:02d}", "type": op_type,
        "interval": [start, end], "track_id": track_id,
        "inputs": [], "depends_on": [], "params": params,
        "evidence": [{"source": "controlled_generator", "timestamp_s": start,
                      "seed_offset": index}],
        "confidence": 1.0, "status": "supported",
    }


def _case_recipe(case_id: str, families: list[str], seed: int, *, duration_s: float,
                 fps: float) -> dict:
    recipe = new_recipe(reference_id=case_id, reference_uri="video.mp4", sha256="",
                        duration_s=duration_s, fps=fps, model="controlled_generator",
                        prompt_version="controlled_v1", seed=seed)
    rng = random.Random(seed)
    tracks = {t["id"]: t for t in recipe["tracks"]}
    for index, family in enumerate(families):
        choices = FAMILY_OPS[family]
        op_type = choices[(seed + index) % len(choices)]
        op = _operation(op_type, index, rng, duration_s)
        track_id, track_type, z = _track_for(op_type)
        tracks.setdefault(track_id, {"id": track_id, "type": track_type, "z": z})
        recipe["operations"].append(op)
    recipe["tracks"] = sorted(tracks.values(), key=lambda item: (item["z"], item["id"]))
    recipe["benchmark"] = {"families": families, "suite": "controlled_v1"}
    return recipe


def build_suite_specs(*, seed: int = 20260909, duration_s: float = 3.0,
                      fps: float = 12.0) -> list[dict]:
    """Return exactly 96 single, 24 paired, and 24 compound case recipes."""
    families = list(FAMILY_OPS)
    specs: list[dict] = []
    case_no = 0
    for family in families:
        for _ in range(12):
            cid = f"single_{case_no:03d}_{family}"
            specs.append(_case_recipe(cid, [family], seed + case_no,
                                      duration_s=duration_s, fps=fps))
            case_no += 1
    for index in range(24):
        chosen = [families[index % 8], families[(index * 3 + 1) % 8]]
        if chosen[0] == chosen[1]:
            chosen[1] = families[(index + 4) % 8]
        cid = f"paired_{index:03d}"
        specs.append(_case_recipe(cid, chosen, seed + case_no,
                                  duration_s=duration_s, fps=fps))
        case_no += 1
    for index in range(24):
        count = 3 + index % 3
        start = (index * 5) % len(families)
        chosen = [families[(start + j * 3) % len(families)] for j in range(count)]
        cid = f"compound_{index:03d}"
        specs.append(_case_recipe(cid, chosen, seed + case_no,
                                  duration_s=duration_s, fps=fps))
        case_no += 1
    return specs


def _active(op: dict, t: float, frame_idx: int, fps: float) -> bool:
    start, end = op["interval"]
    if end == start:
        frames = int(op.get("params", {}).get("duration_frames", 1))
        return start <= t < start + frames / fps
    return start <= t <= end


def render_controlled_video(recipe: dict, output: Path, *, width: int = 256,
                            height: int = 144) -> Path:
    """Render a compact deterministic visual stimulus from a truth Recipe."""
    import cv2
    import numpy as np

    errors = validate_recipe_v2(recipe)
    if errors:
        raise ValueError("invalid controlled recipe: " + "; ".join(errors))
    fps = float(recipe["reference"]["fps"])
    duration = float(recipe["reference"]["duration_s"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create controlled video: {output}")
    seed = int(recipe["provenance"]["seed"])
    n_frames = int(round(duration * fps))
    yy, xx = np.mgrid[0:height, 0:width]
    try:
        for frame_idx in range(n_frames):
            t = frame_idx / fps
            phase = frame_idx
            for op in recipe["operations"]:
                if op["type"] == "speed_ramp" and _active(op, t, frame_idx, fps):
                    phase = int(frame_idx * float(op["params"].get("rate", 1.0)))
            base = np.zeros((height, width, 3), np.uint8)
            base[..., 0] = (xx + phase * 4 + seed % 91) % 255
            base[..., 1] = (yy * 2 + phase * 3 + seed % 67) % 255
            base[..., 2] = ((xx // 2 + yy + seed) % 180) + 40
            cx = int((0.15 + 0.7 * ((phase % n_frames) / max(1, n_frames - 1))) * width)
            cy = int(height * (0.5 + 0.12 * math.sin(phase / 3)))
            cv2.circle(base, (cx, cy), 18, (30, 230, 250), -1)
            frame = base
            for op in recipe["operations"]:
                typ = op["type"]
                start, end = op["interval"]
                active = _active(op, t, frame_idx, fps)
                if typ in {"hard_cut", "match_cut"} and t >= start:
                    frame = frame[..., ::-1].copy()
                elif typ in {"crop_reframe", "zoom_punch"} and active:
                    scale = float(op["params"].get("scale", 1.25))
                    nw, nh = int(width / scale), int(height / scale)
                    x0, y0 = (width - nw) // 2, (height - nh) // 2
                    frame = cv2.resize(frame[y0:y0 + nh, x0:x0 + nw], (width, height))
                elif typ == "luma_flash" and active:
                    frame[:] = 255
                elif typ == "color_adjust" and active:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv[..., 0] = (hsv[..., 0].astype(int)
                                   + int(op["params"].get("hue_shift", 60)) // 2) % 180
                    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                elif typ == "opacity_blend" and active:
                    overlay = np.full_like(frame, (200, 40, 160))
                    alpha = float(op["params"].get("opacity", 0.5))
                    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
                elif typ in {"text_overlay", "text_layer_animation"} and active:
                    progress = (t - start) / max(end - start, 1 / fps)
                    y = height - 18
                    if typ == "text_layer_animation":
                        y = int(height + (height - 18 - height) * min(1, progress * 3))
                    cv2.putText(frame, op["params"].get("text", "EDIT"), (12, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                elif typ in {"tracked_mask_fill", "subject_cutout_composite"} and active:
                    mask = np.zeros((height, width), np.uint8)
                    cv2.circle(mask, (cx, cy), 20, 255, -1)
                    stripe = np.zeros_like(frame)
                    stripe[:, ::8] = (30, 30, 255)
                    frame[mask > 0] = stripe[mask > 0]
                elif typ in {"mask_wipe", "foreground_occlusion_transition"} and active:
                    progress = (t - start) / max(end - start, 1 / fps)
                    edge = int(width * min(1, max(0, progress)))
                    if op["params"].get("direction") == "right":
                        frame[:, width - edge:] = (20, 20, 20)
                    else:
                        frame[:, :edge] = (20, 20, 20)
            writer.write(frame)
    finally:
        writer.release()
    return output


def generate_suite(output_dir: Path, *, render: bool = True,
                   overwrite: bool = False) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    rows = []
    for recipe in build_suite_specs():
        case_id = recipe["reference"]["id"]
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        video_path = case_dir / "video.mp4"
        truth_path = case_dir / "gt.recipe.json"
        if render and (overwrite or not video_path.exists()):
            render_controlled_video(recipe, video_path)
        if video_path.exists():
            recipe["reference"]["sha256"] = sha256_file(video_path)
            recipe["reference"]["uri"] = str(video_path)
        write_recipe(recipe, truth_path)
        rows.append({"case_id": case_id, "kind": case_id.split("_", 1)[0],
                     "families": recipe["benchmark"]["families"],
                     "video": str(video_path), "truth": str(truth_path)})
    manifest_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                             encoding="utf-8")
    return manifest_path


def interval_iou(left: list[float], right: list[float], *, tolerance_s: float = 0.0) -> float:
    if left[0] == left[1] and right[0] == right[1]:
        return 1.0 if abs(left[0] - right[0]) <= tolerance_s else 0.0
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def evaluate_recipes(truths: list[dict], predictions: list[dict], *,
                     boundary_frames: int = 2) -> dict:
    """Greedy type-aware matching with boundary and temporal-IoU metrics."""
    if len(truths) != len(predictions):
        raise ValueError("truth/prediction case counts differ")
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    matched_ious: list[float] = []
    cut_tp = cut_fp = cut_fn = 0
    ungrounded = predicted_count = model_calls = 0
    for truth, pred in zip(truths, predictions):
        fps = float(truth["reference"]["fps"])
        tolerance = boundary_frames / fps
        gt_ops = truth.get("operations") or []
        pr_ops = pred.get("operations") or []
        model_calls += len((pred.get("provenance") or {}).get("tool_calls") or [])
        predicted_count += len(pr_ops)
        ungrounded += sum(1 for op in pr_ops if op.get("type") not in POINT_OP_TYPES
                          and not op.get("evidence"))
        used: set[int] = set()
        for gt in gt_ops:
            candidates = []
            for idx, candidate in enumerate(pr_ops):
                if idx in used or candidate.get("type") != gt.get("type"):
                    continue
                score = interval_iou(gt["interval"], candidate.get("interval", [0, 0]),
                                     tolerance_s=tolerance)
                candidates.append((score, idx))
            best = max(candidates, default=(0.0, -1))
            threshold = 1.0 if gt["type"] in POINT_OP_TYPES else 0.3
            if best[0] >= threshold:
                tp[gt["type"]] += 1
                used.add(best[1])
                matched_ious.append(best[0])
            else:
                fn[gt["type"]] += 1
        for idx, op in enumerate(pr_ops):
            if idx not in used:
                fp[op.get("type", "unknown")] += 1

        gt_cuts = [op for op in gt_ops if op.get("type") in {"hard_cut", "match_cut"}]
        pr_cuts = [op for op in pr_ops if op.get("type") in {"hard_cut", "match_cut"}]
        matched_cuts: set[int] = set()
        for gt in gt_cuts:
            choices = [(abs(gt["interval"][0] - p["interval"][0]), idx)
                       for idx, p in enumerate(pr_cuts) if idx not in matched_cuts]
            best = min(choices, default=(float("inf"), -1))
            if best[0] <= tolerance:
                cut_tp += 1
                matched_cuts.add(best[1])
            else:
                cut_fn += 1
        cut_fp += len(pr_cuts) - len(matched_cuts)

    all_types = sorted(set(tp) | set(fp) | set(fn))
    per_type = {}
    for typ in all_types:
        precision = tp[typ] / max(1, tp[typ] + fp[typ])
        recall = tp[typ] / max(1, tp[typ] + fn[typ])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_type[typ] = {"precision": round(precision, 4), "recall": round(recall, 4),
                         "f1": round(f1, 4), "tp": tp[typ], "fp": fp[typ], "fn": fn[typ]}
    macro = sum(v["f1"] for v in per_type.values()) / max(1, len(per_type))
    layer_values = [per_type[t]["f1"] for t in LAYER_OP_TYPES if t in per_type]
    cut_precision = cut_tp / max(1, cut_tp + cut_fp)
    cut_recall = cut_tp / max(1, cut_tp + cut_fn)
    cut_f1 = 2 * cut_precision * cut_recall / max(1e-12, cut_precision + cut_recall)
    result = {
        "n_cases": len(truths), "cut_boundary_f1": round(cut_f1, 4),
        "op_macro_f1": round(macro, 4),
        "layer_op_f1": round(sum(layer_values) / max(1, len(layer_values)), 4),
        "median_tiou": round(median(matched_ious), 4) if matched_ious else 0.0,
        "ungrounded_rate": round(ungrounded / max(1, predicted_count), 4),
        "model_calls": model_calls, "per_type": per_type,
    }
    result["passes_thresholds"] = (
        result["cut_boundary_f1"] >= 0.90
        and result["op_macro_f1"] >= 0.85
        and result["layer_op_f1"] >= 0.75
        and result["median_tiou"] >= 0.80
        and result["ungrounded_rate"] <= 0.05
    )
    return result


def evaluate_suite(suite_dir: Path, predictions_dir: Path) -> dict:
    rows = [json.loads(line) for line in (Path(suite_dir) / "manifest.jsonl")
            .read_text(encoding="utf-8").splitlines() if line]
    truths, predictions = [], []
    for row in rows:
        truths.append(json.loads(Path(row["truth"]).read_text(encoding="utf-8")))
        pred_path = Path(predictions_dir) / row["case_id"] / "prediction.recipe.json"
        predictions.append(json.loads(pred_path.read_text(encoding="utf-8")))
    return evaluate_recipes(truths, predictions)


def compare_ablation(fixed: dict, agent: dict) -> dict:
    layer_gain = float(agent["layer_op_f1"]) - float(fixed["layer_op_f1"])
    call_reduction = 1.0 - float(agent.get("model_calls", 0)) / max(
        1.0, float(fixed.get("model_calls", 0)))
    similar_accuracy = float(agent["op_macro_f1"]) >= float(fixed["op_macro_f1"]) - 0.01
    passed = layer_gain >= 0.10 or (similar_accuracy and call_reduction >= 0.20)
    return {"layer_f1_gain": round(layer_gain, 4),
            "model_call_reduction": round(call_reduction, 4), "passes_agent_gate": passed}


def compare_ablation_variants(fixed: dict, signal_guided: dict | None,
                              agent: dict) -> dict:
    """Compare the three required perception variants in one auditable record.

    ``fixed`` is the fixed-window baseline, ``signal_guided`` is the
    signal-guided/no-agent baseline, and ``agent`` is the bounded active agent.
    The agent gate is measured against the fixed baseline as specified by the
    protocol; the middle variant is reported for diagnosis even when omitted.
    """
    result = {"fixed_window": fixed, "agent": agent,
              "agent_vs_fixed": compare_ablation(fixed, agent)}
    if signal_guided is not None:
        result["signal_guided"] = signal_guided
        result["signal_guided_vs_fixed"] = compare_ablation(fixed, signal_guided)
        result["agent_vs_signal_guided"] = compare_ablation(signal_guided, agent)
    return result
