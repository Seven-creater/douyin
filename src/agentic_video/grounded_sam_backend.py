"""GroundingDINO + SAM2 video mask execution entry point.

This module runs in the dedicated ``grounded_sam2`` Conda environment. The main
pipeline invokes it as a subprocess so model-specific dependencies do not leak
into the Qwen/FFmpeg environment.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "")[-500:])


def _video_meta(path: Path) -> tuple[float, float]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    rate = data["streams"][0].get("avg_frame_rate", "24/1")
    numerator, denominator = rate.split("/", 1)
    fps = float(numerator) / max(float(denominator), 1e-9)
    return fps, float(data["format"]["duration"])


def run_mask(args: argparse.Namespace) -> dict:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    sys.path.insert(0, str(Path(args.sam2_code_root)))
    from sam2.build_sam import build_sam2_video_predictor

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    work = Path(args.work_dir).resolve()
    frames_dir = work / "frames"
    rendered_dir = work / "rendered"
    if work.exists():
        shutil.rmtree(work)
    frames_dir.mkdir(parents=True)
    rendered_dir.mkdir(parents=True)
    fps, duration = _video_meta(input_path)
    start = max(0.0, min(float(args.start), duration))
    end = max(start + 1 / fps, min(float(args.end), duration))
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(input_path), "-vf",
          f"fps={fps:g}", "-q:v", "2", str(frames_dir / "%06d.jpg")])
    frame_names = sorted(frames_dir.glob("*.jpg"))
    if not frame_names:
        raise RuntimeError("no frames extracted")
    start_idx = min(len(frame_names) - 1, int(round(start * fps)))
    end_idx = min(len(frame_names) - 1, int(round(end * fps)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    image = Image.open(frame_names[start_idx]).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.grounding_model, local_files_only=True)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model, local_files_only=True).to(device).eval()
    inputs = processor(images=image, text=args.target.lower(), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    detections = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.35, text_threshold=0.25,
        target_sizes=[image.size[::-1]])[0]
    if len(detections["boxes"]) == 0:
        raise RuntimeError(f"GroundingDINO found no target for {args.target!r}")
    best = int(torch.argmax(detections["scores"]).item())
    box = detections["boxes"][best].detach().cpu().numpy()
    del grounding_model, processor, inputs, outputs
    if device == "cuda":
        torch.cuda.empty_cache()

    predictor = build_sam2_video_predictor(
        args.sam2_model_config, args.sam2_checkpoint, device=device)
    state = predictor.init_state(video_path=str(frames_dir), offload_video_to_cpu=True,
                                 offload_state_to_cpu=True)
    predictor.add_new_points_or_box(state, frame_idx=start_idx, obj_id=1, box=box)
    masks = {}
    for frame_idx, object_ids, logits in predictor.propagate_in_video(
            state, start_frame_idx=start_idx, max_frame_num_to_track=end_idx - start_idx + 1):
        if frame_idx > end_idx:
            break
        object_pos = list(object_ids).index(1)
        masks[int(frame_idx)] = (logits[object_pos] > 0).cpu().numpy().squeeze()

    for frame_idx, frame_path in enumerate(frame_names):
        frame = cv2.imread(str(frame_path))
        mask = masks.get(frame_idx)
        if mask is not None and start_idx <= frame_idx <= end_idx:
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                  (frame.shape[1], frame.shape[0])) > 0
            if args.mode == "tracked_mask_fill":
                texture = np.zeros_like(frame)
                texture[:, ::12] = (20, 60, 255)
                texture[:, 6::12] = (255, 160, 20)
                frame[mask] = texture[mask]
            elif args.mode == "subject_cutout_composite":
                tinted = cv2.addWeighted(frame, 0.35, np.full_like(frame, (30, 30, 220)),
                                         0.65, 0)
                frame[mask] = tinted[mask]
            elif args.mode == "mask_wipe":
                progress = (frame_idx - start_idx) / max(1, end_idx - start_idx)
                visible = np.indices(mask.shape)[1] <= int(mask.shape[1] * progress)
                frame[mask & visible] = (255, 255, 255)
            else:
                dark = (frame.astype(np.float32) * 0.35).astype(np.uint8)
                frame[~mask] = dark[~mask]
        cv2.imwrite(str(rendered_dir / frame_path.name), frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i",
          str(rendered_dir / "%06d.jpg"), "-i", str(input_path), "-map", "0:v", "-map",
          "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt",
          "yuv420p", "-c:a", "copy", "-t", f"{duration:g}", str(output_path)])
    manifest = {"input": str(input_path), "output": str(output_path), "target": args.target,
                "mode": args.mode, "start_s": start, "end_s": end,
                "detected_score": float(detections["scores"][best].item()),
                "tracked_frames": len(masks)}
    (work / "mask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--target", default="person.")
    parser.add_argument("--grounding-model", required=True)
    parser.add_argument("--sam2-code-root", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--sam2-model-config", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_mask(args), ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"mask backend failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
