"""signals：剪辑信号确定性检测轨（Edit Decomposer 的证据底座，D1）。

Editing Branch V1 计划（2026-09-09）：把爆款剪辑片反编译成 Editing Recipe 的第一步——
先算「不可能被 LLM 编造」的物理证据：ffmpeg 单遍解码（fps 采样/灰度/降分辨率）→
每帧对指标（亮度 / 全局差分 / 6×6 网格差分 / Farneback 光流幅度·角度集中·径向分数）→
类型化事件候选：luma_flash / hard_cut / match_cut / whip_pan / speed_ramp|zoom_punch /
region_change（tracked_mask_fill·texture_replace·mask_wipe 族）/ beat_freeze / montage_burst。
另存候选周边 contact sheet（人工验收与 D2 窗口问答对照用）。

D2 窗口问答与 D3 Recipe 合成以本轨候选为时间锚（反编造：t_s 只能取证据时刻）。

CLI：python -m src.library.signals --id <vid> [--force]
产物：data/library/editing/<id>/signals/result.json (+ frames/cand_*.jpg)
纯函数（frame_stats/pair_grid_diff/derive_event_candidates）零 cv2/ffmpeg 依赖，本地可测。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import AppConfig, ensure_utf8_stdio, load_config, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)

try:
    import cv2  # 服务器 omni_src 有；本地 anaconda 缺失时仅光流/落盘降级
except ImportError:  # pragma: no cover
    cv2 = None


# ---------- 纯函数（单测覆盖，零外部依赖） ----------

def frame_stats(gray) -> tuple[float, float]:
    """单帧亮度均值/标准差（flash 检测的输入）。"""
    return float(gray.mean()), float(gray.std())


def pair_grid_diff(prev, cur, rows: int = 6, cols: int = 6):
    """相邻帧对：全局平均差分 + 网格单元差分（局部变化 → mask/纹理族证据）。"""
    import numpy as np

    diff = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    h, w = diff.shape
    cells = diff.reshape(rows, h // rows, cols, w // cols).mean(axis=(1, 3))
    return float(diff.mean()), cells.ravel()


def pair_flow(prev, cur, downscale: int = 2) -> dict:
    """相邻帧对 Farneback 光流：幅度均值 / 主方向集中度 / 径向分数（zoom 判据）。"""
    import numpy as np

    if cv2 is None:
        return {"mag_mean": 0.0, "angle_topbin": 0.0, "radial_score": 0.0}
    a = prev[::downscale, ::downscale]
    b = cur[::downscale, ::downscale]
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32),
                               flow[..., 1].astype(np.float32))
    mag_mean = float(mag.mean())
    hist = np.histogram(ang.ravel(), bins=8, range=(0, 2 * np.pi),
                        weights=mag.ravel())[0]
    topbin = float(hist.max() / (hist.sum() + 1e-6))
    h, w = a.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    radial = ((xs - cx) * flow[..., 0] + (ys - cy) * flow[..., 1]) \
        / (np.hypot(xs - cx, ys - cy) + 5.0)
    radial_score = float((radial * mag).sum() / (mag.sum() + 1e-6))  # -1..1
    return {"mag_mean": round(mag_mean, 3), "angle_topbin": round(topbin, 3),
            "radial_score": round(radial_score, 3)}


def compute_series(frames, *, grid_rows: int = 6, grid_cols: int = 6,
                   flow_downscale: int = 2) -> dict:
    """帧序列 → 指标序列。t=帧时刻；diff/flow 按边界索引 i（帧 i→i+1，时刻记 t[i+1]）。"""
    import numpy as np

    n = len(frames)
    luma_mean, luma_std, diff_global, grid, flow_mag, flow_topbin, flow_radial = \
        [], [], [], [], [], [], []
    for f in frames:
        m, s = frame_stats(f)
        luma_mean.append(round(m, 2))
        luma_std.append(round(s, 2))
    for a, b in zip(frames[:-1], frames[1:]):
        d, cells = pair_grid_diff(a, b, grid_rows, grid_cols)
        fl = pair_flow(a, b, flow_downscale)
        diff_global.append(round(d, 2))
        grid.append([round(float(c), 2) for c in cells])
        flow_mag.append(fl["mag_mean"])
        flow_topbin.append(fl["angle_topbin"])
        flow_radial.append(fl["radial_score"])
    return {"t": [round(i, 3) for i in range(n)],  # 占位，run 侧按 fps_actual 重算
            "luma_mean": luma_mean, "luma_std": luma_std,
            "diff_global": diff_global, "grid": grid,
            "flow_mag": flow_mag, "flow_topbin": flow_topbin,
            "flow_radial": flow_radial}


def derive_event_candidates(series: dict, *, shot_boundaries=(),
                            thr: dict | None = None) -> list[dict]:
    """指标序列 + ffmpeg 场景边界 → 类型化事件候选（D2/D3 的时间锚）。纯函数。"""
    import numpy as np

    cfg = thr or {}
    flash_delta = cfg.get("flash_delta", 40.0)
    cut_delta = cfg.get("cut_delta", 28.0)
    cell_delta = cfg.get("cell_delta", 25.0)
    region_cells_min = cfg.get("region_cells_min", 6)
    region_diff_max = cfg.get("region_diff_max", 10.0)
    whip_mag_min = cfg.get("whip_mag_min", 4.0)
    freeze_mag_max = cfg.get("freeze_mag_max", 0.15)
    montage_window_s = cfg.get("montage_window_s", 1.2)
    montage_min_cuts = cfg.get("montage_min_cuts", 3)
    max_events = cfg.get("max_events", 64)

    lm = np.asarray(series["luma_mean"], float)
    dg = np.asarray(series["diff_global"], float)
    fm = np.asarray(series["flow_mag"], float)
    tb = np.asarray(series["flow_topbin"], float)
    rd = np.asarray(series["flow_radial"], float)
    t = np.asarray(series["t"], float)
    bt = t[1:] if len(t) > 1 else t                     # 边界时刻
    n = len(dg)
    bounds = sorted(float(b) for b in (shot_boundaries or []))
    near_bounds = lambda tt, tol=0.3: any(abs(tt - b) <= tol for b in bounds)

    cands: list[dict] = []

    def add(i, hy, sig, conf):
        cands.append({"t_s": round(float(bt[i]), 2), "type_hypotheses": hy,
                      "signature": {k: round(float(v), 2) for k, v in sig.items()},
                      "confidence": round(conf, 2), "source": "signals"})

    # 1) luma_flash：亮度单帧跳变且 ≤4 帧内回落（排除切点级全局差分）
    flash_bound: set[int] = set()
    for i in range(1, len(lm)):
        jump = lm[i] - lm[i - 1]
        if jump <= flash_delta or i - 1 >= n:
            continue
        base = float(np.mean(lm[max(0, i - 4):i]))
        recovers = any(abs(lm[j] - base) < max(8.0, 0.25 * jump)
                       for j in range(i, min(len(lm), i + 5)))
        if recovers and dg[i - 1] < cut_delta:
            flash_bound.add(i - 1)
            add(i - 1, ["luma_flash"],
                {"luma_jump": jump, "diff_global": dg[i - 1]}, 0.7)

    # 2) 切点：全局差分尖峰（非 flash）。与场景边界重合 → 高置信；
    #    边界后光流幅度适中且方向集中 → match_cut 线索
    cut_times: list[float] = []
    for i in range(n):
        if dg[i] < cut_delta or i in flash_bound:
            continue
        tt = float(bt[i])
        m_after = float(np.mean(fm[i + 1:i + 4])) if i + 1 < n else 0.0
        coherent = 0.2 < m_after < 2.0 and (tb[i] > 0.55 or tb[min(i + 1, n - 1)] > 0.55)
        if coherent:
            add(i, ["match_cut", "hard_cut"], {"diff_global": dg[i],
                "flow_after": m_after}, 0.5)
        else:
            add(i, ["hard_cut"], {"diff_global": dg[i],
                "flow_after": m_after}, 0.85 if near_bounds(tt) else 0.55)
        cut_times.append(tt)

    # 3) region_change：无切点、全局差分低，但局部网格单元剧变（主体内部换纹理/mask 族）
    last_region_t = -10.0
    for i in range(n):
        if dg[i] > region_diff_max:
            continue
        cells = np.asarray(series["grid"][i], float)
        active = int((cells > cell_delta).sum())
        quiet = float((cells < cell_delta * 0.4).mean())
        if active >= region_cells_min and quiet >= 0.6 and float(bt[i]) - last_region_t > 0.4:
            last_region_t = float(bt[i])
            add(i, ["tracked_mask_fill", "texture_replace", "mask_wipe"],
                {"cells_active": active, "diff_global": dg[i]}, 0.45)

    # 4) whip_pan：光流幅度连续高企 + 主方向集中（单事件取峰）
    i = 0
    while i < n:
        if fm[i] >= whip_mag_min and tb[i] > 0.6:
            j = i
            while j < n and fm[j] >= whip_mag_min * 0.6:
                j += 1
            peak = i + int(np.argmax(fm[i:j]))
            add(peak, ["whip_pan"], {"flow_peak": fm[peak], "angle_topbin": tb[peak]}, 0.5)
            i = j
        else:
            i += 1

    # 5) speed_ramp / zoom_punch：无切点处光流幅度台阶式变化（径向 → zoom）
    last_ramp_t = -10.0
    for i in range(3, n - 5):
        if dg[i] >= cut_delta or dg[i - 1] >= cut_delta:
            continue
        before = float(np.mean(fm[max(0, i - 5):i]))
        after = float(np.mean(fm[i:i + 5]))
        if float(bt[i]) - last_ramp_t < 0.5:
            continue
        if after > 2.2 * before + 0.3 or (before > 0.5 and after < 0.45 * before):
            last_ramp_t = float(bt[i])
            hy = ["speed_ramp"] if abs(rd[i]) <= 0.5 else ["zoom_punch", "speed_ramp"]
            add(i, hy, {"flow_before": before, "flow_after": after,
                        "radial": rd[i]}, 0.45)

    # 6) beat_freeze：光流与差分同时近零 ≥3 边界（画面定格）
    i = 0
    while i < n:
        if fm[i] < freeze_mag_max and dg[i] < 2.0:
            j = i
            while j < n and fm[j] < freeze_mag_max and dg[j] < 2.0:
                j += 1
            if j - i >= 3:
                add(i + (j - i) // 2, ["beat_freeze"],
                    {"flow_mean": float(np.mean(fm[i:j])), "span_s": j - i}, 0.5)
            i = j
        else:
            i += 1

    # 7) montage_burst：短时间内 ≥3 切点成组
    group: list[float] = []
    for tt in cut_times + [1e9]:
        if group and tt - group[0] > montage_window_s:
            if len(group) >= montage_min_cuts:
                cands.append({"t_s": round(group[0], 2),
                              "type_hypotheses": ["montage_burst"],
                              "signature": {"n_cuts": len(group),
                                            "span_s": round(group[-1] - group[0], 2)},
                              "confidence": 0.8, "source": "signals",
                              "members_s": [round(x, 2) for x in group]})
            group = []
        group.append(tt)

    # 上限：先保每种类型 1 个最高分（多样性优先），再按分数补满 max_events
    # （类型数本身是小常数，理论长度上限 = max(类型数, max_events)）
    if len(cands) > max_events:
        by_conf = sorted(cands, key=lambda c: -c["confidence"])
        keep, seen = [], set()
        for c in by_conf:
            if not any(h in seen for h in c["type_hypotheses"]):
                keep.append(c)
                seen.update(c["type_hypotheses"])
        for c in by_conf:
            if len(keep) >= max_events:
                break
            if c not in keep:
                keep.append(c)
        cands = keep
    return sorted(cands, key=lambda c: c["t_s"])


# ---------- 解码/落盘/编排（服务器跑） ----------

def decode_gray_frames(ffmpeg_bin: str, video: Path, *, fps: float,
                       width: int, height: int, timeout_s: float = 600) -> list:
    """ffmpeg 单遍解码 → 灰度工作分辨率帧列表（rawvideo pipe）。"""
    import subprocess

    import numpy as np

    args = [ffmpeg_bin, "-loglevel", "error", "-i", str(video),
            "-vf", f"fps={fps:g},scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames, chunk = [], width * height
    while True:
        buf = proc.stdout.read(chunk)
        if not buf:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(height, width).copy())
    err = proc.stderr.read().decode(errors="replace")
    rc = proc.wait(timeout=60)
    if rc != 0:
        raise RuntimeError(f"ffmpeg 解码失败 rc={rc}: {err[-300:]}")
    return frames


def save_contact_sheets(frames, series: dict, candidates: list[dict],
                        out_dir: Path, *, pad_s: float = 0.3, top: int = 24) -> int:
    """每个候选存一张 [t-0.3, t, t+0.3] 三帧横排 contact sheet（人工验收用）。"""
    if cv2 is None or not frames:
        return 0
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.asarray(series["t"], float)
    saved = 0
    for c in sorted(candidates, key=lambda c: -c["confidence"])[:top]:
        tt = c["t_s"]
        idxs = [int(np.argmin(np.abs(t - tt + d))) for d in (-pad_s, 0.0, pad_s)]
        idxs = sorted(set(min(max(i, 0), len(frames) - 1) for i in idxs))[:3]
        strip = np.hstack([cv2.resize(frames[i], (240, 424)) for i in idxs])
        cv2.imwrite(str(out_dir / f"cand_{saved:02d}_t{tt:g}.jpg"), strip)
        saved += 1
    return saved


def run_signals(cfg: AppConfig, vid: str, *, force: bool = False) -> Path | None:
    s_cfg = cfg.library.get("signals") or {}
    tdir = cfg.paths.library_dir / "editing" / vid / "signals"
    params = {k: s_cfg.get(k) for k in (
        "work_fps", "scale", "grid", "flash_delta", "cut_delta", "cell_delta",
        "region_cells_min", "region_diff_max", "whip_mag_min", "freeze_mag_max",
        "montage_window_s", "montage_min_cuts", "max_events")}
    if not force and common.done_or_skip(tdir, params, force=False) is not None:
        logger.info("[signals %s] 已有产物，跳过", vid)
        return tdir / "result.json"

    video = cfg.paths.videos_dir / vid / "video.mp4"
    if not video.exists():
        logger.error("[signals %s] 视频不存在: %s", vid, video)
        return None

    insp_env = common.read_result_json(cfg.paths.perception_dir / vid / "inspect")
    duration = float(((insp_env or {}).get("output") or {}).get("duration_s") or 0)
    ff_bin = cfg.perception.get("ffprobe_bin", "ffprobe")
    if not duration:
        duration = common.video_duration_s(ff_bin, video)
    probe = common.run_ffprobe_json(ff_bin, video)
    streams = probe.get("streams") or [{}]
    src_w = int(streams[0].get("width") or 544)
    src_h = int(streams[0].get("height") or 960)

    work_fps = float(s_cfg.get("work_fps", 10))
    scale = float(s_cfg.get("scale", 0.5))
    grid = s_cfg.get("grid") or {}
    g_cols = int(grid.get("cols", 6))
    g_rows = int(grid.get("rows", 6))
    # 网格 reshape 需整除：宽按 cols、高按 rows 取整（竖屏 272 宽这类不整除会崩）
    w = max(g_cols, int(src_w * scale) // g_cols * g_cols)
    h = max(g_rows, int(src_h * scale) // g_rows * g_rows)
    frames = decode_gray_frames(cfg.perception.get("ffmpeg_bin", "ffmpeg"), video,
                                fps=work_fps, width=w, height=h)
    if len(frames) < 3:
        raise RuntimeError(f"解码帧数过少: {len(frames)}")
    logger.info("[signals %s] %d 帧 @%dx%d", vid, len(frames), w, h)

    series = compute_series(frames, grid_rows=g_rows, grid_cols=g_cols)
    fps_actual = (len(frames) - 1) / duration if duration else work_fps
    series["t"] = [round(i / fps_actual, 3) for i in range(len(frames))]

    shots_env = common.read_result_json(cfg.paths.perception_dir / vid / "shots")
    bounds = list(((shots_env or {}).get("output") or {}).get("boundaries_s") or [])
    cands = derive_event_candidates(series, shot_boundaries=bounds, thr=s_cfg)

    n_sheets = 0
    if s_cfg.get("save_debug_frames", True):
        n_sheets = save_contact_sheets(frames, series, cands, tdir / "frames")

    tdir.mkdir(parents=True, exist_ok=True)   # write_result_json 不建父目录（前车之鉴）
    path = common.write_result_json(tdir, tool="signals", aweme_id=vid,
                                    params=params, output={
                                        "fps_actual": round(fps_actual, 4),
                                        "n_frames": len(frames), "work_size": [w, h],
                                        "series": series, "candidates": cands,
                                        "n_candidates": len(cands),
                                        "contact_sheets": n_sheets})
    logger.info("[signals %s] %d 候选（%d sheet）→ %s", vid, len(cands), n_sheets, path)
    common.emit_status_line("ok", vid=vid, candidates=len(cands), sheets=n_sheets)
    return path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="剪辑信号确定性检测轨（D1）")
    ap.add_argument("--id", required=True, help="data/videos/<id>")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="lib_signals")
    try:
        run_signals(cfg, args.id, force=args.force)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("signals 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1


if __name__ == "__main__":
    sys.exit(main())
