# -*- coding: utf-8 -*-
"""M2-A 图像文件确定性工具：读写/SHA/尺寸/背景干净度/Lanczos 超分。

全部 PIL 确定性——能程序判的绝不烧 Omni（三层验收第一层）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

# 视图清单（真资产 = 四张单图；identity_sheet 只是 overview）
VIEW_IDS = ("front", "profile", "back", "face")

# 生成尺寸与 4K 目标（9:16，16 对齐）
GEN_WIDTH, GEN_HEIGHT = 1152, 2048
TARGET_4K_LONG_SIDE = 3840

# 背景干净度启发式阈值（边缘条带平均饱和度 + 亮度标准差）
BG_MAX_MEAN_SATURATION = 0.18
BG_MAX_LUMA_STD = 0.30


def save_image(image: Image.Image, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def file_sha256(path: str | Path) -> str:
    """完整 64 hex SHA-256（P2-3：SHA 是 lock/provenance 基础设施，
    底层不截断；显示端自行 [:8]）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_aspect_9_16(image: Image.Image, tolerance: float = 0.03) -> bool:
    """宽高比 9:16（±3% 容差）。"""
    ratio = image.width / image.height
    return abs(ratio - 9 / 16) <= (9 / 16) * tolerance


def check_background_clean(image: Image.Image) -> bool:
    """背景干净度：四周边缘条带（外圈 4%）低饱和 + 低亮度方差。

    中性灰无缝棚拍背景的启发式判定——阈值宽松（假 FAIL 比假 PASS
    代价高：L1 假 FAIL 会直接拒掉合格资产）。
    """
    import numpy as np  # 服务器环境有 numpy；本地测试同样依赖
    w, h = image.size
    strip = max(4, min(w, h) // 25)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    edges = [
        arr[:strip, :, :],          # top
        arr[-strip:, :, :],         # bottom
        arr[:, :strip, :],          # left
        arr[:, -strip:, :],         # right
    ]
    pixels = np.concatenate(
        [e.reshape(-1, 3) for e in edges], axis=0)
    mx, mn = pixels.max(axis=1), pixels.min(axis=1)
    saturation = float(np.mean((mx - mn) / (mx + 1e-6)))
    luma = pixels.mean(axis=1)
    luma_std = float(luma.std())
    return (saturation <= BG_MAX_MEAN_SATURATION
            and luma_std <= BG_MAX_LUMA_STD)


def upscale_lanczos(src_path: str | Path, out_path: str | Path,
                    long_side: int = TARGET_4K_LONG_SIDE) -> Path:
    """确定性 4K 派生（v1：PIL Lanczos；RealESRGAN 可后续替换）。

    命名诚实：这是尺寸 4K（derived_4k_master），不新增生成细节——
    detail_enhanced=false。分辨率检查只代表尺寸与 provenance 合法。
    """
    image = load_image(src_path)
    if image.width >= image.height:
        new_size = (long_side, round(image.height * long_side / image.width))
    else:
        new_size = (round(image.width * long_side / image.height), long_side)
    resized = image.resize(new_size, Image.LANCZOS)
    return save_image(resized, out_path)


def compose_identity_sheet(view_paths: dict[str, str | Path],
                           out_path: str | Path,
                           cell_width: int = 640) -> Path:
    """确定性 2×2 overview 拼图（人审界面，非资产本体）。

    front | profile
    back  | face
    """
    canvas_ratio = 9 / 16
    cell_height = round(cell_width / canvas_ratio)
    gap = 16
    sheet = Image.new("RGB", (cell_width * 2 + gap * 3,
                              cell_height * 2 + gap * 3),
                      (32, 32, 32))
    order = (("front", 0, 0), ("profile", 1, 0),
             ("back", 0, 1), ("face", 1, 1))
    for view, col, row in order:
        cell = load_image(view_paths[view])
        cell = cell.resize((cell_width, cell_height), Image.LANCZOS)
        x = gap + col * (cell_width + gap)
        y = gap + row * (cell_height + gap)
        sheet.paste(cell, (x, y))
    return save_image(sheet, out_path)
