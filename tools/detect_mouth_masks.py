# -*- coding: utf-8 -*-
from __future__ import annotations
"""
tools/detect_mouth_masks.py
===========================
自动检测每个表情的嘴部区域，并计算 loop 帧序列的"开合量"，
输出 config/mouth_masks.json，供 PixiJS 口型同步使用。

用法：
    python tools/detect_mouth_masks.py [--debug]

依赖：
    pip install opencv-python numpy

输出格式 (config/mouth_masks.json)：
{
  "normal": {
    "cx": 12.0,          # 嘴部中心，相对 sprite 中心的像素偏移
    "cy": 85.0,
    "width": 48.0,       # 检测到的嘴部宽度
    "height": 22.0,      # 检测到的嘴部高度（最大开合）
    "curve": -0.12,      # 上唇弧度：正=smile，负=frown，0=直线
    "frames": [          # 每帧的开合量（0=闭合，1=最大张开），与 loop 帧顺序一一对应
      0.02, 0.15, 0.41, ...
    ]
  },
  ...
}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "render" / "assets" / "images"
OUTPUT_PATH = ROOT / "config" / "mouth_masks.json"

# 嘴部在 sprite 垂直方向的搜索范围（相对于非透明区域的高度）
# 半身立绘：头部约占 0~40%，嘴在 30~42% 处
MOUTH_Y_RANGE = (0.28, 0.44)

# 嘴部在水平方向的搜索范围（相对于非透明区域的宽度）
# 中间 50%，排除两侧长发区域的抗锯齿干扰
MOUTH_X_CENTER_RATIO = 0.30   # 只看中间这个比例的宽度（排除两侧头发）

# 差分检测阈值（像素差超过此值才计入变化）
DIFF_THRESHOLD = 18

# 形态学清理核大小
MORPH_KERNEL = 5

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def load_frames(paths: list[Path]) -> list[np.ndarray]:
    """加载 RGBA 帧列表，跳过读取失败的文件。"""
    frames = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        frames.append(img)
    return frames


def sprite_bbox(frame: np.ndarray):
    """返回非透明像素的 bounding box (y0, y1, x0, x1)。"""
    alpha = frame[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(y0), int(y1), int(x0), int(x1)


def compute_mouth_roi(frames: list[np.ndarray], binarize_thresh: float = 0.25):
    """
    通过帧间差分在下半脸找嘴部区域。
    返回 (roi_mask, cx, cy, width, height, curve)，坐标均为 sprite 中心原点。

    策略：
    - 帧数 ≤ 6 时用全部帧 vs. 参考帧做差分（变化少，需要低阈值）
    - 帧数 > 6 时只取前 MAX_REF_FRAMES 帧与 frames[0] 对比，
      避免长序列中头发/衣服的大幅运动淹没嘴部信号。
    - 对检测结果施加硬尺寸约束：嘴宽 ≤ sprite_width × 0.18。
    """
    h, w = frames[0].shape[:2]

    # 先找 sprite 非透明区域的整体 bbox
    bbox = sprite_bbox(frames[0])
    if bbox is None:
        return None
    sy0, sy1, sx0, sx1 = bbox
    sh = sy1 - sy0   # sprite 实际高度
    sw = sx1 - sx0   # sprite 实际宽度

    # 嘴部搜索 ROI：垂直用比例区间，水平取中间段（排除两侧长发）
    roi_y0 = sy0 + int(sh * MOUTH_Y_RANGE[0])
    roi_y1 = sy0 + int(sh * MOUTH_Y_RANGE[1])
    cx_sprite = (sx0 + sx1) // 2
    half_x = int(sw * MOUTH_X_CENTER_RATIO / 2)
    roi_x0 = max(sx0, cx_sprite - half_x)
    roi_x1 = min(sx1, cx_sprite + half_x)

    # 构建变化图
    MAX_REF_FRAMES = 40   # 只用前 N 帧，限制长序列头发漂移影响
    n_frames = len(frames)
    use_n = min(n_frames, MAX_REF_FRAMES) if n_frames > 6 else n_frames
    ref = frames[0].astype(np.float32)
    change_map = np.zeros((h, w), dtype=np.float32)
    for i in range(1, use_n):
        diff = np.abs(frames[i].astype(np.float32) - ref)
        change_map += diff[:, :, :3].max(axis=2)

    # 只保留嘴部 ROI（垂直 + 水平双重限制）
    roi_change = np.zeros_like(change_map)
    roi_change[roi_y0:roi_y1, roi_x0:roi_x1] = change_map[roi_y0:roi_y1, roi_x0:roi_x1]

    if roi_change.max() < 1e-6:
        return None

    # 二值化（高阈值提取最显著变化区域）
    norm = roi_change / roi_change.max()
    binary = (norm > binarize_thresh).astype(np.uint8) * 255

    # 形态学清理：先闭运算（填孔），再开运算（去噪）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 嘴宽硬约束：过滤掉宽度超过 sprite 宽 18% 的轮廓，优先取最紧凑的候选
    MAX_MOUTH_W = sw * 0.18
    valid = []
    for c in contours:
        if cv2.contourArea(c) < 60:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw <= MAX_MOUTH_W:
            valid.append(c)

    if not valid:
        # 宽度限制过滤掉了所有候选：退化到最小轮廓（最可能是嘴）
        valid = [min(contours, key=lambda c: cv2.boundingRect(c)[2])]

    # 在合法候选中取面积最大的
    largest = max(valid, key=cv2.contourArea)
    if cv2.contourArea(largest) < 60:
        return None

    # 拟合椭圆
    if len(largest) >= 5:
        (ex, ey), (ea, eb), angle = cv2.fitEllipse(largest)
        mw = max(ea, eb)
        mh = min(ea, eb)
    else:
        bx, by, bw, bh = cv2.boundingRect(largest)
        ex, ey = bx + bw / 2, by + bh / 2
        mw, mh = float(bw), float(bh)

    # 尺寸最终限幅
    mw = min(mw, MAX_MOUTH_W)
    mh = min(mh, sh * 0.12)   # 嘴高 ≤ sprite 高 12%

    # 估计上唇弧度：取轮廓中上 30% 点，拟合二次曲线
    pts = largest.reshape(-1, 2)
    top_thresh = ey - mh * 0.3
    top_pts = pts[pts[:, 1] < top_thresh + 4]
    curve = 0.0
    if len(top_pts) >= 4:
        try:
            coeffs = np.polyfit(top_pts[:, 0].astype(float),
                                top_pts[:, 1].astype(float), 2)
            raw_curve = float(coeffs[0]) * (mw ** 2) / max(mh, 1.0)
            curve = float(np.clip(raw_curve, -1.5, 1.5))
        except Exception:
            pass

    # 转换到 sprite 中心坐标系
    cx_c = ex - w / 2
    cy_c = ey - h / 2

    roi_mask = binary
    return roi_mask, float(cx_c), float(cy_c), float(mw), float(mh), float(curve)


def compute_frame_openness(frames: list[np.ndarray], roi_mask: np.ndarray) -> list[float]:
    """
    计算每帧相对于参考帧在 ROI 内的开合量（0=闭合，1=最大张开）。
    参考帧取所有帧中在 ROI 内变化最小的那帧（"最闭合帧"）。
    """
    h, w = frames[0].shape[:2]
    mask_3d = roi_mask.astype(bool)

    # 逐帧计算 ROI 内的像素均值（RGB 三通道）
    def roi_mean(frame):
        rgb = frame[:, :, :3].astype(np.float32)
        return rgb[mask_3d].mean(axis=0) if mask_3d.any() else np.zeros(3)

    means = [roi_mean(f) for f in frames]

    # 找彼此之间差异最小的帧作为"闭合参考"（对比所有帧对）
    # 简化：取整体平均最接近中值的帧
    mean_arr = np.array(means)          # (N, 3)
    median_val = np.median(mean_arr, axis=0)
    dists_to_median = np.linalg.norm(mean_arr - median_val, axis=1)
    ref_idx = int(np.argmin(dists_to_median))
    ref_mean = means[ref_idx]

    # 每帧与参考帧的 L2 距离 = 开合量（未归一化）
    raw = np.array([np.linalg.norm(m - ref_mean) for m in means])
    max_val = raw.max()
    if max_val < 1e-6:
        return [0.0] * len(frames)

    openness = (raw / max_val).tolist()
    return openness


# ──────────────────────────────────────────────
# 主检测逻辑
# ──────────────────────────────────────────────

def collect_frame_paths(expr_dir: Path) -> list[Path]:
    """
    按优先级收集帧路径：
    1. loop/ 子目录（三段式）
    2. in/ 子目录（如果没有 loop）
    3. 直接在目录下的帧文件
    """
    loop_dir = expr_dir / "loop"
    in_dir = expr_dir / "in"

    if loop_dir.exists():
        paths = sorted(loop_dir.glob("*.png"))
        if paths:
            return paths

    if in_dir.exists():
        paths = sorted(in_dir.glob("*.png"))
        if paths:
            return paths

    paths = sorted(expr_dir.glob("*.png"))
    return [p for p in paths if not p.name.startswith(".")]


def process_expression(expr_name: str, expr_dir: Path) -> dict | None:
    frame_paths = collect_frame_paths(expr_dir)
    if len(frame_paths) < 2:
        print(f"  [{expr_name}] 帧数不足 ({len(frame_paths)})，跳过")
        return None

    print(f"  [{expr_name}] 加载 {len(frame_paths)} 帧...")
    frames = load_frames(frame_paths)
    if len(frames) < 2:
        print(f"  [{expr_name}] 图片读取失败")
        return None

    result = compute_mouth_roi(frames, binarize_thresh=0.25)
    if result is None:
        # 低阈值重试（适用于帧间变化微弱的 3 帧静态表情）
        result = compute_mouth_roi(frames, binarize_thresh=0.06)
    if result is None:
        print(f"  [{expr_name}] 嘴部检测失败（变化量不足或无轮廓）")
        return None

    roi_mask, cx, cy, width, height, curve = result
    openness = compute_frame_openness(frames, roi_mask)

    print(f"  [{expr_name}] cx={cx:.1f} cy={cy:.1f} w={width:.1f} h={height:.1f} "
          f"curve={curve:.3f} | 开合范围 {min(openness):.2f}~{max(openness):.2f}")

    # 保存帧名（相对路径）供渲染器加载
    frame_names = [p.name for p in frame_paths]

    openness_arr = np.array(openness)
    closed_frame_idx = int(np.argmin(openness_arr))
    open_frame_idx = int(np.argmax(openness_arr))

    return {
        "cx": round(cx, 1),
        "cy": round(cy, 1),
        "width": round(width, 1),
        "height": round(height, 1),
        "curve": round(curve, 3),
        "closed_frame_idx": closed_frame_idx,
        "open_frame_idx": open_frame_idx,
        "frame_names": frame_names,
        "openness": [round(v, 4) for v in openness],
        # 内部用：调试图需要原始帧和 roi_mask
        "_debug_frames": frames,
        "_debug_roi_mask": roi_mask,
    }


def _save_debug_image(expr_name, frames, roi_mask, cx, cy, width, height):
    """将检测结果可视化，保存到 tools/debug_mouth/ 目录。"""
    debug_dir = Path(__file__).parent / "debug_mouth"
    debug_dir.mkdir(exist_ok=True)

    # 取第一帧画框
    vis = frames[0].copy()
    h, w = vis.shape[:2]
    cx_abs = cx + w / 2
    cy_abs = cy + h / 2

    # 画 ROI mask
    mask_color = np.zeros_like(vis)
    mask_color[:, :, 2] = roi_mask   # 红色通道
    mask_color[:, :, 3] = (roi_mask > 0).astype(np.uint8) * 120
    vis = cv2.addWeighted(vis, 1.0, mask_color, 0.5, 0)

    # 画椭圆
    cv2.ellipse(vis,
                (int(cx_abs), int(cy_abs)),
                (int(width / 2), int(height / 2)),
                0, 0, 360,
                (0, 255, 0, 255), 2)

    out_path = debug_dir / f"{expr_name}_mouth.png"
    cv2.imwrite(str(out_path), vis)
    print(f"  [{expr_name}] 调试图已保存: {out_path}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="自动检测立绘嘴部区域")
    parser.add_argument("--debug", action="store_true", help="保存调试可视化图片")
    parser.add_argument("--expr", nargs="*", help="只处理指定表情（默认全部）")
    args = parser.parse_args()

    if not IMAGES_DIR.exists():
        print(f"错误：找不到图片目录 {IMAGES_DIR}")
        sys.exit(1)

    # 枚举表情目录（排除非目录和背景文件）
    expr_dirs = sorted([
        d for d in IMAGES_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if args.expr:
        expr_dirs = [d for d in expr_dirs if d.name in args.expr]
        if not expr_dirs:
            print(f"未找到指定表情: {args.expr}")
            sys.exit(1)

    print(f"共找到 {len(expr_dirs)} 个表情目录")
    print(f"图片根目录: {IMAGES_DIR}")

    results = {}
    for expr_dir in expr_dirs:
        data = process_expression(expr_dir.name, expr_dir)
        if data:
            results[expr_dir.name] = data

    # ── 同尺寸组内中位数修正 ──────────────────────────────────────────
    # 同一角色不同表情的嘴部 cx/cy 应基本一致（图像尺寸相同时）。
    # 按 cy 范围分组（cy < -140：大图组；cy >= -140：小图组），
    # 计算组内 cx/cy 中位数，将偏差超过 35px 的离群值替换为中位数。
    # 大图组用最负（最高）的 cy 作为参考（更靠近嘴部而非领口）
    _median = lambda arr: sorted(arr)[len(arr) // 2]
    _ref_cy = lambda arr: min(arr)   # 最负 = 图像最高位置 = 嘴，而非领口

    for threshold_cy, label in [(-120, "大图组"), (-50, "小图组")]:
        group = {k: v for k, v in results.items()
                 if (v["cy"] < threshold_cy if label == "大图组" else v["cy"] >= threshold_cy)}
        if len(group) < 2:
            continue
        med_cx = _median([v["cx"] for v in group.values()])
        # 大图组用最高位（最负 cy）作参考，防止领口检测污染中位数
        med_cy = (_ref_cy([v["cy"] for v in group.values()])
                  if label == "大图组" else _median([v["cy"] for v in group.values()]))
        for name, data in group.items():
            if abs(data["cx"] - med_cx) > 35 or abs(data["cy"] - med_cy) > 35:
                print(f"  [{name}] 位置离群 cx={data['cx']:.1f} cy={data['cy']:.1f}"
                      f" → 修正为组内中位数 cx={med_cx:.1f} cy={med_cy:.1f}")
                data["cx"] = round(med_cx, 1)
                data["cy"] = round(med_cy, 1)
    # ─────────────────────────────────────────────────────────────────

    # 修正后保存调试图（坐标与 JSON 一致）
    if args.debug:
        for name, data in results.items():
            frames_dbg = data.pop("_debug_frames", None)
            roi_dbg = data.pop("_debug_roi_mask", None)
            if frames_dbg is not None and roi_dbg is not None:
                _save_debug_image(name, frames_dbg, roi_dbg,
                                  data["cx"], data["cy"], data["width"], data["height"])
    else:
        for data in results.values():
            data.pop("_debug_frames", None)
            data.pop("_debug_roi_mask", None)

    # 写出结果（去掉临时调试字段）
    clean = {k: {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
             for k, v in results.items()}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"version": 1, "expressions": clean}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n[OK] 已写出 {len(clean)} 个表情的嘴部配置到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
