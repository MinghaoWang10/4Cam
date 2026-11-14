"""结果输出：实验室布置表、俯视图/侧视图、覆盖图、误差表。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .layout import LayoutScore
from .pipeline import KneeError
from .trajectory import Trajectory
from .walkway import Walkway


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_setup_csv(path: Path, score: LayoutScore, walkway: Walkway) -> None:
    path = _ensure(Path(path))
    rows = []
    for i, (p, cam) in enumerate(zip(score.placements, score.cams), 1):
        c = cam.center
        rows.append({
            "相机": f"C{i}",
            "型号": p.spec.model,
            "镜头_mm": p.spec.focal_mm,
            "朝向": p.spec.orientation,
            "X_mm": round(float(c[0]), 1),
            "Y_mm": round(float(c[1]), 1),
            "Z_mm": round(float(c[2]), 1),
            "方位角_deg": p.azimuth_deg,
            "距离_mm": p.distance_mm,
            "高度_mm": p.height_mm,
            "瞄准X_mm": p.aim_x_mm,
            "瞄准Y_mm": p.aim_y_mm,
            "HFOV_deg": round(p.spec.fov_deg[0], 2),
            "VFOV_deg": round(p.spec.fov_deg[1], 2),
        })
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_layout_json(path: Path, score: LayoutScore, walkway: Walkway,
                      mc: dict | None = None) -> None:
    path = _ensure(Path(path))
    payload = {
        "walkway": walkway.summary(),
        "layout": score.to_dict(),
        "mc": {},
    }
    if mc:
        for (side, sigma), err in mc.items():
            payload["mc"][f"{side}_sigma{sigma}"] = {
                "rmse": np.asarray(err.rmse).tolist(),
                "p95": np.asarray(err.p95).tolist(),
                "fail_frac": float(err.fail_frac),
            }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_summary(path: Path, score: LayoutScore, walkway: Walkway,
                  mc: dict | None = None) -> None:
    path = _ensure(Path(path))
    w = walkway.summary()
    lines = [
        "=" * 72,
        "sim-v3 四相机布局结果",
        "=" * 72,
        "",
        "=== 步道几何 ===",
        f"总长: {w['length_mm']:.0f} mm  总宽: {w['width_mm']:.0f} mm",
        f"分析窗口: X {w['analysis_x_mm'][0]:.0f} ~ {w['analysis_x_mm'][1]:.0f} mm"
        f"（{w['analysis_length_mm']:.0f} mm）",
        f"进入段: {w['entry_mm']:.0f} mm  退出段: {w['exit_mm']:.0f} mm",
        f"受试者起点: X={w['subject_start_x_mm']:.0f} mm  "
        f"终点: X={w['subject_end_x_mm']:.0f} mm",
        f"步道 X 范围: {w['walkway_x_mm'][0]:.0f} ~ {w['walkway_x_mm'][1]:.0f} mm",
        f"步道 Z 范围: {w['walkway_z_mm'][0]:.0f} ~ {w['walkway_z_mm'][1]:.0f} mm",
        "",
        "=== 相机布置 ===",
    ]
    for i, (p, cam) in enumerate(zip(score.placements, score.cams), 1):
        c = cam.center
        lines.append(
            f"C{i}: {p.spec.model}  {p.spec.focal_mm:.0f} mm  {p.spec.orientation}  "
            f"位置=({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}) mm  "
            f"方位={p.azimuth_deg:+.0f}°  距离={p.distance_mm:.0f} mm  "
            f"FOV={p.spec.fov_deg[0]:.1f}°×{p.spec.fov_deg[1]:.1f}°")
    lines += [
        "",
        "=== 布局指标 ===",
        f"模板: {score.template}",
        f"失败帧: {score.fail_frac*100:.1f}%",
        f"解析旋转 RMS: {score.rot_deg:.3f}°",
        f"解析平移 RMS: {score.trans_mm:.3f} mm",
        f"标记平均覆盖(≥2视图): {score.coverage_mean:.3f}",
        f"标记最差覆盖: {score.coverage_min:.3f}",
    ]
    if score.cov is not None:
        lines += ["", "=== 逐标记覆盖 ==="]
        for name, v in score.cov.marker_cov.items():
            lines.append(f"  {name:<16} {v:.3f}")
    if mc:
        lines += ["", "=== 端到端蒙特卡洛 ==="]
        for (side, sigma), err in mc.items():
            rmse = np.asarray(err.rmse)
            lines.append(
                f"  {side} σ={sigma}: 屈伸 {rmse[0]:.3f}° 内外翻 {rmse[1]:.3f}° "
                f"内外旋 {rmse[2]:.3f}° | AP {rmse[3]:.3f} ML {rmse[4]:.3f} "
                f"牵张 {rmse[5]:.3f} mm | 失败帧 {err.fail_frac*100:.1f}%")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_top_view(path: Path, score: LayoutScore, walkway: Walkway,
                  traj: Trajectory) -> None:
    path = _ensure(Path(path))
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((walkway.x_min, walkway.z_min),
                           walkway.length_mm, walkway.width_mm,
                           fill=False, ec="0.4", lw=1.5, label="Walkway"))
    ax.add_patch(Rectangle((walkway.analysis_x_min, walkway.z_min),
                           walkway.analysis_length_mm, walkway.width_mm,
                           fc="tab:blue", alpha=0.08, ec="tab:blue",
                           lw=1.2, label="Analysis window"))
    ax.annotate("", xy=(walkway.x_max, 0), xytext=(walkway.x_min, 0),
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1.2))
    ax.text(walkway.x_min, -walkway.width_mm * 0.55, "walking +X",
            fontsize=9, color="0.4")
    colors = ["tab:red", "tab:red", "tab:green", "tab:green"]
    for i, (p, cam) in enumerate(zip(score.placements, score.cams), 1):
        c = cam.center
        d = cam.R[2]
        xaxis = cam.R[0]
        h = np.deg2rad(p.spec.fov_deg[0] / 2)
        for sign in (-1, 1):
            edge = np.cos(h) * d + sign * np.sin(h) * xaxis
            edge = edge / np.linalg.norm(edge)
            ax.plot([c[0], c[0] + 6000 * edge[0]],
                    [c[2], c[2] + 6000 * edge[2]],
                    color=colors[i - 1], alpha=0.15, lw=1)
        ax.plot([c[0], c[0] + 3000 * d[0]], [c[2], c[2] + 3000 * d[2]],
                color=colors[i - 1], lw=1.2, ls="--")
        ax.scatter(c[0], c[2], color=colors[i - 1], s=60, zorder=5)
        ax.text(c[0], c[2] + 150, f"C{i} {p.spec.model}",
                fontsize=8, ha="center", color=colors[i - 1])
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm, walking)")
    ax.set_ylabel("Z (mm, right +)")
    ax.invert_yaxis()
    ax.set_title("Top view: walkway and camera layout")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_side_view(path: Path, score: LayoutScore, walkway: Walkway,
                   traj: Trajectory) -> None:
    path = _ensure(Path(path))
    plt = _setup_matplotlib()
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([walkway.x_min, walkway.x_max], [0, 0], color="0.3", lw=2)
    ax.plot([walkway.analysis_x_min, walkway.analysis_x_max], [0, 0],
            color="tab:blue", lw=5, alpha=0.3, label="analysis window")
    pts = traj.xyz.reshape(-1, 3)
    ax.add_patch(Rectangle(
        (pts[:, 0].min(), pts[:, 1].min()),
        pts[:, 0].max() - pts[:, 0].min(),
        pts[:, 1].max() - pts[:, 1].min(),
        fill=False, ec="0.6", ls=":", label="marker envelope"))
    colors = ["tab:red", "tab:red", "tab:green", "tab:green"]
    for i, (p, cam) in enumerate(zip(score.placements, score.cams), 1):
        c = cam.center
        aim = walkway.aim_point(p.aim_x_mm, p.aim_y_mm)
        ax.plot([c[0], aim[0]], [c[1], aim[1]], color=colors[i - 1],
                lw=1.2, ls="--")
        ax.scatter(c[0], c[1], color=colors[i - 1], s=60, zorder=5)
        ax.text(c[0], c[1] + 60, f"C{i}", fontsize=9,
                ha="center", color=colors[i - 1])
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm, walking)")
    ax.set_ylabel("Y (mm, up)")
    ax.set_title("Side view: camera heights and optical axes")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_coverage(path: Path, score: LayoutScore) -> None:
    path = _ensure(Path(path))
    plt = _setup_matplotlib()
    if score.cov is None:
        return
    names = list(score.cov.marker_cov.keys())
    vals = [score.cov.marker_cov[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(names, vals, color="tab:blue")
    ax.axvline(0.95, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of frames seen by >=2 cameras")
    ax.set_title("Per-marker coverage")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
