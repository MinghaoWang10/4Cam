"""4Cam 主入口：搜索 2 Basler + 2 海康的四相机布局与步道几何。

用法（在 simulation 目录下）：
    python scripts/search_layout.py
    python scripts/search_layout.py --travel 1300 --sides RL --sigma 0.3
    python scripts/search_layout.py --top-k 3 --mc-n 80
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import layout, report, trajectory  # noqa: E402
from sim.walkway import default_walkway  # noqa: E402


def fmt_layout(score) -> str:
    lines = [f"模板: {score.template}",
             f"失败帧: {score.fail_frac*100:.1f}%  "
             f"旋转: {score.rot_deg:.3f}°  平移: {score.trans_mm:.3f} mm  "
             f"覆盖均值: {score.coverage_mean:.3f}  最差: {score.coverage_min:.3f}"]
    for i, (p, cam) in enumerate(zip(score.placements, score.cams), 1):
        c = cam.center
        lines.append(
            f"  C{i} {p.spec.model:<7} {p.spec.orientation:<9} "
            f"({c[0]:7.0f},{c[1]:6.0f},{c[2]:7.0f}) mm  "
            f"az={p.azimuth_deg:+5.0f}°  d={p.distance_mm:.0f}  h={p.height_mm:.0f}  "
            f"FOV={p.spec.fov_deg[0]:.1f}×{p.spec.fov_deg[1]:.1f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="4Cam 异构四相机布局搜索")
    ap.add_argument("--travel", type=float, default=1300.0,
                    help="一个步态周期的前进距离 (mm)")
    ap.add_argument("--sides", default="RL", choices=("R", "L", "RL"))
    ap.add_argument("--sigma", type=float, default=0.3, help="全局像素噪声 σ (px)")
    ap.add_argument("--distances", default="2500,3000,3500,4000,4500",
                    help="相机距离候选 (mm)，逗号分隔")
    ap.add_argument("--heights", default="300,600,900,1200,1500,1800,2100,2400",
                    help="相机高度候选 (mm)，逗号分隔")
    ap.add_argument("--aims", default="-500,0,500",
                    help="瞄准点 X 偏移候选 (mm)，逗号分隔")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--mc-n", type=int, default=80, help="蒙特卡洛次数")
    ap.add_argument("--out", default="results", help="输出目录")
    args = ap.parse_args()

    t0 = time.time()
    traj = trajectory.load(travel_mm=args.travel)
    walkway = default_walkway()
    print("=" * 72)
    print(f"4Cam 布局搜索：1 周期，前进 {args.travel:.0f} mm，"
          f"{args.sides}，σ={args.sigma} px")
    print(f"步道：{walkway.length_mm:.0f} × {walkway.width_mm:.0f} mm，"
          f"分析窗口 {walkway.analysis_length_mm:.0f} mm")
    print("=" * 72, flush=True)

    results = layout.search(traj, walkway, sides=args.sides,
                            distances_mm=tuple(float(x) for x in args.distances.split(",")),
                            heights_mm=tuple(float(x) for x in args.heights.split(",")),
                            aims_mm=tuple(float(x) for x in args.aims.split(",")),
                            sigma_px=args.sigma, top_k=args.top_k)
    if not results:
        print("[ERROR] 没有候选布局通过 FOV/角间隔预筛。")
        sys.exit(1)

    print(f"\n粗筛+局部优化完成，用时 {time.time()-t0:.1f}s，前 {len(results)} 名：")
    for i, sc in enumerate(results, 1):
        print(f"\n--- 第 {i} 名 ---")
        print(fmt_layout(sc))

    best = results[0]
    if best.fail_frac > 0:
        print("\n[WARN] 最优布局仍有失败帧；请放宽步道/增加距离/调整模板。")

    print("\n端到端蒙特卡洛验证中 ...", flush=True)
    mc = layout.monte_carlo_validate(traj, best, args.sides,
                                     sigmas=(0.1, args.sigma), n_mc=args.mc_n)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    report.write_setup_csv(outdir / "setup_sheet.csv", best, walkway)
    report.write_layout_json(outdir / "layout_recommended.json", best, walkway, mc)
    report.write_summary(outdir / "summary.txt", best, walkway, mc)
    report.plot_top_view(outdir / "top_view.png", best, walkway, traj)
    report.plot_side_view(outdir / "side_view.png", best, walkway, traj)
    report.plot_coverage(outdir / "coverage_map.png", best)

    print(f"\n输出已写入：{outdir.resolve()}")
    print("  setup_sheet.csv / layout_recommended.json / summary.txt / "
          "top_view.png / side_view.png / coverage_map.png")
    print(f"总用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
