"""异构四相机布局搜索：2 台 Basler + 2 台海康，1 个步态周期。

搜索策略：
  1. 角色模板（十字、斜向、侧向+前后等）固定 4 台相机的型号/方位/朝向；
  2. 阶段 A：模板 × 距离 × 高度 × 瞄准点，粗筛；
  3. 阶段 B：对前几名逐台做距离/高度/瞄准点局部优化；
  4. 阶段 C：对前几名做端到端蒙特卡洛（由脚本调用）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import BASLER, HIK, Camera, CameraSpec
from .detection import DetectionModel, detect
from .kinematics import grood_suntay
from .pipeline import (KNEE_SEGMENTS, camera_sigmas, coverage,
                       knee_monte_carlo, segment_markers, views)
from .trajectory import Trajectory
from .walkway import Walkway


# (型号, 方位角, 朝向)；每个模板固定 2 Basler + 2 海康
TEMPLATES: dict[str, list[tuple[str, float, str]]] = {
    "十字_B侧向_H前后": [
        ("Basler", -90, "landscape"), ("Basler", 90, "landscape"),
        ("Hik", 0, "portrait"), ("Hik", 180, "portrait"),
    ],
    "十字_B前后_H侧向": [
        ("Basler", 0, "landscape"), ("Basler", 180, "landscape"),
        ("Hik", -90, "landscape"), ("Hik", 90, "landscape"),
    ],
    "斜向_B前斜_H后斜": [
        ("Basler", -45, "landscape"), ("Basler", 45, "landscape"),
        ("Hik", -135, "landscape"), ("Hik", 135, "landscape"),
    ],
    "侧向_B_H斜向": [
        ("Basler", -90, "landscape"), ("Basler", 90, "landscape"),
        ("Hik", -45, "landscape"), ("Hik", 135, "landscape"),
    ],
    "前后_B_H斜向": [
        ("Basler", 0, "landscape"), ("Basler", 180, "landscape"),
        ("Hik", -45, "landscape"), ("Hik", 135, "landscape"),
    ],
    "四角_B斜_H后斜": [
        ("Basler", -45, "landscape"), ("Basler", 45, "landscape"),
        ("Hik", -135, "landscape"), ("Hik", 135, "landscape"),
    ],
}


@dataclass(frozen=True)
class CameraPlacement:
    spec: CameraSpec
    azimuth_deg: float
    distance_mm: float
    height_mm: float
    aim_x_mm: float = 0.0
    aim_y_mm: float = 550.0
    name: str = ""

    def camera(self, walkway: Walkway) -> Camera:
        return walkway.camera(self.spec, self.azimuth_deg, self.distance_mm,
                              self.height_mm, self.aim_x_mm, self.aim_y_mm,
                              self.name or None)

    def to_dict(self) -> dict:
        cam = None
        return {
            "name": self.name,
            "model": self.spec.model,
            "lens_mm": self.spec.focal_mm,
            "orientation": self.spec.orientation,
            "azimuth_deg": self.azimuth_deg,
            "distance_mm": self.distance_mm,
            "height_mm": self.height_mm,
            "aim_x_mm": self.aim_x_mm,
            "aim_y_mm": self.aim_y_mm,
            "hfov_deg": self.spec.fov_deg[0],
            "vfov_deg": self.spec.fov_deg[1],
        }


@dataclass
class LayoutScore:
    template: str
    fail_frac: float
    rot_deg: float
    trans_mm: float
    objective: float
    coverage_mean: float
    coverage_min: float
    placements: list[CameraPlacement]
    cams: list[Camera]
    cov: object | None = None

    def to_dict(self) -> dict:
        return {
            "template": self.template,
            "fail_frac": self.fail_frac,
            "rot_deg": self.rot_deg,
            "trans_mm": self.trans_mm,
            "objective": self.objective,
            "coverage_mean": self.coverage_mean,
            "coverage_min": self.coverage_min,
            "placements": [p.to_dict() for p in self.placements],
            "marker_coverage": (self.cov.marker_cov if self.cov is not None else {}),
            "segment_frames_ok": (self.cov.seg_frames_ok if self.cov is not None else {}),
        }


def _skew(v: np.ndarray) -> np.ndarray:
    S = np.zeros(v.shape + (3,))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def required_fov(cam: Camera, points: np.ndarray, margin_deg: float = 4.0):
    """相机光轴对准包络中心时覆盖全部 points 所需的最小 FOV。"""
    Xc = cam.to_cam(points)
    z = Xc[:, 2]
    ok = z > 1.0
    if not np.any(ok):
        return float("inf"), float("inf")
    tan_h = np.abs(Xc[ok, 0] / z[ok])
    tan_v = np.abs(Xc[ok, 1] / z[ok])
    return (2.0 * np.degrees(np.arctan(np.max(tan_h))) + margin_deg,
            2.0 * np.degrees(np.arctan(np.max(tan_v))) + margin_deg)


def fov_fits(cam: Camera, points: np.ndarray, margin_deg: float = 4.0) -> bool:
    req_h, req_v = required_fov(cam, points, margin_deg)
    if cam.spec is None:
        return True
    have_h, have_v = cam.spec.fov_deg
    return req_h <= have_h + 1e-6 and req_v <= have_v + 1e-6


def analytic_knee_error(traj: Trajectory, cams: list[Camera], nviews: np.ndarray,
                        sides: str, sigma_px, vis: np.ndarray | None = None
                        ) -> tuple[float, float, float]:
    """解析 6DOF 误差（线性化协方差传播）。返回 (fail_frac, rot_deg, trans_mm)。"""
    T, M = traj.T, traj.M
    X = traj.points()
    sigmas = camera_sigmas(cams, sigma_px)
    info = np.zeros((len(X), 3, 3))
    for v, cam in enumerate(cams):
        if vis is None:
            ok = detect(cam, X, traj.occ) == 0
        else:
            # vis (T,M,V) -> 展平顺序 n = t + m*T
            ok = vis[:, :, v].T.reshape(-1)
        idx = np.flatnonzero(ok)
        if idx.size:
            J = cam.jacobian(X[idx])
            info[idx] += np.einsum("nij,nik->njk", J, J) / sigmas[v] ** 2

    fail = np.zeros(T, bool)
    worst_rot = worst_tr = 0.0
    for side in sides:
        rot_cov = np.zeros((T, 3, 3))
        org_cov = np.zeros((T, 3, 3))
        for body in KNEE_SEGMENTS[side]:
            idx = traj.markers_on(body)
            s = (idx[:, None] * T + np.arange(T)[None]).T
            use = nviews[:, idx] >= 2
            ok = use.sum(axis=1) >= 3
            fail |= ~ok
            P = X[s]
            w = use[..., None].astype(float)
            c = (P * w).sum(1) / np.maximum(w.sum(1), 1)
            d = P - c[:, None]
            A = np.zeros(s.shape + (3, 6))
            A[..., :3] = -_skew(d)
            A[..., 3:] = np.eye(3)
            I3 = info[s] + np.eye(3) * 1e-9
            Sig = np.linalg.inv(I3)
            Aw = A * use[..., None, None]
            H = np.einsum("tnia,tnib->tab", Aw, A)
            Bm = np.einsum("tnia,tnij,tnjb->tab", Aw,
                           Sig * use[..., None, None], A)
            H[~ok] = np.eye(6)
            Hi = np.linalg.inv(H)
            cov = Hi @ Bm @ Hi
            o = traj.body_origin[:, traj.body(KNEE_SEGMENTS[side][1])]
            G = np.zeros((T, 3, 6))
            G[:, :, :3] = -_skew(o - c)
            G[:, :, 3:] = np.eye(3)
            rot_cov += cov[:, :3, :3]
            org_cov += G @ cov @ np.transpose(G, (0, 2, 1))
        good = ~fail
        if good.any():
            rot = np.rad2deg(np.sqrt(np.trace(rot_cov[good], axis1=1, axis2=2).mean() / 3))
            trn = np.sqrt(np.trace(org_cov[good], axis1=1, axis2=2).mean() / 3)
            worst_rot, worst_tr = max(worst_rot, rot), max(worst_tr, trn)
        else:
            worst_rot = worst_tr = np.inf
    return float(fail.mean()), float(worst_rot), float(worst_tr)


def evaluate_layout(traj: Trajectory, cams: list[Camera], sides: str,
                    sigma_px=0.3, model: DetectionModel = DetectionModel()) -> LayoutScore:
    """评价一个 4 相机布局：覆盖 + 解析 6DOF。"""
    vis = views(traj, cams, model)
    nviews = vis.sum(axis=2)
    cov = coverage(traj, nviews, sides)
    fail, rot, trn = analytic_knee_error(traj, cams, nviews, sides, sigma_px, vis=vis)
    # 目标：旋转/平移都压到 0.5 以内；同时避免个别标记覆盖过低导致 MC 误差变大
    coverage_penalty = 2.0 * max(0.0, 0.75 - cov.min_cov)
    objective = rot / 0.5 + trn / 0.35 + coverage_penalty
    return LayoutScore("", fail, rot, trn, objective,
                       cov.mean_cov, cov.min_cov, [], cams, cov)


def make_layout(template: str, walkway: Walkway, distance_mm: float,
                height_mm: float, aim_x_mm: float, aim_y_mm: float = 550.0,
                lens_basler: float = 12.0, lens_hik: float = 12.0) -> list[CameraPlacement]:
    """按模板 + 共享距离/高度/瞄准点生成 4 台相机。"""
    placements = []
    for i, (model, az, orientation) in enumerate(TEMPLATES[template], 1):
        base = BASLER if model == "Basler" else HIK
        spec = base.with_lens(lens_basler if model == "Basler" else lens_hik)
        spec = spec.with_orientation(orientation)
        placements.append(CameraPlacement(
            spec, az, distance_mm, height_mm, aim_x_mm, aim_y_mm,
            f"C{i}_{model}_{az:+.0f}"))
    return placements


def layout_cameras(placements: list[CameraPlacement], walkway: Walkway) -> list[Camera]:
    return [p.camera(walkway) for p in placements]


def check_fov(placements: list[CameraPlacement], walkway: Walkway,
              traj: Trajectory, sides: str, margin_deg: float = 4.0) -> bool:
    """所有相机都要覆盖所选标记包络（保守预筛）。"""
    idx = np.array([i for i, name in enumerate(traj.names) if name[0] in sides])
    points = traj.xyz[:, idx, :].reshape(-1, 3)
    for p in placements:
        cam = p.camera(walkway)
        if not fov_fits(cam, points, margin_deg):
            return False
    return True


def angle_separation_ok(placements: list[CameraPlacement], min_deg: float = 30.0) -> bool:
    """任意两台相机方位角间隔 >= min_deg（避免挤在一起）。"""
    az = np.array([p.azimuth_deg for p in placements])
    for i in range(len(az)):
        for j in range(i + 1, len(az)):
            d = abs((az[i] - az[j] + 180.0) % 360.0 - 180.0)
            if d < min_deg:
                return False
    return True


def local_improve(traj: Trajectory, walkway: Walkway, placements: list[CameraPlacement],
                  sides: str, sigma_px=0.3, max_rounds: int = 2,
                  template: str = ""):
    """逐台相机做距离/高度/瞄准点的坐标下降。"""
    best = evaluate_layout(traj, layout_cameras(placements, walkway), sides, sigma_px)
    best.template = template
    best_placements = list(placements)
    for _ in range(max_rounds):
        improved = False
        for i in range(len(best_placements)):
            p = best_placements[i]
            candidates = []
            for dd in (-0.5, 0.0, 0.5):
                for dh in (-0.3, 0.0, 0.3):
                    for da in (-0.5, 0.0, 0.5):
                        if dd == dh == da == 0:
                            continue
                        cand = CameraPlacement(
                            p.spec, p.azimuth_deg,
                            min(4500.0, max(2000.0, p.distance_mm + dd * 1000.0)),
                            min(2400.0, max(300.0, p.height_mm + dh * 1000.0)),
                            min(1000.0, max(-1000.0, p.aim_x_mm + da * 1000.0)),
                            p.aim_y_mm, p.name)
                        trial = list(best_placements)
                        trial[i] = cand
                        if not angle_separation_ok(trial):
                            continue
                        if not check_fov(trial, walkway, traj, sides):
                            continue
                        sc = evaluate_layout(traj, layout_cameras(trial, walkway), sides, sigma_px)
                        if sc.fail_frac <= best.fail_frac and sc.objective < best.objective - 1e-6:
                            candidates.append((sc, cand))
            if candidates:
                sc, cand = min(candidates, key=lambda x: x[0].objective)
                best_placements[i] = cand
                best = sc
                best.template = template
                improved = True
        if not improved:
            break
    best.placements = best_placements
    best.cams = layout_cameras(best_placements, walkway)
    return best


def search(traj: Trajectory, walkway: Walkway, sides: str = "RL",
           distances_mm=(2500, 3000, 3500, 4000, 4500),
           heights_mm=(300, 600, 900, 1200, 1500, 1800, 2100, 2400),
           aims_mm=(-500, 0, 500),
           sigma_px=0.3, top_k: int = 5) -> list[LayoutScore]:
    """阶段 A 粗筛 + 阶段 B 局部优化，返回前 top_k 个布局。"""
    results: list[LayoutScore] = []
    for name in TEMPLATES:
        for d in distances_mm:
            for h in heights_mm:
                for aim in aims_mm:
                    placements = make_layout(name, walkway, float(d), float(h), float(aim))
                    if not angle_separation_ok(placements):
                        continue
                    if not check_fov(placements, walkway, traj, sides):
                        continue
                    cams = layout_cameras(placements, walkway)
                    sc = evaluate_layout(traj, cams, sides, sigma_px)
                    sc.template = name
                    sc.placements = placements
                    results.append(sc)
    if not results:
        return []
    results.sort(key=lambda s: (s.fail_frac, s.objective))
    # 取全局前 5 个进入局部优化，避免每个模板都做昂贵的坐标下降
    seeds = results[:5]
    improved = [local_improve(traj, walkway, s.placements, sides, sigma_px,
                              template=s.template) for s in seeds]
    improved.sort(key=lambda s: (s.fail_frac, s.objective))
    return improved[:top_k]


def monte_carlo_validate(traj: Trajectory, score: LayoutScore, sides: str,
                         sigmas=(0.1, 0.3), n_mc: int = 100):
    """对布局做端到端 MC，返回每个膝/σ 的 KneeError。"""
    vis = views(traj, score.cams)
    out = {}
    for side in sides:
        for sp in sigmas:
            out[(side, sp)] = knee_monte_carlo(traj, score.cams, vis, side, sp, n_mc=n_mc)
    return out
