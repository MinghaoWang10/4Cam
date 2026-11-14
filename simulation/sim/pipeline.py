"""布局评价：多相机检出、覆盖统计和端到端 6DOF 蒙特卡洛。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .detection import DetectionModel, detect_traj
from .kinematics import grood_suntay
from .trajectory import Trajectory

# 每个膝关节参与位姿拟合的体段与标记点（足跟在跟骨上，不属于胫骨，不参与）
KNEE_SEGMENTS = {
    "R": ("femur_r", "tibia_r"),
    "L": ("femur_l", "tibia_l"),
}


def views(traj: Trajectory, cams: list[Camera],
          model: DetectionModel = DetectionModel()) -> np.ndarray:
    """(T, M, V) bool，相机 v 在帧 t 检出标记点 m。"""
    return np.stack([detect_traj(c, traj, model) == 0 for c in cams], axis=2)


def segment_markers(traj: Trajectory, side: str) -> tuple[np.ndarray, np.ndarray]:
    fem, tib = KNEE_SEGMENTS[side]
    return traj.markers_on(fem), traj.markers_on(tib)


@dataclass
class Coverage:
    marker_cov: dict[str, float]     # 各标记点 ≥2 视图的帧占比
    frames_ok: float                 # 所有目标体段同时 ≥3 个可重建标记点的帧占比
    seg_frames_ok: dict[str, float]  # 各体段 ≥3 个可重建标记点的帧占比
    mean_cov: float
    min_cov: float


def coverage(traj: Trajectory, nviews: np.ndarray, sides: str) -> Coverage:
    """nviews (T, M) 为各样本的检出相机数。sides 为 'R'、'L' 或 'RL'。"""
    rec = nviews >= 2
    seg_ok, mk = {}, []
    all_ok = np.ones(traj.T, bool)
    for s in sides:
        for name, idx in zip(KNEE_SEGMENTS[s], segment_markers(traj, s)):
            ok = rec[:, idx].sum(axis=1) >= 3
            seg_ok[name] = float(ok.mean())
            all_ok &= ok
            mk.extend(idx)
    mk = np.array(mk)
    cov = rec[:, mk].mean(axis=0)
    return Coverage({traj.names[m]: float(c) for m, c in zip(mk, cov)},
                    float(all_ok.mean()), seg_ok, float(cov.mean()), float(cov.min()))


def knee_offset(traj: Trajectory, side: str) -> np.ndarray:
    """股骨系下的膝关节参考点：胫骨原点在股骨系中的平均位置。

    OpenSim 股骨原点在髋关节中心，直接用它做平移原点会把股骨旋转误差
    乘上约 400 mm 力臂算进"膝平移"。改用膝关节处的股骨固连点。
    """
    fem, tib = (traj.body(b) for b in KNEE_SEGMENTS[side])
    d = traj.body_origin[:, tib] - traj.body_origin[:, fem]
    return np.einsum("tji,tj->ti", traj.body_R[:, fem], d).mean(axis=0)


def _true_knee(traj: Trajectory, side: str) -> np.ndarray:
    fem, tib = (traj.body(b) for b in KNEE_SEGMENTS[side])
    off = knee_offset(traj, side)
    return np.array([grood_suntay(traj.body_R[t, fem],
                                  traj.body_origin[t, fem] + traj.body_R[t, fem] @ off,
                                  traj.body_R[t, tib], traj.body_origin[t, tib], side)
                     for t in range(traj.T)])


def camera_sigmas(cams: list[Camera], sigma_px) -> np.ndarray:
    """把全局 σ_px 展开成每台相机的 σ（含 spec.sigma_rel 相对权重）。"""
    if np.isscalar(sigma_px):
        base = np.full(len(cams), float(sigma_px))
    else:
        base = np.asarray(sigma_px, float)
        if base.ndim == 0:
            base = np.full(len(cams), float(base))
        if len(base) != len(cams):
            raise ValueError("sigma_px array length must equal number of cameras")
    rel = np.array([c.spec.sigma_rel if c.spec is not None else 1.0 for c in cams])
    return base * rel


def _batched_triangulate(Ps: np.ndarray, uv: np.ndarray,
                         sigmas: np.ndarray | None = None) -> np.ndarray:
    """Ps (V,3,4)，uv (S,V,2) -> (S,3)：加权 DLT + 3 次高斯-牛顿细化。"""
    S, V = uv.shape[:2]
    if sigmas is None:
        sigmas = np.ones(V)
    sigmas = np.asarray(sigmas, float)
    wv = np.where(sigmas > 0, 1.0 / np.maximum(sigmas, 1e-12), 1e6)
    A = np.empty((S, 2 * V, 4))
    A[:, 0::2] = uv[:, :, 0:1] * Ps[None, :, 2] - Ps[None, :, 0]
    A[:, 1::2] = uv[:, :, 1:2] * Ps[None, :, 2] - Ps[None, :, 1]
    A /= np.linalg.norm(A[:, :, :3], axis=2, keepdims=True)
    A *= np.repeat(wv, 2)[None, :, None]
    X = np.linalg.svd(A)[2][:, -1]
    X = X[:, :3] / X[:, 3:4]
    for _ in range(3):
        p = np.einsum("vij,sj->svi", Ps, np.concatenate([X, np.ones((S, 1))], 1))
        w = p[:, :, 2:3]
        r = ((p[:, :, :2] / w - uv) * wv[None, :, None]).reshape(S, -1)
        J = (Ps[None, :, :2, :3] * w[..., None]
             - p[:, :, :2, None] * Ps[None, :, 2:3, :3]) / w[..., None] ** 2
        J = (J * wv[None, :, None, None]).reshape(S, -1, 3)
        JtJ = np.einsum("sni,snj->sij", J, J)
        Jtr = np.einsum("sni,sn->si", J, r)
        X = X - np.linalg.solve(JtJ, Jtr[..., None])[..., 0]
    return X


def _batched_rigid_fit(P: np.ndarray, Q: np.ndarray):
    """P (S,n,3) 测量点，Q (n,3) 体段系模板 -> R (S,3,3), t (S,3)。"""
    cp, cq = P.mean(1), Q.mean(0)
    H = np.einsum("ni,snj->sij", Q - cq, P - cp[:, None])
    U, _, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ np.transpose(U, (0, 2, 1))))
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1
    D[:, 2, 2] = d
    R = V @ D @ np.transpose(U, (0, 2, 1))
    return R, cp - np.einsum("sij,j->si", R, cq)


@dataclass
class KneeError:
    rmse: np.ndarray        # (6,) 各自由度 RMSE（成功帧上）
    p95: np.ndarray         # (6,) 各自由度绝对误差 95 分位
    fail_frac: float        # 无法拟合（任一体段 <3 个可重建标记点）的帧占比


def knee_monte_carlo(traj: Trajectory, cams: list[Camera], vis: np.ndarray, side: str,
                     sigma_px, n_mc: int = 200, seed: int = 42) -> KneeError:
    """端到端：真值投影 -> 像素高斯噪声 -> 多视图三角测量 -> 刚体拟合 -> G&S。

    vis (T, M, V) 为检出矩阵；每个标记点只用检出它的相机。
    sigma_px 可为标量或每台相机一个值；spec.sigma_rel 会乘上去。
    """
    rng = np.random.default_rng(seed)
    sigmas = camera_sigmas(cams, sigma_px)
    Ps_all = np.stack([c.P for c in cams])
    truth = _true_knee(traj, side)
    off = knee_offset(traj, side)
    seg_idx = segment_markers(traj, side)
    errs, fails = [], 0
    for t in range(traj.T):
        poses = []
        for idx in seg_idx:
            usable = [m for m in idx if vis[t, m].sum() >= 2]
            if len(usable) < 3:
                poses = None
                break
            P = np.empty((n_mc, len(usable), 3))
            for j, m in enumerate(usable):
                vv = np.flatnonzero(vis[t, m])
                X = traj.xyz[t, m]
                uv = np.stack([cams[v].project(X[None])[0][0] for v in vv])
                noisy = uv[None] + sigmas[vv][None, :, None] * rng.standard_normal(
                    (n_mc, len(vv), 2))
                P[:, j] = _batched_triangulate(Ps_all[vv], noisy, sigmas[vv])
            poses.append(_batched_rigid_fit(P, traj.marker_local[usable]))
        if poses is None:
            fails += 1
            continue
        (Rf, tf), (Rt, tt) = poses
        tf = tf + np.einsum("sij,j->si", Rf, off)
        ks = np.array([grood_suntay(Rf[k], tf[k], Rt[k], tt[k], side) for k in range(n_mc)])
        errs.append(ks - truth[t])
    if not errs:
        return KneeError(np.full(6, np.nan), np.full(6, np.nan), 1.0)
    E = np.concatenate(errs)
    return KneeError(np.sqrt(np.mean(E**2, axis=0)),
                     np.percentile(np.abs(E), 95, axis=0), fails / traj.T)
