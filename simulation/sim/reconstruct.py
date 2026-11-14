"""多视图三角测量与三维精度场。"""

from __future__ import annotations

import numpy as np

from .camera import Camera


def triangulate(Ps: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """线性 DLT 多视图三角测量。

    Ps (V,3,4) 投影矩阵，uv (V,2) 像素坐标，V >= 2。返回 (3,)。
    """
    A = np.empty((2 * len(Ps), 4))
    A[0::2] = uv[:, 0:1] * Ps[:, 2] - Ps[:, 0]
    A[1::2] = uv[:, 1:2] * Ps[:, 2] - Ps[:, 1]
    # 行归一化，避免各相机尺度不同导致加权失衡
    A /= np.linalg.norm(A[:, :3], axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def refine(Ps: np.ndarray, uv: np.ndarray, X: np.ndarray, iters: int = 3) -> np.ndarray:
    """高斯-牛顿最小化重投影误差（DLT 结果作为初值）。"""
    X = X.copy()
    for _ in range(iters):
        Xh = np.append(X, 1.0)
        p = Ps @ Xh                              # (V,3)
        proj = p[:, :2] / p[:, 2:3]
        r = (proj - uv).ravel()
        J = (Ps[:, :2, :3] * p[:, 2:3, None] - p[:, :2, None] * Ps[:, 2:3, :3]) / p[:, 2:3, None] ** 2
        J = J.reshape(-1, 3)
        dX, *_ = np.linalg.lstsq(J, -r, rcond=None)
        X += dX
        if np.linalg.norm(dX) < 1e-6:
            break
    return X


def sigma_worst(cams: list[Camera], X: np.ndarray, seen: np.ndarray, sigma_px: float = 1.0):
    """各样本最差方向 1σ 三维误差（mm）：Fisher 信息 Σ JᵀJ/σ² 的最小特征值。

    X (N,3)，seen (N,V) bool。少于 2 台看到的样本返回 inf。
    """
    N = len(X)
    Minfo = np.zeros((N, 3, 3))
    for v, cam in enumerate(cams):
        idx = np.flatnonzero(seen[:, v])
        if idx.size:
            J = cam.jacobian(X[idx])
            Minfo[idx] += np.einsum("nij,nik->njk", J, J)
    out = np.full(N, np.inf)
    ok = seen.sum(axis=1) >= 2
    if ok.any():
        lam = np.linalg.eigvalsh(Minfo[ok])[:, 0]
        good = lam > 1e-12
        vals = np.full(ok.sum(), np.inf)
        vals[good] = sigma_px / np.sqrt(lam[good])
        out[ok] = vals
    return out


def triangulate_weighted(Ps: np.ndarray, uv: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """按每台相机 σ_px 加权的 DLT 三角测量。"""
    sigmas = np.asarray(sigmas, float)
    A = np.empty((2 * len(Ps), 4))
    A[0::2] = uv[:, 0:1] * Ps[:, 2] - Ps[:, 0]
    A[1::2] = uv[:, 1:2] * Ps[:, 2] - Ps[:, 1]
    A /= np.linalg.norm(A[:, :3], axis=1, keepdims=True)
    A *= np.repeat(1.0 / sigmas, 2)[:, None]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def refine_weighted(Ps: np.ndarray, uv: np.ndarray, X: np.ndarray,
                    sigmas: np.ndarray, iters: int = 3) -> np.ndarray:
    """按每台相机 σ_px 加权的高斯-牛顿重投影细化。"""
    sigmas = np.asarray(sigmas, float)
    w = np.repeat(1.0 / sigmas, 2)
    X = X.copy()
    for _ in range(iters):
        Xh = np.append(X, 1.0)
        p = Ps @ Xh
        proj = p[:, :2] / p[:, 2:3]
        r = ((proj - uv) * w[:, None]).ravel()
        J = (Ps[:, :2, :3] * p[:, 2:3, None]
             - p[:, :2, None] * Ps[:, 2:3, :3]) / p[:, 2:3, None] ** 2
        J = (J * w[:, None, None]).reshape(-1, 3)
        dX, *_ = np.linalg.lstsq(J, -r, rcond=None)
        X += dX
        if np.linalg.norm(dX) < 1e-6:
            break
    return X


def sigma_worst_weighted(cams: list[Camera], X: np.ndarray, seen: np.ndarray,
                         sigmas: np.ndarray):
    """每台相机不同 σ_px 的 Fisher 信息三维精度场。

    返回 (sigma_worst, sigma_z)。M = Σ JᵀJ/σ²，σ_worst = 1/√λ_min(M)。
    """
    sigmas = np.asarray(sigmas, float)
    N = len(X)
    Minfo = np.zeros((N, 3, 3))
    for v, cam in enumerate(cams):
        idx = np.flatnonzero(seen[:, v])
        if idx.size:
            J = cam.jacobian(X[idx])
            Minfo[idx] += np.einsum("nij,nik->njk", J, J) / sigmas[v] ** 2
    out = np.full(N, np.inf)
    out_z = np.full(N, np.inf)
    ok = seen.sum(axis=1) >= 2
    if ok.any():
        lam = np.linalg.eigvalsh(Minfo[ok])[:, 0]
        good = lam > 1e-12
        vals = np.full(ok.sum(), np.inf)
        vals[good] = 1.0 / np.sqrt(lam[good])
        out[ok] = vals
        for n in np.flatnonzero(ok):
            M = Minfo[n]
            if np.linalg.eigvalsh(M)[0] > 1e-12:
                out_z[n] = np.sqrt(max(0.0, np.linalg.inv(M)[2, 2]))
    return out, out_z


def ray_angle_ok(cams: list[Camera], X: np.ndarray, seen: np.ndarray, min_deg: float) -> np.ndarray:
    """样本是否至少有一对检出相机的视线夹角 >= min_deg（排除几乎共线的两视图）。"""
    N, V = seen.shape
    dirs = np.stack([X - c.center for c in cams], axis=1)       # (N,V,3)
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    cos_max = np.cos(np.deg2rad(min_deg))
    ok = np.zeros(N, bool)
    for i in range(V):
        for j in range(i + 1, V):
            both = seen[:, i] & seen[:, j]
            cosij = np.sum(dirs[:, i] * dirs[:, j], axis=1)
            ok |= both & (cosij <= cos_max)
    return ok
