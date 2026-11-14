"""单台相机对反光球标记点的可检出性（移植自旧仓库 acc.detectMarkers）。

判定顺序固定，每个未检出样本只记一个原因：
  1 视锥外（相机背后或出画面）
  2 成像直径小于最小像素
  3 背向（法向与视线夹角超过掠射角）
  4 被其他体段胶囊遮挡
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .trajectory import Occluders, Trajectory

OUT_OF_VIEW, TOO_SMALL, FACING_AWAY, OCCLUDED = 1, 2, 3, 4
REASON_NAMES = {OUT_OF_VIEW: "视锥外", TOO_SMALL: "成像过小",
                FACING_AWAY: "背向", OCCLUDED: "遮挡"}


@dataclass(frozen=True)
class DetectionModel:
    ball_diameter_mm: float = 14.0
    min_blob_px: float = 3.0
    grazing_deg: float = 70.0
    z_near_mm: float = 1.0


def segment_hits_capsules(P, C, A, B, r, allow):
    """线段 P->C 是否穿过任一允许的胶囊（Ericson 线段-线段最近点）。

    P (N,3)，C (3,)，A/B (N,K,3)，r (K,)，allow (N,K)。返回 (hit (N,), which (N,))。
    """
    N, K = allow.shape
    if N == 0:
        return np.zeros(0, bool), np.zeros(0, int)
    d1 = (C[None, :] - P)[:, None, :]          # (N,1,3)
    d2 = B - A                                 # (N,K,3)
    r0 = P[:, None, :] - A                     # (N,K,3)
    a = np.sum(d1**2, axis=2)                  # (N,1)
    e = np.sum(d2**2, axis=2)                  # (N,K)
    b = np.sum(d1 * d2, axis=2)
    c = np.sum(d1 * r0, axis=2)
    f = np.sum(d2 * r0, axis=2)
    aK = np.broadcast_to(a, (N, K))

    denom = aK * e - b**2
    s = np.zeros((N, K))
    nz = denom > 1e-12 * max(np.max(aK * e), 1.0)
    s[nz] = (b[nz] * f[nz] - c[nz] * e[nz]) / denom[nz]
    s = np.clip(s, 0, 1)
    tt = (b * s + f) / e
    lo, hi = tt < 0, tt > 1
    tt = np.clip(tt, 0, 1)
    s = np.where(lo, np.clip(-c / aK, 0, 1), s)
    s = np.where(hi, np.clip((b - c) / aK, 0, 1), s)

    p1 = P[:, None, :] + s[:, :, None] * d1
    p2 = A + tt[:, :, None] * d2
    dist2 = np.sum((p1 - p2) ** 2, axis=2)
    blocked = allow & (dist2 <= r[None, :] ** 2)
    hit = blocked.any(axis=1)
    which = np.where(hit, blocked.argmax(axis=1), -1)
    return hit, which


def detect(cam: Camera, points: np.ndarray, occ: Occluders | None,
           model: DetectionModel = DetectionModel()) -> np.ndarray:
    """返回 (N,) uint8 原因码，0 表示检出。points 为展平样本。"""
    uv, z = cam.project(points)
    with np.errstate(divide="ignore", invalid="ignore"):
        in_view = ((z > model.z_near_mm) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width)
                   & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height))
        blob = model.ball_diameter_mm * cam.f / z
    reason = np.zeros(len(points), np.uint8)
    reason[~in_view] = OUT_OF_VIEW
    reason[in_view & (blob < model.min_blob_px)] = TOO_SMALL
    if occ is None:
        return reason

    C = cam.center
    view = C[None, :] - points
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    facing = np.sum(occ.normal * view, axis=1) >= np.cos(np.deg2rad(model.grazing_deg))
    reason[(reason == 0) & ~facing] = FACING_AWAY

    idx = np.flatnonzero(reason == 0)
    if idx.size:
        hit, _ = segment_hits_capsules(occ.ray_origin[idx], C, occ.A[idx], occ.B[idx],
                                       occ.radii, occ.allow[idx])
        reason[idx[hit]] = OCCLUDED
    return reason


def detect_traj(cam: Camera, traj: Trajectory,
                model: DetectionModel = DetectionModel()) -> np.ndarray:
    """返回 (T, M) 原因码。"""
    return detect(cam, traj.points(), traj.occ, model).reshape(traj.M, traj.T).T
