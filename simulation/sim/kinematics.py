"""刚体拟合与膝关节 Grood & Suntay 六自由度分解。

体段坐标系直接用 OpenSim 体段系（Gait2392：X 向前、Y 向上、Z 向右）。
标记点刚连于体段，故用 marker_local 作模板的无噪拟合精确等于体段真值位姿。
"""

from __future__ import annotations

import numpy as np

DOF_NAMES = ["屈伸°", "内外翻°", "内外旋°", "AP mm", "ML mm", "牵张 mm"]


def rigid_fit(P: np.ndarray, Q: np.ndarray):
    """求 R, t 使 R @ Q + t ≈ P。P、Q 为 (n,3)，n >= 3。"""
    cp, cq = P.mean(0), Q.mean(0)
    H = (Q - cq).T @ (P - cp)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cp - R @ cq


def grood_suntay(Rf, tf, Rt, tt, side: str = "R") -> np.ndarray:
    """股骨 (Rf, tf) 与胫骨 (Rt, tt) 世界位姿 -> [屈伸, 内外翻, 内外旋, AP, ML, 牵张]。

    e1 = 股骨内外侧轴（Z），e3 = 胫骨长轴（Y），e2 = e3 × e1 浮动轴。
    左膝把 Z 轴取反，使左右两侧符号含义一致（内外侧轴都指向外侧）。
    平移为胫骨原点相对股骨参考点（调用方给出，通常取膝关节处的股骨固连点），
    投影到 e2(AP)、e1(ML)、e3(牵张)。
    """
    s = 1.0 if side == "R" else -1.0
    Rrel = Rf.T @ Rt
    trel = Rf.T @ (tt - tf)
    e1 = np.array([0.0, 0.0, s])
    e3 = Rrel[:, 1]
    zt = Rrel[:, 2] * s
    e2 = np.cross(e3, e1)
    e2 /= np.linalg.norm(e2)
    flex = np.arctan2(e3[0], e3[1])
    abd = np.arccos(np.clip(e1 @ e3, -1, 1)) - np.pi / 2
    w = e1 - (e1 @ e3) * e3
    w /= np.linalg.norm(w)
    v = zt - (zt @ e3) * e3
    v /= np.linalg.norm(v)
    ir = np.arctan2(np.cross(w, v) @ e3, w @ v)
    return np.array([np.rad2deg(flex), np.rad2deg(abd), np.rad2deg(ir),
                     trel @ e2, trel @ e1, trel @ e3])
