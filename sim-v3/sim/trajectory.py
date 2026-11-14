"""步态轨迹与遮挡几何的加载。

数据来自旧仓库 matlab-simulation/results/trajectory_cache.mat（OpenSim Gait2392
正常步态，1 个完整步态周期，51 帧，双腿各 12 个标记点）。

世界坐标沿用 Gait2392：+X 为行走方向，+Y 向上，+Z 为右侧。
展平顺序沿用旧代码：样本 n = t + m*T（0 起，帧变化最快）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.io as sio

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "trajectory_cache.mat"


@dataclass(frozen=True)
class Occluders:
    """与相机无关的体段胶囊遮挡几何，按展平样本存储。"""

    A: np.ndarray            # (N, K, 3) 胶囊端点
    B: np.ndarray            # (N, K, 3)
    radii: np.ndarray        # (K,)
    allow: np.ndarray        # (N, K) bool，False = 标记点所在体段自身的胶囊
    normal: np.ndarray       # (N, 3) 标记点外法向
    ray_origin: np.ndarray   # (N, 3) 遮挡射线起点（推出自身胶囊表面外）
    capsule_names: list[str]


@dataclass(frozen=True)
class Trajectory:
    time: np.ndarray             # (T,)
    names: list[str]             # (M,)
    xyz: np.ndarray              # (T, M, 3) mm
    body_names: list[str]        # (B,)
    body_origin: np.ndarray      # (T, B, 3) mm
    body_R: np.ndarray           # (T, B, 3, 3) 体段系 -> 世界
    marker_body: np.ndarray      # (M,) 0 起体段索引
    marker_local: np.ndarray     # (M, 3) 标记点在体段系下的坐标
    forward_offset: np.ndarray   # (T,) 叠加的 +X 前进位移
    occ: Occluders

    @property
    def T(self) -> int:
        return self.xyz.shape[0]

    @property
    def M(self) -> int:
        return self.xyz.shape[1]

    @property
    def travel_mm(self) -> float:
        return float(self.forward_offset[-1] - self.forward_offset[0])

    def points(self) -> np.ndarray:
        """(N, 3) 展平样本，n = t + m*T。"""
        return self.xyz.transpose(1, 0, 2).reshape(-1, 3)

    def index(self, names) -> np.ndarray:
        return np.array([self.names.index(n) for n in names])

    def body(self, name: str) -> int:
        return self.body_names.index(name)

    def markers_on(self, body_name: str) -> np.ndarray:
        return np.flatnonzero(self.marker_body == self.body(body_name))


def load(path: str | Path = DEFAULT_CACHE, travel_mm: float | None = None) -> Trajectory:
    """读取缓存；travel_mm 不为 None 时把整周期的前进距离改成该值。"""
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    t, o = m["trajectory"], m["occluderTrack"]
    occ = Occluders(
        A=np.asarray(o.A_world, float),
        B=np.asarray(o.B_world, float),
        radii=np.asarray(o.radii, float),
        allow=np.asarray(o.allow, bool),
        normal=np.asarray(o.normal_world, float),
        ray_origin=np.asarray(o.ray_origin, float),
        capsule_names=[str(s) for s in o.capsule_names],
    )
    traj = Trajectory(
        time=np.asarray(t.time, float),
        names=[str(s) for s in t.names],
        xyz=np.asarray(t.xyz_mm, float),
        body_names=[str(s) for s in t.body_names],
        body_origin=np.asarray(t.body_origin_mm, float),
        body_R=np.asarray(t.body_R, float),
        marker_body=np.asarray(t.marker_body, int) - 1,
        marker_local=np.asarray(t.marker_local_mm, float),
        forward_offset=np.asarray(t.forward_offset_mm, float),
        occ=occ,
    )
    if travel_mm is not None:
        traj = set_travel(traj, travel_mm)
    return traj


def set_travel(traj: Trajectory, travel_mm: float) -> Trajectory:
    """把线性前进位移改为 travel_mm（以 X=0 为中点），标记点、体段和遮挡体同步平移。"""
    if travel_mm < 0:
        raise ValueError("travel_mm must be >= 0")
    progress = (traj.time - traj.time[0]) / (traj.time[-1] - traj.time[0])
    new_offset = -travel_mm / 2 + travel_mm * progress
    dx = new_offset - traj.forward_offset            # (T,)

    xyz = traj.xyz.copy()
    xyz[:, :, 0] += dx[:, None]
    origin = traj.body_origin.copy()
    origin[:, :, 0] += dx[:, None]

    dx_n = np.tile(dx, traj.M)                       # 展平样本的帧 = n % T
    occ = traj.occ
    A, B, ro = occ.A.copy(), occ.B.copy(), occ.ray_origin.copy()
    A[:, :, 0] += dx_n[:, None]
    B[:, :, 0] += dx_n[:, None]
    ro[:, 0] += dx_n
    return replace(traj, xyz=xyz, body_origin=origin, forward_offset=new_offset,
                   occ=replace(occ, A=A, B=B, ray_origin=ro))
