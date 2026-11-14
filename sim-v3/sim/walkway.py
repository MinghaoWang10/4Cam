"""步道几何与相机相对布置。

世界坐标：+X 行走方向，+Y 向上，+Z 受试者右侧。
分析窗口中心为原点；步道沿 X 方向延伸，宽度沿 Z 方向。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraSpec, Placement, camera_from_spec


@dataclass(frozen=True)
class Walkway:
    length_mm: float = 4000.0          # 总长（进入 + 分析 + 退出）
    width_mm: float = 1200.0           # 总宽
    analysis_length_mm: float = 2200.0 # 中央分析窗口
    entry_mm: float = 900.0            # 分析区前的进入段
    exit_mm: float = 900.0             # 分析区后的退出段
    center_x_mm: float = 0.0
    center_z_mm: float = 0.0

    def __post_init__(self):
        if self.length_mm <= 0 or self.width_mm <= 0:
            raise ValueError("walkway length/width must be positive")
        if self.analysis_length_mm <= 0:
            raise ValueError("analysis length must be positive")
        if self.analysis_length_mm + self.entry_mm + self.exit_mm > self.length_mm + 1e-9:
            raise ValueError("analysis + entry + exit exceeds walkway length")

    @property
    def analysis_x_min(self) -> float:
        return self.center_x_mm - self.analysis_length_mm / 2.0

    @property
    def analysis_x_max(self) -> float:
        return self.center_x_mm + self.analysis_length_mm / 2.0

    @property
    def x_min(self) -> float:
        return self.center_x_mm - self.length_mm / 2.0

    @property
    def x_max(self) -> float:
        return self.center_x_mm + self.length_mm / 2.0

    @property
    def z_min(self) -> float:
        return self.center_z_mm - self.width_mm / 2.0

    @property
    def z_max(self) -> float:
        return self.center_z_mm + self.width_mm / 2.0

    @property
    def subject_start_x(self) -> float:
        return self.x_min

    @property
    def subject_end_x(self) -> float:
        return self.x_max

    def contains(self, x_mm, z_mm) -> bool:
        return (self.x_min <= x_mm <= self.x_max
                and self.z_min <= z_mm <= self.z_max)

    def aim_point(self, aim_x_mm: float = 0.0, aim_y_mm: float = 550.0) -> np.ndarray:
        """瞄准点：X 可沿步道前后移动，Y 取膝高，Z 取步道中心。"""
        return np.array([self.center_x_mm + aim_x_mm, aim_y_mm, self.center_z_mm])

    def camera(self, spec: CameraSpec, azimuth_deg: float, distance_mm: float,
               height_mm: float, aim_x_mm: float = 0.0,
               aim_y_mm: float = 550.0, name: str | None = None):
        """按步道坐标构造相机：方位角/距离相对瞄准点，高度为绝对世界 Y。"""
        aim = self.aim_point(aim_x_mm, aim_y_mm)
        pl = Placement(azimuth_deg, distance_mm, height_mm,
                       spec.orientation, spec.model)
        return pl.camera_spec(aim, spec, name)

    def summary(self) -> dict:
        return {
            "length_mm": self.length_mm,
            "width_mm": self.width_mm,
            "analysis_length_mm": self.analysis_length_mm,
            "entry_mm": self.entry_mm,
            "exit_mm": self.exit_mm,
            "analysis_x_mm": [self.analysis_x_min, self.analysis_x_max],
            "walkway_x_mm": [self.x_min, self.x_max],
            "walkway_z_mm": [self.z_min, self.z_max],
            "subject_start_x_mm": self.subject_start_x,
            "subject_end_x_mm": self.subject_end_x,
        }


def default_walkway() -> Walkway:
    """本轮确认的 4.0 m × 1.2 m 步道。"""
    return Walkway(length_mm=4000.0, width_mm=1200.0,
                   analysis_length_mm=2200.0, entry_mm=900.0, exit_mm=900.0)
