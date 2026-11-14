"""针孔相机模型、异构相机规格与单台相机的摆放。

约定：X_cam = R @ X_world + t；u = f*x/z + cx，v = f*y/z + cy。
图像 x 轴保持水平（无横滚）。海康传感器可横放或竖放：竖放通过交换
有效像面宽高实现，镜头焦距与像元尺寸不变。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WORLD_UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class CameraSpec:
    """一台相机的物理规格。

    sensor_w_mm / sensor_h_mm 是传感器在横放状态下的物理宽高；
    orientation='portrait' 时有效像面宽高互换。focal_mm 为镜头标称焦距，
    sigma_rel 是相对像素噪声权重（用于异构相机，1.0 表示基准）。
    """

    name: str
    model: str                 # "Basler" / "Hik"
    sensor_w_mm: float
    sensor_h_mm: float
    image_w: int
    image_h: int
    focal_mm: float
    sigma_rel: float = 1.0
    orientation: str = "landscape"   # landscape / portrait

    @property
    def pixel_size_mm(self) -> float:
        return self.sensor_w_mm / self.image_w

    @property
    def f_px(self) -> float:
        return self.focal_mm / self.pixel_size_mm

    @property
    def image_size(self) -> tuple[int, int]:
        if self.orientation == "portrait":
            return self.image_h, self.image_w
        return self.image_w, self.image_h

    @property
    def fov_deg(self) -> tuple[float, float]:
        w, h = self.image_size
        f = self.f_px
        return (2.0 * np.degrees(np.arctan(w / (2.0 * f))),
                2.0 * np.degrees(np.arctan(h / (2.0 * f))))

    def K(self) -> np.ndarray:
        w, h = self.image_size
        f = self.f_px
        return np.array([[f, 0.0, w / 2.0],
                         [0.0, f, h / 2.0],
                         [0.0, 0.0, 1.0]])

    def with_orientation(self, orientation: str) -> "CameraSpec":
        if orientation not in ("landscape", "portrait"):
            raise ValueError("orientation must be 'landscape' or 'portrait'")
        return CameraSpec(self.name, self.model, self.sensor_w_mm, self.sensor_h_mm,
                          self.image_w, self.image_h, self.focal_mm,
                          self.sigma_rel, orientation)

    def with_lens(self, focal_mm: float) -> "CameraSpec":
        return CameraSpec(self.name, self.model, self.sensor_w_mm, self.sensor_h_mm,
                          self.image_w, self.image_h, focal_mm,
                          self.sigma_rel, self.orientation)


# 本轮使用的两台相机。海康默认竖放由摆放时指定。
BASLER = CameraSpec("Basler acA2040-90umNIR", "Basler",
                    sensor_w_mm=11.26, sensor_h_mm=11.26,
                    image_w=2048, image_h=2048,
                    focal_mm=12.0)
HIK = CameraSpec("Hikvision MV-CU020-90UM", "Hik",
                 sensor_w_mm=8.19, sensor_h_mm=4.80,
                 image_w=2048, image_h=1200,
                 focal_mm=12.0)


@dataclass(frozen=True)
class Camera:
    K: np.ndarray        # (3, 3)
    R: np.ndarray        # (3, 3) 世界 -> 相机
    t: np.ndarray        # (3,)
    width: int
    height: int
    name: str = "cam"
    spec: CameraSpec | None = None

    @property
    def f(self) -> float:
        return float(self.K[0, 0])

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def P(self) -> np.ndarray:
        """(3, 4) 投影矩阵。"""
        return self.K @ np.hstack([self.R, self.t[:, None]])

    def to_cam(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R.T + self.t

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (uv (N,2), z (N,))。"""
        Xc = self.to_cam(X)
        z = Xc[:, 2]
        uv = Xc[:, :2] / z[:, None] * self.f + self.K[:2, 2]
        return uv, z

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """(N, 2, 3) 投影对世界坐标的雅可比。"""
        Xc = self.to_cam(X)
        z = Xc[:, 2:3]
        du = self.f * (self.R[0] * z - Xc[:, 0:1] * self.R[2]) / z**2
        dv = self.f * (self.R[1] * z - Xc[:, 1:2] * self.R[2]) / z**2
        return np.stack([du, dv], axis=1)


def intrinsics(hfov_deg: float, image_size=(2048, 2048)) -> np.ndarray:
    w, h = image_size
    f = (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _look_at_rotation(position, target) -> np.ndarray:
    c = np.asarray(position, float)
    z = np.asarray(target, float) - c
    z /= np.linalg.norm(z)
    if abs(z @ WORLD_UP) > 0.999:
        raise ValueError("optical axis cannot be vertical")
    x = np.cross(z, WORLD_UP)          # 图像右方向（水平）
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                 # 图像下方向（v 增大）
    return np.vstack([x, y, z])


def look_at(position, target, hfov_deg: float, image_size=(2048, 2048),
            name: str = "cam") -> Camera:
    """相机放在 position，光轴对准 target，图像 x 轴水平（兼容旧接口）。"""
    c = np.asarray(position, float)
    R = _look_at_rotation(c, target)
    return Camera(intrinsics(hfov_deg, image_size), R, -R @ c,
                  image_size[0], image_size[1], name)


def camera_from_spec(spec: CameraSpec, position, target,
                     name: str | None = None) -> Camera:
    """按物理规格构造相机（支持海康横/竖放）。"""
    c = np.asarray(position, float)
    R = _look_at_rotation(c, target)
    w, h = spec.image_size
    return Camera(spec.K(), R, -R @ c, w, h, name or spec.name, spec)


@dataclass(frozen=True)
class Placement:
    """单台相机的摆放参数（相对目标中心）。

    azimuth_deg：0 表示相机在目标 +X 侧（正对行走方向迎面拍），
    90 表示在 -Z 侧（受试者左侧），-90 表示 +Z 侧（右侧），±180 表示背后。
    distance_mm：相机到目标中心的水平距离；height_mm：相机绝对高度（世界 Y）。
    orientation / camera_type 供异构搜索使用。
    """

    azimuth_deg: float
    distance_mm: float
    height_mm: float
    orientation: str = "landscape"
    camera_type: str = "Basler"

    def position(self, target) -> np.ndarray:
        a = np.deg2rad(self.azimuth_deg)
        tx, _, tz = np.asarray(target, float)
        return np.array([tx + self.distance_mm * np.cos(a), self.height_mm,
                         tz - self.distance_mm * np.sin(a)])

    def camera(self, target, hfov_deg: float, image_size=(2048, 2048),
               name: str | None = None) -> Camera:
        label = name or f"az{self.azimuth_deg:+.0f}_d{self.distance_mm:.0f}_h{self.height_mm:.0f}"
        return look_at(self.position(target), target, hfov_deg, image_size, label)

    def camera_spec(self, target, spec: CameraSpec,
                    name: str | None = None) -> Camera:
        label = name or f"{spec.model}_az{self.azimuth_deg:+.0f}_d{self.distance_mm:.0f}_h{self.height_mm:.0f}"
        return camera_from_spec(spec.with_orientation(self.orientation),
                                self.position(target), target, label)
