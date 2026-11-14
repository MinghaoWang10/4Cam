import numpy as np
import pytest

from sim import trajectory
from sim.camera import BASLER, HIK, Placement, camera_from_spec
from sim.layout import (CameraPlacement, evaluate_layout, fov_fits, make_layout,
                        search)
from sim.pipeline import camera_sigmas
from sim.walkway import Walkway, default_walkway


def test_camera_spec_fov():
    assert BASLER.f_px == pytest.approx(12.0 / (11.26 / 2048), rel=1e-9)
    assert BASLER.fov_deg[0] == pytest.approx(50.27, abs=0.05)
    assert HIK.fov_deg[0] == pytest.approx(37.68, abs=0.05)
    assert HIK.fov_deg[1] == pytest.approx(22.61, abs=0.05)
    hp = HIK.with_orientation("portrait")
    assert hp.fov_deg[0] == pytest.approx(HIK.fov_deg[1], abs=1e-9)
    assert hp.fov_deg[1] == pytest.approx(HIK.fov_deg[0], abs=1e-9)


def test_camera_from_spec_projects_target_to_principal_point():
    cam = camera_from_spec(BASLER, [3000, 800, 0], [0, 550, 0])
    uv, z = cam.project(np.array([[0, 550, 0.0]]))
    assert np.allclose(uv, [[cam.width / 2, cam.height / 2]], atol=1e-9)
    assert z[0] > 0


def test_walkway_geometry_and_camera_position():
    w = default_walkway()
    assert w.length_mm == 4000 and w.width_mm == 1200
    assert w.analysis_x_min == -1100 and w.analysis_x_max == 1100
    assert w.x_min == -2000 and w.x_max == 2000
    assert w.z_min == -600 and w.z_max == 600
    cam = w.camera(BASLER, 0, 3000, 800)
    # 方位角 0 = 相机在 +X 侧，朝 -X
    assert cam.center[0] == pytest.approx(3000, abs=1e-9)
    assert cam.center[1] == pytest.approx(800, abs=1e-9)
    assert cam.center[2] == pytest.approx(0, abs=1e-9)


def test_camera_sigmas_expands_relative_weights():
    from dataclasses import replace
    hik_noisy = replace(HIK, sigma_rel=1.5)
    cams = [camera_from_spec(BASLER, [3000, 800, 0], [0, 550, 0]),
            camera_from_spec(hik_noisy, [0, 800, 3000], [0, 550, 0])]
    s = camera_sigmas(cams, 0.2)
    assert np.allclose(s, [0.2, 0.3])


def test_fov_filter_rejects_narrow_portrait_side_camera():
    traj = trajectory.load(travel_mm=1300)
    w = default_walkway()
    idx = np.array([i for i, n in enumerate(traj.names) if n[0] in "RL"])
    pts = traj.xyz[:, idx, :].reshape(-1, 3)
    side_hik = w.camera(HIK.with_orientation("portrait"), -90, 3000, 600)
    assert not fov_fits(side_hik, pts)
    side_basler = w.camera(BASLER, -90, 3000, 600)
    assert fov_fits(side_basler, pts)


def test_make_layout_has_two_basler_two_hik():
    w = default_walkway()
    ps = make_layout("十字_B侧向_H前后", w, 3000, 800, 0)
    assert len(ps) == 4
    assert sum(p.spec.model == "Basler" for p in ps) == 2
    assert sum(p.spec.model == "Hik" for p in ps) == 2
    assert sum(p.spec.orientation == "portrait" for p in ps) == 2


def test_evaluate_layout_returns_valid_metrics():
    traj = trajectory.load(travel_mm=1300)
    w = default_walkway()
    ps = [
        CameraPlacement(BASLER, -45, 3000, 300, 0, 550, "C1"),
        CameraPlacement(BASLER, 45, 3000, 600, 0, 550, "C2"),
        CameraPlacement(HIK, -135, 3500, 300, 0, 550, "C3"),
        CameraPlacement(HIK, 135, 4000, 300, 0, 550, "C4"),
    ]
    cams = [p.camera(w) for p in ps]
    sc = evaluate_layout(traj, cams, "RL", sigma_px=0.3)
    assert sc.fail_frac == pytest.approx(0.0)
    assert np.isfinite(sc.rot_deg) and np.isfinite(sc.trans_mm)
    assert 0 <= sc.coverage_min <= sc.coverage_mean <= 1
    assert sc.rot_deg < 0.5 and sc.trans_mm < 0.6
