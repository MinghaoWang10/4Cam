import numpy as np
import pytest

from sim import trajectory
from sim.camera import Placement, look_at
from sim.detection import (FACING_AWAY, OCCLUDED, OUT_OF_VIEW, TOO_SMALL, DetectionModel,
                           detect, segment_hits_capsules)
from sim.kinematics import grood_suntay, rigid_fit
from sim.pipeline import _batched_rigid_fit, _batched_triangulate, _true_knee, knee_monte_carlo, views
from sim.reconstruct import refine, sigma_worst, triangulate


@pytest.fixture(scope="module")
def traj():
    return trajectory.load()


# ---------- 轨迹 ----------

def test_markers_are_rigid_on_bodies(traj):
    for t in (0, 25, 50):
        for m in range(traj.M):
            b = traj.marker_body[m]
            p = traj.body_origin[t, b] + traj.body_R[t, b] @ traj.marker_local[m]
            assert np.allclose(p, traj.xyz[t, m], atol=1e-9)


def test_set_travel_moves_everything_together(traj):
    t2 = trajectory.set_travel(traj, 1300.0)
    assert t2.travel_mm == pytest.approx(1300.0)
    d = t2.xyz - traj.xyz
    assert np.allclose(d[:, :, 1:], 0)
    assert np.allclose(d[:, :, 0], (t2.forward_offset - traj.forward_offset)[:, None])
    # 遮挡胶囊随帧同步平移：展平样本 n 的帧为 n % T
    dx = np.tile(t2.forward_offset - traj.forward_offset, traj.M)
    assert np.allclose(t2.occ.A[:, :, 0] - traj.occ.A[:, :, 0], dx[:, None])
    assert np.allclose(t2.occ.ray_origin[:, 0] - traj.occ.ray_origin[:, 0], dx)
    # 体段与标记点仍刚连
    b = t2.marker_body[5]
    assert np.allclose(t2.body_origin[10, b] + t2.body_R[10, b] @ t2.marker_local[5], t2.xyz[10, 5])


def test_one_full_gait_cycle(traj):
    heel = traj.xyz[:, traj.names.index("R.Heel"), 1]
    assert heel[0] < 100 and heel[-1] < 100 and heel.max() > 200


# ---------- 相机 ----------

def test_look_at_projects_target_to_principal_point():
    cam = look_at([3000, 1200, -800], [0, 500, 0], 50)
    uv, z = cam.project(np.array([[0, 500, 0.0]]))
    assert np.allclose(uv, [[1024, 1024]], atol=1e-9) and z[0] > 0
    assert np.allclose(cam.R @ cam.R.T, np.eye(3))
    assert np.allclose(cam.center, [3000, 1200, -800])
    # 图像 x 轴水平、v 向下：目标上方的点 v 更小
    uv2, _ = cam.project(np.array([[0, 700, 0.0]]))
    assert uv2[0, 1] < 1024


def test_placement_azimuth_convention():
    tgt = np.array([0, 500, 0.0])
    assert np.allclose(Placement(0, 3000, 1000).position(tgt), [3000, 1000, 0])
    assert np.allclose(Placement(90, 3000, 1000).position(tgt), [0, 1000, -3000])
    assert np.allclose(Placement(-90, 3000, 1000).position(tgt), [0, 1000, 3000])


def test_jacobian_matches_finite_difference():
    cam = look_at([2500, 900, 1500], [0, 500, 0], 50)
    X = np.array([[100.0, 400.0, -50.0]])
    J = cam.jacobian(X)[0]
    h = 1e-4
    num = np.stack([(cam.project(X + h * e)[0] - cam.project(X - h * e)[0])[0] / (2 * h)
                    for e in np.eye(3)], axis=1)
    assert np.allclose(J, num, rtol=1e-6, atol=1e-8)


# ---------- 检出 ----------

def test_detection_reasons():
    cam = look_at([3000, 500, 0], [0, 500, 0], 50)
    pts = np.array([[0, 500, 0.0],          # 正中
                    [4000, 500, 0.0],       # 相机背后
                    [-200000, 500, 0.0]])   # 极远，成像过小
    r = detect(cam, pts, None, DetectionModel())
    assert list(r) == [0, OUT_OF_VIEW, TOO_SMALL]


def test_capsule_blocks_segment():
    P = np.array([[0, 0, 0.0], [0, 0, 0.0]])
    C = np.array([10, 0, 0.0])
    A = np.array([[[5, -1, 0.0]], [[5, -1, 0.0]]])
    B = np.array([[[5, 1, 0.0]], [[5, 1, 0.0]]])
    hit, _ = segment_hits_capsules(P, C, A, B, np.array([0.5]), np.array([[True], [False]]))
    assert list(hit) == [True, False]
    hit, _ = segment_hits_capsules(P, np.array([0, 10, 0.0]), A, B, np.array([0.5]), np.array([[True], [True]]))
    assert not hit.any()


def test_side_camera_visibility_pattern(traj):
    """相机在受试者右侧（+Z）：右膝外侧可见，右膝内侧背向，
    左膝内侧朝向相机但会在两腿交叠时被右腿挡住。"""
    cam = Placement(-90, 3000, 500).camera([0, 500, 0], 50)
    r = detect(cam, traj.points(), traj.occ).reshape(traj.M, traj.T).T
    col = lambda n: r[:, traj.names.index(n)]
    assert np.mean(col("R.Knee.Lat") == 0) > 0.8
    assert np.mean(col("R.Knee.Med") == FACING_AWAY) > 0.8
    assert 0.05 < np.mean(col("L.Knee.Med") == OCCLUDED) < 0.5


# ---------- 重建与运动学 ----------

def _rig():
    return [Placement(a, 3000, h).camera([0, 500, 0], 50)
            for a, h in [(45, 1200), (135, 1200), (-45, 1200), (-135, 1200)]]


def test_triangulation_exact_without_noise():
    cams = _rig()
    X = np.array([120.0, 430.0, -60.0])
    Ps = np.stack([c.P for c in cams])
    uv = np.stack([c.project(X[None])[0][0] for c in cams])
    assert np.allclose(triangulate(Ps, uv), X, atol=1e-6)
    assert np.allclose(refine(Ps, uv, X + 5), X, atol=1e-6)
    assert np.allclose(_batched_triangulate(Ps, uv[None]), X, atol=1e-6)


def test_triangulation_noise_matches_sigma_worst():
    cams = _rig()[:2]
    X = np.array([[0.0, 500.0, 0.0]])
    sw = sigma_worst(cams, X, np.ones((1, 2), bool), 0.3)[0]
    Ps = np.stack([c.P for c in cams])
    uv = np.stack([c.project(X)[0][0] for c in cams])
    rng = np.random.default_rng(0)
    est = _batched_triangulate(Ps, uv[None] + 0.3 * rng.standard_normal((4000, 2, 2)))
    cov = np.cov((est - X).T)
    assert np.sqrt(np.linalg.eigvalsh(cov)[-1]) == pytest.approx(sw, rel=0.1)


def test_rigid_fit_recovers_pose():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(5, 3)) * 50
    a = 0.4
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    t = np.array([10.0, -20, 5])
    P = Q @ R.T + t
    R2, t2 = rigid_fit(P, Q)
    assert np.allclose(R2, R) and np.allclose(t2, t)
    Rb, tb = _batched_rigid_fit(P[None], Q)
    assert np.allclose(Rb[0], R) and np.allclose(tb[0], t)


def test_grood_suntay_identity_and_pure_flexion():
    I = np.eye(3)
    assert np.allclose(grood_suntay(I, np.zeros(3), I, np.zeros(3)), 0, atol=1e-12)
    a = np.deg2rad(30)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ks = grood_suntay(I, np.zeros(3), Rz, np.zeros(3))
    assert abs(abs(ks[0]) - 30) < 1e-9 and np.allclose(ks[1:], 0, atol=1e-9)


def test_true_knee_flexion_range_is_gait_like(traj):
    k = _true_knee(traj, "R")
    rom = k[:, 0].max() - k[:, 0].min()
    assert 45 < rom < 75


def test_monte_carlo_zero_noise_is_zero_error(traj):
    cams = _rig()
    vis = views(traj, cams)
    err = knee_monte_carlo(traj, cams, vis, "R", sigma_px=0.0, n_mc=2)
    assert err.fail_frac < 1
    assert np.all(err.rmse < 1e-6)
