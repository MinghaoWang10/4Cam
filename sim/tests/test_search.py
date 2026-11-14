import numpy as np
import pytest

from sim import trajectory
from sim.camera import Placement
from sim.pipeline import knee_monte_carlo, views
from sim.search import CandidateCache, Grid, Scorer, objective, stage_a, stage_b

AIM = (0.0, 550.0, 0.0)


@pytest.fixture(scope="module")
def traj():
    return trajectory.load(travel_mm=1300)


@pytest.mark.parametrize("layout", [
    [Placement(a, 3000, 1300) for a in (45, 135, -45, -135)],
    [Placement(a, 2500, 300) for a in (60, 120, -60, -120)],
    [Placement(300, 2500, 300), Placement(330, 2500, 300),
     Placement(330, 2500, 1300), Placement(300, 3000, 300)],
])
def test_analytic_error_matches_monte_carlo(traj, layout):
    cache = CandidateCache(traj, layout, AIM, 50)
    fail, rot, trn = Scorer(cache, "R").knee_error(range(4), sigma_px=0.3)
    cams = [cache.camera(i) for i in range(4)]
    e = knee_monte_carlo(traj, cams, views(traj, cams), "R", 0.3, n_mc=150, seed=3)
    assert fail == pytest.approx(e.fail_frac)
    assert rot == pytest.approx(np.sqrt(np.mean(e.rmse[:3] ** 2)), rel=0.15)
    assert trn == pytest.approx(np.sqrt(np.mean(e.rmse[3:] ** 2)), rel=0.15)


def test_objective_constraint():
    assert objective((0.0, 1.0), 0) == 1.0
    assert objective((2 / 51, 0.1), 2) == 0.1
    assert objective((3 / 51, 0.1), 2) > objective((0.0, 50.0), 2)


def test_stage_b_never_worsens(traj):
    grid = Grid(distances=(2000, 3000), heights=(300, 1300))
    cache = CandidateCache(traj, grid.placements(), AIM, 50)
    sc = Scorer(cache, "R")
    start = stage_a(cache, sc, grid, keep=1)[0][0]
    s0 = sc.score(start)
    m, s1 = stage_b(sc, start, max_fail_frames=0)
    assert round(s1[0] * 51) <= max(0, round(s0[0] * 51))
    if round(s0[0] * 51) == 0:
        assert s1[1] <= s0[1]
    assert s1 == sc.score(m)
