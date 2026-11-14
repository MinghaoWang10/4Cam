"""4 台单相机布局搜索。

候选位置 = 方位角 × 水平距离 × 高度 的网格，每台相机都对准同一个瞄准点。
  阶段 A：4 台共用同一距离和高度，枚举全部方位角组合，只按覆盖率打分；
  阶段 B：从阶段 A 前若干名出发，逐台把相机换到网格上任意位置（可混合高度/距离），
          在「失败帧 ≤ k」约束下按解析 6DOF 误差（线性化协方差传播）下降，直到不再改进；
  阶段 C：对阶段 B 的前几名做端到端 6DOF 蒙特卡洛（由脚本调用 pipeline）。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .camera import Camera, Placement
from .detection import DetectionModel, detect
from .pipeline import KNEE_SEGMENTS, segment_markers
from .trajectory import Trajectory


@dataclass(frozen=True)
class Grid:
    azimuths: tuple = tuple(range(0, 360, 15))
    distances: tuple = (1500, 2000, 2500, 3000, 3500, 4000)
    heights: tuple = (300, 800, 1300, 1800)

    def placements(self) -> list[Placement]:
        return [Placement(float(a), float(d), float(h))
                for d in self.distances for h in self.heights for a in self.azimuths]


class CandidateCache:
    """预先算好每个候选位置的检出结果和 Fisher 信息。"""

    def __init__(self, traj: Trajectory, placements: list[Placement], aim, hfov_deg: float,
                 image_size=(2048, 2048), model: DetectionModel = DetectionModel()):
        self.traj, self.placements, self.aim = traj, placements, np.asarray(aim, float)
        self.hfov, self.image_size = hfov_deg, image_size
        X = traj.points()
        P, N = len(placements), len(X)
        self.det = np.zeros((P, N), bool)
        self.info = np.zeros((P, N, 3, 3))
        for i, pl in enumerate(placements):
            cam = self.camera(i)
            ok = detect(cam, X, traj.occ, model) == 0
            self.det[i] = ok
            idx = np.flatnonzero(ok)
            if idx.size:
                J = cam.jacobian(X[idx])
                self.info[i, idx] = np.einsum("nij,nik->njk", J, J)

    def camera(self, i: int) -> Camera:
        return self.placements[i].camera(self.aim, self.hfov, self.image_size)

    def index(self, az, d, h) -> int:
        for i, p in enumerate(self.placements):
            if (p.azimuth_deg, p.distance_mm, p.height_mm) == (az, d, h):
                return i
        raise KeyError((az, d, h))


class Scorer:
    """针对目标膝（'R' 或 'RL'）的打分。"""

    def __init__(self, cache: CandidateCache, sides: str):
        tr = cache.traj
        self.cache, self.sides, self.T, self.M = cache, sides, tr.T, tr.M
        self.segments = [idx for s in sides for idx in segment_markers(tr, s)]
        mk = np.concatenate(self.segments)
        # 目标标记点对应的展平样本
        self.target_samples = (mk[:, None] * tr.T + np.arange(tr.T)[None]).ravel()

    def frames_ok_batch(self, nviews: np.ndarray) -> np.ndarray:
        """nviews (S, N) -> (S,) 所有目标体段同时有 ≥3 个可重建标记点的帧占比。"""
        rec = (nviews >= 2).reshape(len(nviews), self.M, self.T)
        ok = np.ones((len(nviews), self.T), bool)
        for idx in self.segments:
            ok &= rec[:, idx].sum(axis=1) >= 3
        return ok.mean(axis=1)

    def coverage_batch(self, nviews: np.ndarray) -> np.ndarray:
        return (nviews[:, self.target_samples] >= 2).mean(axis=1)

    def sigma_p90(self, members) -> float:
        """目标样本中可重建者的 σ_worst（σ_px=1）90 分位；不可重建的样本不计入。"""
        s = self.target_samples
        info = self.cache.info[list(members)][:, s].sum(axis=0)
        nv = self.cache.det[list(members)][:, s].sum(axis=0)
        ok = nv >= 2
        if not ok.any():
            return np.inf
        lam = np.linalg.eigvalsh(info[ok])[:, 0]
        sig = np.where(lam > 1e-12, 1 / np.sqrt(np.maximum(lam, 1e-300)), np.inf)
        return float(np.percentile(sig, 90))

    def knee_error(self, members, sigma_px: float = 0.3):
        """解析 6DOF 误差（线性化协方差传播）。

        返回 (fail_frac, rot_deg, trans_mm)：rot/trans 为成功帧上膝相对位姿的
        旋转 RMS（°，三轴平均）与膝关节处平移的 RMS（mm，三轴平均），
        多膝时取最差的一侧。
        """
        tr, T = self.cache.traj, self.T
        mem = list(members)
        info = self.cache.info[mem].sum(axis=0)          # (N,3,3)，σ_px=1
        nv = self.cache.det[mem].sum(axis=0)
        X = tr.points()
        fail = np.zeros(T, bool)
        worst_rot = worst_tr = 0.0
        for side in self.sides:
            rot_cov = np.zeros((T, 3, 3))
            org_cov = np.zeros((T, 3, 3))
            for body in KNEE_SEGMENTS[side]:
                idx = tr.markers_on(body)
                s = (idx[:, None] * T + np.arange(T)[None]).T          # (T, n)
                use = nv[s] >= 2
                ok = use.sum(axis=1) >= 3
                fail |= ~ok
                P = X[s]                                               # (T,n,3)
                w = use[..., None].astype(float)
                c = (P * w).sum(1) / np.maximum(w.sum(1), 1)
                d = P - c[:, None]
                A = np.zeros(s.shape + (3, 6))
                A[..., :3] = -_skew(d)
                A[..., 3:] = np.eye(3)
                I3 = info[s] + np.eye(3) * 1e-9
                Sig = np.linalg.inv(I3) * sigma_px**2                  # (T,n,3,3)
                Aw = A * use[..., None, None]
                H = np.einsum("tnia,tnib->tab", Aw, A)
                Bm = np.einsum("tnia,tnij,tnjb->tab", Aw, Sig * use[..., None, None], A)
                H[~ok] = np.eye(6)
                Hi = np.linalg.inv(H)
                cov = Hi @ Bm @ Hi
                # 平移误差 = 胫骨原点相对"股骨上与之重合的固连点"的运动，两段都在胫骨原点处求
                o = tr.body_origin[:, tr.body(KNEE_SEGMENTS[side][1])]
                G = np.zeros((T, 3, 6))
                G[:, :, :3] = -_skew(o - c)
                G[:, :, 3:] = np.eye(3)
                rot_cov += cov[:, :3, :3]
                org_cov += G @ cov @ np.transpose(G, (0, 2, 1))
            good = ~fail
            if good.any():
                rot = np.rad2deg(np.sqrt(np.trace(rot_cov[good], axis1=1, axis2=2).mean() / 3))
                trn = np.sqrt(np.trace(org_cov[good], axis1=1, axis2=2).mean() / 3)
                worst_rot, worst_tr = max(worst_rot, rot), max(worst_tr, trn)
            else:
                worst_rot = worst_tr = np.inf
        return float(fail.mean()), float(worst_rot), float(worst_tr)

    def score(self, members) -> tuple[float, float]:
        """(失败帧占比, 误差) ；误差 = 旋转 RMS(°) + 平移 RMS(mm)/4，σ_px=0.3。"""
        f, r, t = self.knee_error(members)
        return f, r + t / 4


def _skew(v: np.ndarray) -> np.ndarray:
    S = np.zeros(v.shape + (3,))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S





def objective(sc, max_fail_frames: int, T: int = 51) -> float:
    """在"失败帧 ≤ max_fail_frames"的约束下最小化误差；不满足约束的排在所有可行解之后。"""
    f, e = sc
    nfail = round(f * T)
    return e if nfail <= max_fail_frames else 1e6 * (1 + nfail)


def stage_a(cache: CandidateCache, scorer: Scorer, grid: Grid, keep: int = 30):
    """4 台共用距离/高度，枚举 C(24,4) 个方位角组合。返回 [(members, frames_ok, cov)]。"""
    nA = len(grid.azimuths)
    combos = np.array(list(combinations(range(nA), 4)))
    out = []
    for d in grid.distances:
        for h in grid.heights:
            base = cache.index(float(grid.azimuths[0]), float(d), float(h))
            det = cache.det[base:base + nA].astype(np.int8)     # 同一 (d,h) 下按方位角连续
            for chunk in np.array_split(combos, 8):
                nv = det[chunk].sum(axis=1)
                fo = scorer.frames_ok_batch(nv)
                cv = scorer.coverage_batch(nv)
                for k in np.argsort(-(fo * 10 + cv))[:keep]:
                    out.append((tuple(base + chunk[k]), float(fo[k]), float(cv[k])))
    out.sort(key=lambda r: -(r[1] * 10 + r[2]))
    seen, uniq = set(), []
    for r in out:
        if r[0] not in seen:
            seen.add(r[0])
            uniq.append(r)
    return uniq[:keep]


def stage_b(scorer: Scorer, start, max_fail_frames: int, max_rounds: int = 6):
    """逐台替换为网格上任意位置，在失败帧约束下按解析 6DOF 误差下降，直到不再改进。"""
    T = scorer.T
    members = list(start)
    best_sc = scorer.score(members)
    best = objective(best_sc, max_fail_frames, T)
    det16 = scorer.cache.det.astype(np.int16)
    for _ in range(max_rounds):
        improved = False
        for slot in range(4):
            others = members[:slot] + members[slot + 1:]
            nv = det16[others].sum(axis=0)[None] + det16
            nfail = np.rint((1 - scorer.frames_ok_batch(nv)) * T)   # 与 knee_error 的失败判据相同
            limit = max(max_fail_frames, round(best_sc[0] * T))
            for p in np.flatnonzero(nfail <= limit):
                if p in others or p == members[slot]:
                    continue
                cand = others[:slot] + [int(p)] + others[slot:]
                sc = scorer.score(cand)
                val = objective(sc, max_fail_frames, T)
                if val < best * 0.999:
                    best, best_sc, members, improved = val, sc, cand, True
        if not improved:
            break
    return tuple(sorted(members)), best_sc
