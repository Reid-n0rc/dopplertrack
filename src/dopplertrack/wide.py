"""Nationwide candidate generation: start from the flights, not from a grid.

A regional search grids a box and asks every point which flights it heard; that cannot scale
to a continent. This module turns the question around, the way colsto did. Two events heard
dt apart at slant distances d1 and d2 mean the two aircraft were within d1 + d2 of each other
at emission times dt apart (give or take the propagation delays). That is a cheap geometric
join over a day's trajectories, and it leaves a few tens of thousands of flight pairs, each
with a small lens of ground where both could have been heard. Ground points are laid only in
those lenses and tested with the same exact CPA quantities, gates, silence filter and locked
scoring as the regional search, so the result table has the same columns and `refine` works
on it unchanged.

Stages, each a function of the previous one's output so it can be cached on its inputs:
  1. DayIndex       a day's prop flights as 1 Hz ENU segments in ONE frame (lla_to_enu is a
                    true ECEF rotation, so slant distances are exact anywhere), plus every
                    trajectory sampled once per time bucket (10 s), sorted by bucket.
  2. pair_join      (segment 1, segment 2, bucket range) for every pair the geometry allows.
  3. lens_candidates per pair, a coarse then fine lattice of ground points with exact CPA
                    values, gated as in the regional search, then silence-filtered.
  4. score_candidates locked_score per (point, pair) triple, in a process pool.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .adsb import Trace
from .doppler import speed_of_sound
from .geo import enu_to_lla, lla_to_enu
from .search import Segment, build_segments


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def default_workers() -> int:
    """Process pool size: every core on Linux; 1 on macOS, where forking a process that has
    already used BLAS threads is unsafe (pass workers explicitly to override)."""
    return 1 if sys.platform == "darwin" else (os.cpu_count() or 1)


# ----------------------------------------------------------------------------------------
# 1. the day index
# ----------------------------------------------------------------------------------------

@dataclass
class DayIndex:
    lat0: float
    lon0: float
    bucket_s: float
    off: np.ndarray          # (S+1,) segment offsets into t/enu/spd
    t: np.ndarray            # (T,) 1 Hz emission times, unix s
    enu: np.ndarray          # (T,3)
    spd: np.ndarray          # (T,)
    icao: np.ndarray         # (S,) str
    trace_idx: np.ndarray    # (S,)
    sb: np.ndarray           # (M,) bucket id per sample, sorted
    ss: np.ndarray           # (M,) segment index per sample
    sj: np.ndarray           # (M,) sample index inside the segment
    segs: list[Segment] = field(default_factory=list, repr=False)
    spos: np.ndarray = field(default=None, repr=False)   # (M,3) sample positions
    sspd: np.ndarray = field(default=None, repr=False)   # (M,) sample speeds
    _trees: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not self.segs:
            self.segs = [Segment(int(self.trace_idx[i]), str(self.icao[i]), self.t[a:b], self.enu[a:b], self.spd[a:b])
                         for i, (a, b) in enumerate(zip(self.off[:-1], self.off[1:]))]
        if self.spos is None:
            k = self.off[self.ss] + self.sj
            self.spos = self.enu[k]; self.sspd = self.spd[k]

    @classmethod
    def build(cls, traces: list[Trace], lat0: float | None = None, lon0: float | None = None,
              bucket_s: float = 10.0, min_len: int = 20) -> "DayIndex":
        if lat0 is None:
            lat0 = float(np.mean([np.median(tr.lat) for tr in traces])); lon0 = float(np.mean([np.median(tr.lon) for tr in traces]))
        segs = build_segments(traces, lat0, lon0, min_len=min_len)
        off = np.r_[0, np.cumsum([len(s.t) for s in segs])].astype(np.int64)
        t = np.concatenate([s.t for s in segs]) if segs else np.zeros(0)
        enu = np.concatenate([s.enu for s in segs]) if segs else np.zeros((0, 3))
        spd = np.concatenate([s.speed for s in segs]) if segs else np.zeros(0)
        sb, ss, sj = [], [], []
        for i, s in enumerate(segs):
            b = np.arange(np.ceil((s.t[0] - bucket_s / 2) / bucket_s), np.floor((s.t[-1] - bucket_s / 2) / bucket_s) + 1, dtype=np.int64)
            j = np.rint((b + 0.5) * bucket_s - s.t[0]).astype(np.int64)
            ok = (j >= 0) & (j < len(s.t))
            sb.append(b[ok]); ss.append(np.full(int(ok.sum()), i, np.int32)); sj.append(j[ok].astype(np.int32))
        sb = np.concatenate(sb) if sb else np.zeros(0, np.int64)
        ss = np.concatenate(ss) if ss else np.zeros(0, np.int32)
        sj = np.concatenate(sj) if sj else np.zeros(0, np.int32)
        order = np.argsort(sb, kind="stable")
        return cls(lat0, lon0, bucket_s, off, t, enu, spd, np.array([s.icao for s in segs]), np.array([s.trace_idx for s in segs], np.int32),
                   sb[order], ss[order], sj[order], segs=segs)

    def __len__(self):
        return len(self.segs)

    def save(self, path):
        np.savez(path, lat0=self.lat0, lon0=self.lon0, bucket_s=self.bucket_s, off=self.off, t=self.t, enu=self.enu, spd=self.spd,
                 icao=self.icao, trace_idx=self.trace_idx, sb=self.sb, ss=self.ss, sj=self.sj)

    @classmethod
    def load(cls, path) -> "DayIndex":
        z = np.load(path, allow_pickle=False)
        return cls(float(z["lat0"]), float(z["lon0"]), float(z["bucket_s"]), z["off"], z["t"], z["enu"], z["spd"], z["icao"], z["trace_idx"], z["sb"], z["ss"], z["sj"])

    def fingerprint(self) -> dict:
        return {"segs": len(self.segs), "points": int(len(self.t)), "lat0": float(self.lat0), "lon0": float(self.lon0), "bucket_s": float(self.bucket_s)}

    def bucket_time(self, b) -> float:
        return (b + 0.5) * self.bucket_s

    def bucket_slice(self, b: int) -> slice:
        return slice(int(np.searchsorted(self.sb, b, "left")), int(np.searchsorted(self.sb, b, "right")))

    def tree(self, b: int) -> cKDTree | None:
        """KD-tree of every sample in bucket b (cached); None if the bucket is empty."""
        if b not in self._trees:
            sl = self.bucket_slice(b)
            self._trees[b] = cKDTree(self.spos[sl]) if sl.stop > sl.start else None
        return self._trees[b]

    def window(self, si: int, t_a: float, t_b: float) -> slice:
        """Sample slice of segment si with t in [t_a, t_b]."""
        s = self.segs[si]
        return slice(int(np.searchsorted(s.t, t_a, "left")), int(np.searchsorted(s.t, t_b, "right")))


# ----------------------------------------------------------------------------------------
# 2. the pair join
# ----------------------------------------------------------------------------------------

def lag_range(dt_heard: float, d1, d2, c: float, tol_s: float) -> tuple[float, float]:
    """Emission-time gap between the two CPAs allowed by the heard gap dt +- tol and the
    slant-distance gates: t_cpa2 - t_cpa1 = dt_heard + (d1 - d2) / c."""
    return dt_heard + (d1[0] - d2[1]) / c - tol_s, dt_heard + (d1[1] - d2[0]) / c + tol_s


def pair_join(idx: DayIndex, dt_heard: float, v1, d1, v2, d2, tol_s: float = 6.0, c: float | None = None,
              v_slack: float = 0.1, progress=None) -> pd.DataFrame:
    """Flight pairs that could be heard as event 1 then event 2 by one observer: in some
    bucket k flight 1 (speed in v1) is within d1max + d2max of where flight 2 (speed in v2) is
    at the allowed lag. Returns one row per pair: seg1, seg2, b_min, b_max (buckets of the
    flight-1 side), n (matched buckets). Nothing here is exact; it only has to be complete."""
    c = speed_of_sound() if c is None else c
    W = idx.bucket_s
    lo, hi = lag_range(dt_heard, d1, d2, c, tol_s)
    v1g = (v1[0] * (1 - v_slack), v1[1] * (1 + v_slack)); v2g = (v2[0] * (1 - v_slack), v2[1] * (1 + v_slack))
    R = d1[1] + d2[1] + (v1g[1] + v2g[1]) * W / 2
    A = (idx.sspd >= v1g[0]) & (idx.sspd <= v1g[1])
    B = (idx.sspd >= v2g[0]) & (idx.sspd <= v2g[1])
    buckets = np.unique(idx.sb[A])
    btrees: dict[int, tuple] = {}
    out1, out2, outb = [], [], []
    t0 = time.time()
    for n, k in enumerate(buckets):
        sl = idx.bucket_slice(int(k)); a_idx = np.flatnonzero(A[sl]) + sl.start
        ta = cKDTree(idx.spos[a_idx])
        tk = idx.bucket_time(k)
        kb_lo = int(np.floor((tk - W / 2 + lo) / W)); kb_hi = int(np.floor((tk + W / 2 + hi) / W))
        for kb in list(btrees):
            if kb < kb_lo:
                del btrees[kb]
        for kb in range(kb_lo, kb_hi + 1):
            if kb not in btrees:
                slb = idx.bucket_slice(kb); b_idx = np.flatnonzero(B[slb]) + slb.start
                btrees[kb] = (b_idx, cKDTree(idx.spos[b_idx])) if len(b_idx) else (b_idx, None)
            b_idx, tb = btrees[kb]
            if tb is None:
                continue
            m = ta.sparse_distance_matrix(tb, R, output_type="coo_matrix")
            if m.nnz == 0:
                continue
            s1 = idx.ss[a_idx[m.row]]; s2 = idx.ss[b_idx[m.col]]
            keep = s1 != s2
            out1.append(s1[keep]); out2.append(s2[keep]); outb.append(np.full(int(keep.sum()), k, np.int64))
        if progress and (n + 1) % 1000 == 0:
            progress(n + 1, len(buckets), time.time() - t0)
    if not out1:
        return pd.DataFrame({"seg1": np.zeros(0, np.int32), "seg2": np.zeros(0, np.int32), "b_min": np.zeros(0, np.int64), "b_max": np.zeros(0, np.int64), "n": np.zeros(0, np.int64)})
    df = pd.DataFrame({"seg1": np.concatenate(out1), "seg2": np.concatenate(out2), "b": np.concatenate(outb)}).drop_duplicates()
    g = df.groupby(["seg1", "seg2"], sort=False)["b"]
    return g.agg(b_min="min", b_max="max", n="size").reset_index()


# ----------------------------------------------------------------------------------------
# 3. lens lattices, exact gates, silence filter
# ----------------------------------------------------------------------------------------

def cpa_to_points(P: np.ndarray, X: np.ndarray, T: np.ndarray, V: np.ndarray, c: float):
    """Exact CPA of one trajectory window (X (N,3), T, V) to many points P (G,3):
    (d, t_heard, v, interior); interior is False where the closest sample is the window's first
    or last, i.e. the true CPA lies outside the window. A KD-tree nearest-sample query rather
    than a dense G x N distance matrix: same answer, no multi-megabyte temporaries, so a
    process per core scales instead of fighting over memory bandwidth."""
    d, j = cKDTree(X).query(P, workers=1)
    return d, T[j] + d / c, V[j], (j > 0) & (j < len(T) - 1)


@dataclass
class LensParams:
    dt_heard: float
    v1: tuple; d1: tuple; v2: tuple; d2: tuple
    tol_s: float = 6.0
    c: float = 340.0
    spacing_m: float = 200.0
    coarse_m: float = 1000.0
    max_range_m: float = 12000.0
    silence_d_m: float | None = 6000.0
    silence_margin_s: float = 40.0
    pad_s: float = 30.0

    def fingerprint(self) -> dict:
        return {k: (tuple(float(x) for x in v) if isinstance(v, (tuple, list)) else v) for k, v in self.__dict__.items()}


_G: dict = {}      # per-process state for the pools (set before forking)


def _ground_lattice(e, n, lat_c, lon_c, elevation, lat0, lon0):
    """Local tangent-plane offsets (e, n) at (lat_c, lon_c) -> lat, lon, elev, global ENU."""
    lat, lon, _ = enu_to_lla(e, n, np.zeros_like(e), lat_c, lon_c)
    el = np.maximum(elevation(lat, lon), 0.0) if elevation is not None else np.zeros_like(lat)
    return lat, lon, el, lla_to_enu(lat, lon, el, lat0, lon0)


def _lens_one(row) -> pd.DataFrame | None:
    idx: DayIndex = _G["idx"]; p: LensParams = _G["params"]; elevation = _G.get("elevation")
    seg1, seg2, b_min, b_max = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    s1, s2 = idx.segs[seg1], idx.segs[seg2]
    W, c = idx.bucket_s, p.c
    lo, hi = lag_range(p.dt_heard, p.d1, p.d2, c, p.tol_s)
    t1a, t1b = b_min * W - p.pad_s, (b_max + 1) * W + p.pad_s
    w1 = idx.window(seg1, t1a, t1b)
    w2 = idx.window(seg2, t1a + lo - p.pad_s, t1b + hi + p.pad_s)
    if w1.stop - w1.start < 3 or w2.stop - w2.start < 3:
        return None
    X1, T1, V1 = s1.enu[w1], s1.t[w1], s1.speed[w1]
    X2, T2, V2 = s2.enu[w2], s2.t[w2], s2.speed[w2]
    d1hi = min(p.d1[1], p.max_range_m); d2hi = min(p.d2[1], p.max_range_m)
    # coarse lattice in the tangent plane at the flight-1 window's centre, within d1hi of its path
    lat_c, lon_c, _ = enu_to_lla(*X1.mean(0), idx.lat0, idx.lon0)
    lat1, lon1, _ = enu_to_lla(X1[:, 0], X1[:, 1], X1[:, 2], idx.lat0, idx.lon0)
    P1 = lla_to_enu(lat1, lon1, np.zeros_like(lat1), lat_c, lon_c)[:, :2]
    cm = p.coarse_m
    e = np.arange(np.floor((P1[:, 0].min() - d1hi) / cm), np.ceil((P1[:, 0].max() + d1hi) / cm) + 1) * cm
    n = np.arange(np.floor((P1[:, 1].min() - d1hi) / cm), np.ceil((P1[:, 1].max() + d1hi) / cm) + 1) * cm
    E, N = np.meshgrid(e, n, indexing="ij"); E, N = E.ravel(), N.ravel()
    near = cKDTree(P1).query(np.stack([E, N], 1), distance_upper_bound=d1hi + cm)[0] <= d1hi + cm
    E, N = E[near], N[near]
    if len(E) == 0:
        return None
    _, _, _, G = _ground_lattice(E, N, lat_c, lon_c, None, idx.lat0, idx.lon0)
    sd = 0.75 * cm; st = sd * (1.0 / max(min(p.v1[0], p.v2[0]), 10.0) + 1.0 / c)
    da, ha, va, ia = cpa_to_points(G, X1, T1, V1, c); db, hb, vb, ib = cpa_to_points(G, X2, T2, V2, c)
    ok = (ia & ib & (da >= p.d1[0] - sd) & (da <= d1hi + sd) & (db >= p.d2[0] - sd) & (db <= d2hi + sd)
          & (va >= p.v1[0] * 0.9) & (va <= p.v1[1] * 1.1) & (vb >= p.v2[0] * 0.9) & (vb <= p.v2[1] * 1.1)
          & (np.abs(hb - ha - p.dt_heard) <= p.tol_s + 2 * st))
    if not ok.any():
        return None
    # fine lattice around each surviving coarse cell
    m = int(np.ceil(cm / 2 / p.spacing_m))
    offs = np.arange(-m, m + 1) * p.spacing_m
    OE, ON = np.meshgrid(offs, offs, indexing="ij")
    Ef = (E[ok][:, None] + OE.ravel()[None, :]).ravel(); Nf = (N[ok][:, None] + ON.ravel()[None, :]).ravel()
    key = np.unique(np.rint(np.stack([Ef, Nf], 1) / p.spacing_m).astype(np.int64), axis=0)
    Ef, Nf = key[:, 0] * p.spacing_m, key[:, 1] * p.spacing_m
    lat, lon, el, G = _ground_lattice(Ef, Nf, lat_c, lon_c, elevation, idx.lat0, idx.lon0)
    da, ha, va, ia = cpa_to_points(G, X1, T1, V1, c); db, hb, vb, ib = cpa_to_points(G, X2, T2, V2, c)
    ok = (ia & ib & (da >= p.d1[0]) & (da <= d1hi) & (db >= p.d2[0]) & (db <= d2hi)
          & (va >= p.v1[0]) & (va <= p.v1[1]) & (vb >= p.v2[0]) & (vb <= p.v2[1])
          & (np.abs(hb - ha - p.dt_heard) <= p.tol_s))
    if not ok.any():
        return None
    F = G[ok]; ha, da, va, hb, db, vb = ha[ok], da[ok], va[ok], hb[ok], db[ok], vb[ok]
    lat, lon, el = lat[ok], lon[ok], el[ok]
    # silence: no other flight may pass within silence_d_m with its CPA heard between the events
    if p.silence_d_m is not None:
        mg = p.silence_margin_s
        ta, tb = float(ha.min()) + mg, float(hb.max()) - mg
        if tb > ta:
            ctr = F.mean(0); rad = float(np.linalg.norm(F - ctr, axis=1).max())
            others: set[int] = set()
            for b in range(int(np.floor(ta / W)), int(np.floor(tb / W)) + 1):
                tr = idx.tree(b)
                if tr is None:
                    continue
                sl = idx.bucket_slice(b)
                hit = tr.query_ball_point(ctr, rad + p.silence_d_m + 150.0 * W / 2)
                others.update(int(x) for x in idx.ss[np.asarray(hit, int) + sl.start])
            others.discard(seg1); others.discard(seg2)
            bad = np.zeros(len(F), bool)
            for so in others:
                w = idx.window(so, ta - 60.0, tb + 60.0)
                if w.stop - w.start < 3:
                    continue
                s = idx.segs[so]
                do, ho, _, io = cpa_to_points(F, s.enu[w], s.t[w], s.speed[w], c)
                bad |= io & (do <= p.silence_d_m) & (ho > ha + mg) & (ho < hb - mg)
            if bad.all():
                return None
            keep = ~bad
            F, ha, da, va, hb, db, vb, lat, lon, el = (x[keep] for x in (F, ha, da, va, hb, db, vb, lat, lon, el))
    return pd.DataFrame({"lat": lat, "lon": lon, "elev": el, "e": F[:, 0], "n": F[:, 1], "u": F[:, 2],
                         "seg1": np.int32(seg1), "seg2": np.int32(seg2), "t_heard1": ha, "d1": da.astype(np.float32), "v1": va.astype(np.float32),
                         "t_heard2": hb, "d2": db.astype(np.float32), "v2": vb.astype(np.float32)})


CAND_COLUMNS = ["lat", "lon", "elev", "e", "n", "u", "seg1", "seg2", "t_heard1", "d1", "v1", "t_heard2", "d2", "v2"]


def empty_candidates() -> pd.DataFrame:
    return pd.DataFrame({k: np.zeros(0, np.int32 if k in ("seg1", "seg2") else float) for k in CAND_COLUMNS})


def _pool_map(fn, items, workers: int, chunksize: int = 8):
    if workers <= 1:
        for it in items:
            yield fn(it)
        return
    from multiprocessing import get_context
    with get_context("fork").Pool(workers) as pool:
        yield from pool.imap_unordered(fn, items, chunksize=chunksize)


def lens_candidates(idx: DayIndex, pairs: pd.DataFrame, params: LensParams, elevation=None,
                    workers: int | None = None, progress=None) -> pd.DataFrame:
    """Exact (point, flight pair) candidates for every pair from pair_join: a 200 m lattice
    (params.spacing_m) wherever a 1 km (params.coarse_m) sweep passed loosened gates, with
    the regional search's gates and silence filter applied exactly."""
    workers = default_workers() if workers is None else workers
    _G.update(idx=idx, params=params, elevation=elevation)
    rows = pairs[["seg1", "seg2", "b_min", "b_max"]].to_numpy()
    out = []
    t0 = time.time()
    for k, df in enumerate(_pool_map(_lens_one, list(rows), workers)):
        if df is not None:
            out.append(df)
        if progress and (k + 1) % 1000 == 0:
            progress(k + 1, len(rows), time.time() - t0)
    return pd.concat(out, ignore_index=True) if out else empty_candidates()


def save_candidates(path, cands: pd.DataFrame, fingerprint: dict):
    np.savez_compressed(path, fingerprint=np.array(repr(fingerprint)), **{k: cands[k].to_numpy() for k in CAND_COLUMNS})


def load_candidates(path, fingerprint: dict) -> pd.DataFrame | None:
    z = np.load(path, allow_pickle=False)
    if str(z["fingerprint"]) != repr(fingerprint):
        return None
    return pd.DataFrame({k: z[k] for k in CAND_COLUMNS})


# ----------------------------------------------------------------------------------------
# 4. scoring
# ----------------------------------------------------------------------------------------

def _score_chunk(args):
    from .score import locked_score
    idx: DayIndex = _G["idx"]; w1 = _G["w1"]; w2 = _G["w2"]; c = _G["c"]
    i0, seg1, seg2, obs, th1 = args
    out = np.empty((len(seg1), 4))
    for k in range(len(seg1)):
        out[k] = locked_score(w1, w2, idx.segs[int(seg1[k])], idx.segs[int(seg2[k])], obs[k], float(th1[k]), c=c)
    return i0, out


def score_candidates(sp, ev1, ev2, idx: DayIndex, cands: pd.DataFrame, harmonics=(1, 1), score_mode="brightness",
                     c: float | None = None, workers: int | None = None, chunk: int = 200, progress=None) -> pd.DataFrame:
    """locked_score for every candidate row; returns the regional search's result table
    (gidx, lat, lon, seg1, seg2, icao1, icao2, t_heard1, d1, v1, d2, v2, score, s1, s2, shift),
    best first, with elev added and lat0/lon0 of the index frame."""
    from .score import EventWindow
    c = speed_of_sound() if c is None else c
    workers = default_workers() if workers is None else workers
    w1 = EventWindow.from_fit(sp, ev1, harmonics[0], mode=score_mode)
    w2 = EventWindow.from_fit(sp, ev2, harmonics[1], mode=score_mode)
    _G.update(idx=idx, w1=w1, w2=w2, c=c)
    obs = cands[["e", "n", "u"]].to_numpy(); s1 = cands.seg1.to_numpy(); s2 = cands.seg2.to_numpy(); th = cands.t_heard1.to_numpy()
    jobs = [(a, s1[a:a + chunk], s2[a:a + chunk], obs[a:a + chunk], th[a:a + chunk]) for a in range(0, len(cands), chunk)]
    sc = np.zeros((len(cands), 4))
    t0 = time.time(); done = 0
    for i0, out in _pool_map(_score_chunk, jobs, workers, chunksize=1):
        sc[i0:i0 + len(out)] = out; done += len(out)
        if progress:
            progress(done, len(cands), time.time() - t0)
    res = pd.DataFrame({
        "gidx": np.arange(len(cands)), "lat": cands.lat.to_numpy(), "lon": cands.lon.to_numpy(), "elev": cands.elev.to_numpy(),
        "seg1": s1, "seg2": s2, "icao1": idx.icao[s1], "icao2": idx.icao[s2],
        "t_heard1": th, "d1": cands.d1.to_numpy(), "v1": cands.v1.to_numpy(), "d2": cands.d2.to_numpy(), "v2": cands.v2.to_numpy(),
        "score": sc[:, 0], "s1": sc[:, 1], "s2": sc[:, 2], "shift": sc[:, 3]})
    res["lat0"] = idx.lat0; res["lon0"] = idx.lon0
    return res.sort_values("score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# the whole thing
# ----------------------------------------------------------------------------------------

def run_wide_search(sp, ev1, ev2, idx: DayIndex, harmonics=(1, 1), tol_s=6.0, v_tol=0.35, d_tol=(0.4, 2.5),
                    silence_d_m=6000.0, max_range_m=12000.0, spacing_m=200.0, coarse_m=1000.0, elevation=None,
                    c=None, score_mode="brightness", workers=None, cand_cache=None, log=_log) -> pd.DataFrame:
    """Pair join -> lens candidates -> locked scores for one day index. cand_cache: optional
    .npz path; the candidate table depends on the index, the events' gates and the geometry
    parameters but not on the spectrogram or scorer, so it is reused when those match."""
    c = speed_of_sound() if c is None else c
    dt = ev2.t_cpa - ev1.t_cpa
    params = LensParams(dt_heard=float(dt), v1=(ev1.v * (1 - v_tol), ev1.v * (1 + v_tol)), d1=(ev1.d * d_tol[0], min(ev1.d * d_tol[1], max_range_m)),
                        v2=(ev2.v * (1 - v_tol), ev2.v * (1 + v_tol)), d2=(ev2.d * d_tol[0], min(ev2.d * d_tol[1], max_range_m)),
                        tol_s=tol_s, c=float(c), spacing_m=spacing_m, coarse_m=coarse_m, max_range_m=max_range_m, silence_d_m=silence_d_m)
    fp = {"index": idx.fingerprint(), "params": params.fingerprint(), "dem": elevation is not None}
    cands = None
    if cand_cache and os.path.exists(cand_cache):
        cands = load_candidates(cand_cache, fp)
        log(f"loaded {len(cands)} candidates from {cand_cache}" if cands is not None else f"{cand_cache} is for other inputs; recomputing")
    if cands is None:
        t0 = time.time()
        pairs = pair_join(idx, dt, params.v1, params.d1, params.v2, params.d2, tol_s=tol_s, c=c,
                          progress=lambda k, n, s: log(f"  join: bucket {k}/{n}, {s:.0f} s"))
        log(f"{len(pairs)} flight pairs pass the join ({time.time() - t0:.0f} s); gates v1 {params.v1[0]:.0f}-{params.v1[1]:.0f} m/s, d1 {params.d1[0]:.0f}-{params.d1[1]:.0f} m, v2 {params.v2[0]:.0f}-{params.v2[1]:.0f} m/s, d2 {params.d2[0]:.0f}-{params.d2[1]:.0f} m")
        t0 = time.time()
        cands = lens_candidates(idx, pairs, params, elevation=elevation, workers=workers,
                                progress=lambda k, n, s: log(f"  lens: pair {k}/{n}, {s:.0f} s, {s / k * 1000:.1f} ms each"))
        log(f"{len(cands)} candidate (point, pair) triples on {cands.groupby(['seg1', 'seg2']).ngroups if len(cands) else 0} pairs ({time.time() - t0:.0f} s)")
        if cand_cache:
            save_candidates(cand_cache, cands, fp)
    if not len(cands):
        res = pd.DataFrame(columns=["gidx", "lat", "lon", "elev", "seg1", "seg2", "icao1", "icao2", "t_heard1", "d1", "v1", "d2", "v2", "score", "s1", "s2", "shift", "lat0", "lon0"])
        return res
    t0 = time.time()
    res = score_candidates(sp, ev1, ev2, idx, cands, harmonics=harmonics, score_mode=score_mode, c=c, workers=workers,
                           progress=lambda k, n, s: log(f"  scored {k}/{n} ({s / k * 1000:.1f} ms each)") if k % 5000 < 200 or k == n else None)
    log(f"scored {len(res)} triples in {time.time() - t0:.0f} s")
    return res


# ----------------------------------------------------------------------------------------
# 5. more events: rescore the top pairs with further flyovers locked to the same clock
# ----------------------------------------------------------------------------------------

F0_SCALES = np.arange(0.95, 1.0501, 0.005)


def pick_flight_wide(idx: DayIndex, w, obs_enu, t0_map: float, shift: float, c: float,
                     max_range_m: float = 12000.0, exclude=()) -> tuple[int, float]:
    """Which flight does an observer at obs_enu hear as the event in window w, given the
    recording's clock (audio = unix + t0_map) and the time shift chosen for event 1? The
    flights within max_range at that time come from the index's bucket trees; the best
    locked score wins. Returns (segment index, score), (-1, 0) if nothing is near."""
    from .score import modelled_curve
    t_unix = w.fit.t_cpa - t0_map
    W = idx.bucket_s
    b = int(np.floor(t_unix / W))
    near: set[int] = set()
    for bb in (b - 1, b, b + 1):
        tr = idx.tree(bb)
        if tr is None:
            continue
        sl = idx.bucket_slice(bb)
        hit = tr.query_ball_point(np.asarray(obs_enu, float), max_range_m + 150.0 * W)
        near.update(int(x) for x in idx.ss[np.asarray(hit, int) + sl.start])
    best, best_s = -1, 0.0
    for si in near - set(exclude):
        seg = idx.segs[si]
        to, r = modelled_curve(seg, obs_enu, c)
        sc = float(w.sweep_scores(to + t0_map, r, [shift], F0_SCALES).max())
        if sc > best_s:
            best, best_s = si, sc
    return best, best_s


def _rescore_pair(args):
    from .score import modelled_curve
    idx: DayIndex = _G["idx"]; t1 = _G["ev1_t_cpa"]; xw = _G["xw"]; c = _G["c"]; mr = _G["max_range_m"]
    i0, obs, th1, shift, seg1, seg2 = args
    k = int(np.argmax(_G["score"][i0:i0 + len(obs)]))                      # the pair's best point chooses the flights
    t0_map = t1 - float(th1[k])
    chosen = [pick_flight_wide(idx, w, obs[k], t0_map, float(shift[k]), c, mr, exclude=(seg1, seg2))[0] for w in xw]
    out = np.zeros((len(obs), len(xw)))
    for j, (w, si) in enumerate(zip(xw, chosen)):
        if si < 0:
            continue
        seg = idx.segs[si]
        for n in range(len(obs)):
            to, r = modelled_curve(seg, obs[n], c)
            out[n, j] = w.sweep_scores(to + (t1 - float(th1[n])), r, [float(shift[n])], F0_SCALES).max()
    return i0, out, chosen


def rescore_with_extra(sp, ev1, extra, idx: DayIndex, res: pd.DataFrame, extra_harmonics=None, top_pairs: int = 200,
                       score_mode="brightness", c: float | None = None, max_range_m: float = 12000.0,
                       workers: int | None = None, log=_log) -> pd.DataFrame:
    """Add further measured events to the top pairs of a search-wide result. For each of the
    best `top_pairs` flight pairs, each extra event is matched to a flight once, at the pair's
    best point (pick_flight_wide), then scored at every point of that pair with that point's
    own time shift, so every event is locked to one clock, exactly as `refine --extra` does
    regionally. Returns those pairs' rows re-ranked by score = s1 + s2 + s3 + ..., with the
    two-event score kept as score12 and the chosen flights as x_icao1, x_icao2, ..."""
    from .score import EventWindow
    c = speed_of_sound() if c is None else c
    workers = default_workers() if workers is None else workers
    extra = list(extra); extra_harmonics = list(extra_harmonics or [1] * len(extra))
    xw = [EventWindow.from_fit(sp, ev, h, mode=score_mode) for ev, h in zip(extra, extra_harmonics)]
    best = res.sort_values("score", ascending=False).drop_duplicates(["seg1", "seg2"]).head(top_pairs)
    keys = set(zip(best.seg1.tolist(), best.seg2.tolist()))
    sel = res[[k in keys for k in zip(res.seg1.tolist(), res.seg2.tolist())]].sort_values(["seg1", "seg2"]).reset_index(drop=True)
    obs = np.stack([lla_to_enu(sel.lat.to_numpy(), sel.lon.to_numpy(), sel.elev.to_numpy(), idx.lat0, idx.lon0)], 0)[0]
    _G.update(idx=idx, ev1_t_cpa=float(ev1.t_cpa), xw=xw, c=c, max_range_m=max_range_m, score=sel.score.to_numpy())
    jobs = []
    bounds = np.flatnonzero(np.r_[True, (sel.seg1.to_numpy()[1:] != sel.seg1.to_numpy()[:-1]) | (sel.seg2.to_numpy()[1:] != sel.seg2.to_numpy()[:-1]), True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        jobs.append((int(a), obs[a:b], sel.t_heard1.to_numpy()[a:b], sel["shift"].to_numpy()[a:b], int(sel.seg1.iloc[a]), int(sel.seg2.iloc[a])))
    xs = np.zeros((len(sel), len(xw))); chosen = np.full((len(sel), len(xw)), -1, int)
    t0 = time.time()
    for n, (i0, out, ch) in enumerate(_pool_map(_rescore_pair, jobs, workers, chunksize=1)):
        xs[i0:i0 + len(out)] = out; chosen[i0:i0 + len(out)] = ch
        if (n + 1) % 50 == 0:
            log(f"  rescored {n + 1}/{len(jobs)} pairs ({time.time() - t0:.0f} s)")
    sel["score12"] = sel.score
    for j in range(len(xw)):
        sel[f"s{j + 3}"] = xs[:, j]
        sel[f"x_icao{j + 1}"] = np.where(chosen[:, j] >= 0, idx.icao[np.maximum(chosen[:, j], 0)], "")
    sel["score"] = sel.score12 + xs.sum(1)
    log(f"rescored {len(sel)} points on {len(jobs)} pairs with {len(xw)} extra events in {time.time() - t0:.0f} s")
    return sel.sort_values("score", ascending=False).reset_index(drop=True)
