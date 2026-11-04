"""End-to-end on synthetic data: two straight flights, a synthetic recording,
and a small grid. The true observer must rank first."""
import numpy as np

from dopplertrack import audio, doppler, search, score, tracks
from dopplertrack.adsb import Trace
from dopplertrack.geo import enu_to_lla


def _synth_trace(icao, t, enu, lat0, lon0, gs):
    lat, lon, alt = enu_to_lla(enu[:, 0], enu[:, 1], enu[:, 2], lat0, lon0)
    return Trace(icao, None, "C172", "synthetic", t, lat, lon, alt, np.full_like(t, gs), np.zeros(len(t), bool))


def test_true_observer_ranks_first():
    c = 340.0
    lat0, lon0 = 40.8765, -73.525
    sr = 4000
    T0 = 1_700_000_000.0                     # recording starts at this unix time
    # flight 1: east-bound, passes 2.5 km north of origin at 1500 m, t_cpa_emit = T0+300
    t1 = T0 + np.arange(0, 900, 5.0)
    enu1 = np.stack([110.0 * (t1 - (T0 + 300)), np.full_like(t1, 2500.0), np.full_like(t1, 1500.0)], -1)
    # flight 2: north-bound with a gentle turn, passes ~3 km east at 1200 m, t_cpa_emit ≈ T0+1220
    t2 = T0 + np.arange(900, 1800, 5.0)
    tau = t2 - (T0 + 1220)
    enu2 = np.stack([3000.0 + 0.02 * tau**2, 95.0 * tau, np.full_like(t2, 1200.0)], -1)
    traces = [_synth_trace("aaa001", t1, enu1, lat0, lon0, 110 / 0.5144), _synth_trace("aaa002", t2, enu2, lat0, lon0, 95 / 0.5144)]
    segs = search.build_segments(traces, lat0, lon0)

    # synthesize the audio the observer at the origin hears: f0=120 Hz and 130 Hz
    obs = np.zeros(3)
    dur = 1800
    tt = np.arange(0, dur, 1 / sr)
    x = 0.4 * np.random.default_rng(1).standard_normal(len(tt)).astype(np.float32)
    for seg, f0 in zip(segs, (120.0, 130.0)):
        to, ratio, _ = doppler.curve_from_trajectory(seg.t, seg.enu, obs, f0=1.0, c=c)
        finst = np.interp(tt + T0, to, ratio, left=np.nan, right=np.nan) * f0
        m = np.isfinite(finst)
        ph = 2 * np.pi * np.cumsum(np.nan_to_num(finst, nan=0.0)) / sr
        x[m] += 0.05 * np.sin(ph[m])
    sp = audio.compute_spectrogram(x, sr, nperseg=8192, hop_s=0.5, fmin=60, fmax=300, whiten_window_s=None)

    found = tracks.detect_events(sp, 90, 180, c=c, top=4, refine=False, workers=1)
    cpas = sorted(e.t_cpa for e in found[:2])
    assert abs(cpas[0] - (300 + 2500 / c)) < 15 and abs(cpas[1] - (1220 + 3000 / c)) < 15, [e.t_cpa for e in found]

    ev1 = tracks.fit_event(sp, 230, 400, 90, 170, c=c)
    ev2 = tracks.fit_event(sp, 1150, 1310, 100, 180, c=c)
    assert abs(ev1.v - 110) < 25 and abs(ev2.v - 95) < 25
    evb = tracks.fit_event(sp, 1150, 1310, 100, 180, c=c, objective="brightness")    # what `fit` uses
    assert abs(evb.v - 95) < 25 and abs(evb.t_cpa - ev2.t_cpa) < 6
    dt_meas = ev2.t_cpa - ev1.t_cpa
    assert abs(dt_meas - (1220 - 300 + 3000 / c - 2500 / c)) < 6   # heard-CPA offset

    grid = search.Grid.make(lat0 - 0.05, lat0 + 0.05, lon0 - 0.06, lon0 + 0.06, 500.0)
    cpa = search.cpa_table(grid, segs, max_range_m=12000, c=c)
    pairs = search.pair_candidates(cpa, dt_meas, tol_s=8,
                                   v1=(ev1.v * 0.7, ev1.v * 1.3), d1=(ev1.d * 0.5, ev1.d * 2),
                                   v2=(ev2.v * 0.7, ev2.v * 1.3), d2=(ev2.d * 0.5, ev2.d * 2), silence_d_m=None)
    assert len(pairs) > 0
    w1 = score.EventWindow.from_fit(sp, ev1); w2 = score.EventWindow.from_fit(sp, ev2)
    results = []
    for _, p in pairs.iterrows():
        tot, s1, s2, sh = score.locked_score(w1, w2, segs[int(p.seg1)], segs[int(p.seg2)], grid.enu[int(p.gidx)], float(p.t_heard1), c=c)
        results.append((tot, int(p.gidx)))
    results.sort(reverse=True)
    best_g = results[0][1]
    dist = np.hypot(*grid.enu[best_g][:2])
    assert dist < 600.0, f"best candidate {dist:.0f} m from truth"

    # refine around the winner with event 2 passed again as an "extra" event: pick_flight must
    # choose flight 2, and its locked score must equal s2 at every fine point
    import pandas as pd
    from dopplertrack import pipeline
    p = pairs[pairs.gidx == best_g].iloc[0]
    tot, s1, s2, sh = score.locked_score(w1, w2, segs[int(p.seg1)], segs[int(p.seg2)], grid.enu[best_g], float(p.t_heard1), c=c)
    row = pd.Series({"lat": grid.lat[best_g], "lon": grid.lon[best_g], "seg1": int(p.seg1), "seg2": int(p.seg2), "icao1": "aaa001", "icao2": "aaa002",
                     "t_heard1": float(p.t_heard1), "shift": sh, "score": tot, "lat0": lat0, "lon0": lon0})
    segs2 = pipeline.segments_for_search(traces, (lat0 - 0.05, lat0 + 0.05, lon0 - 0.06, lon0 + 0.06), lat0, lon0)
    row = pipeline.relocate_row(row, segs2, ev2.t_cpa - ev1.t_cpa)
    assert segs2[int(row.seg1)].icao == "aaa001" and segs2[int(row.seg2)].icao == "aaa002"
    fine = pipeline.refine(sp, ev1, ev2, segs2, row, spacing_m=250.0, half_m=500.0, extra=[ev2], c=c)
    assert fine.attrs["extra_icao"] == ["aaa002"]
    assert np.allclose(fine.s3, fine.s2) and np.allclose(fine.score, fine.s1 + fine.s2 + fine.s3)
    assert np.hypot(*(np.array([(fine.lat[0] - lat0) * 111320, (fine.lon[0] - lon0) * 111320 * np.cos(np.radians(lat0))]))) < 600.0


def test_build_segments_clips_to_box():
    lat0, lon0 = 40.0, -74.0
    t = 1e9 + np.arange(0, 2000, 5.0)
    enu = np.stack([100.0 * (t - t[0]) - 100000, np.zeros_like(t), np.full_like(t, 1000.0)], -1)
    tr = _synth_trace("x", t, enu, lat0, lon0, 200)
    segs = search.build_segments([tr], lat0, lon0, clip_enu=(-20000, 20000, -5000, 5000))
    assert len(segs) == 1
    assert segs[0].enu[:, 0].min() >= -20000 - 100 and segs[0].enu[:, 0].max() <= 20000 + 100
    assert abs(segs[0].speed.mean() - 100.0) < 1.0
