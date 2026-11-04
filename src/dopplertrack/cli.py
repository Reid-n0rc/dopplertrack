"""dopplertrack command line."""
from __future__ import annotations

import argparse
import json
import os
import sys

# One BLAS thread per process: the heavy stages run a process per core, and the matrices
# are small, so BLAS threads on top of that only contend. Set before numpy loads.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from . import pipeline
from .adsb import load_traces
from .audio import Spectrogram
from .search import Grid
from .terrain import Terrain
from .tracks import EventFit


def load_event(path) -> EventFit:
    with open(path) as fh:
        return EventFit(**json.load(fh))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dopplertrack")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spectrogram", help="audio/video file -> whitened low-band spectrogram (.npz)")
    s.add_argument("audio"); s.add_argument("out"); s.add_argument("--hop", type=float, default=0.25)
    s.add_argument("--fmin", type=float, default=40); s.add_argument("--fmax", type=float, default=450)

    f = sub.add_parser("fit", help="fit an ideal Doppler curve inside a time/frequency window")
    f.add_argument("spec"); f.add_argument("--window", nargs=2, type=float, required=True, metavar=("T0", "T1"))
    f.add_argument("--band", nargs=2, type=float, required=True, metavar=("FLO", "FHI"))
    f.add_argument("--harmonics", type=int, default=1); f.add_argument("--out")
    f.add_argument("--objective", choices=("brightness", "contrast"), default="brightness",
                   help="brightness: mean brightness under the curve (the docs/results fits); contrast: the sweep-weighted ridge contrast detect uses")

    d = sub.add_parser("detect", help="scan a whole spectrogram for Doppler events and rank them")
    d.add_argument("spec"); d.add_argument("--band", nargs=2, type=float, required=True, metavar=("FLO", "FHI"))
    d.add_argument("--harmonics", type=int, default=1); d.add_argument("--top", type=int, default=20)
    d.add_argument("--window", type=float, default=240.0); d.add_argument("--out", help="write ranked fits as a JSON list")

    e = sub.add_parser("extract-region", help="pull traces touching a bbox out of an adsb.lol day archive")
    e.add_argument("archive_dir"); e.add_argument("out")
    e.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    e.add_argument("--keep-jets", action="store_true")

    fd = sub.add_parser("fetch-day", help="stream one UTC day of adsb.lol history from GitHub, keeping traces in a bbox (no disk needed)")
    fd.add_argument("date", help="YYYY-MM-DD (UTC day)"); fd.add_argument("out")
    fd.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    fd.add_argument("--keep-jets", action="store_true"); fd.add_argument("--pod", default="planes-readsb-prod-0")

    r = sub.add_parser("search", help="rank candidate locations for a two-event recording")
    r.add_argument("spec"); r.add_argument("--events", nargs=2, required=True, metavar=("EV1_JSON", "EV2_JSON"))
    r.add_argument("--traces", required=True); r.add_argument("--out", required=True)
    r.add_argument("--bbox", nargs=4, type=float, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    r.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"), help="score a single known point instead of a grid")
    r.add_argument("--spacing", type=float, default=200.0); r.add_argument("--harmonics", nargs=2, type=int, default=(1, 1))
    r.add_argument("--no-dem", action="store_true"); r.add_argument("--no-silence", action="store_true")
    r.add_argument("--tol", type=float, default=6.0)
    r.add_argument("--v-tol", type=float, default=0.35, help="speed gate, as a fraction of each event's fitted speed")
    r.add_argument("--d-tol", nargs=2, type=float, default=(0.4, 2.5), metavar=("LO", "HI"), help="distance gate, as multiples of each event's fitted CPA distance")
    r.add_argument("--cpa-cache", metavar="NPZ", help="cache the (point, flight) CPA table here; reused on reruns with the same grid and traces")
    r.add_argument("--score", choices=("brightness", "ridge"), default="brightness",
                   help="brightness: brightness under the modelled curve; ridge: how closely it follows the ridge traced in the window (for faint curves)")

    rf = sub.add_parser("refine", help="fine grid around one search result, optionally adding more events")
    rf.add_argument("spec"); rf.add_argument("results", help="CSV written by search")
    rf.add_argument("--events", nargs=2, required=True, metavar=("EV1_JSON", "EV2_JSON"))
    rf.add_argument("--traces", required=True); rf.add_argument("--out", required=True)
    rf.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"), help="the bbox the search used")
    rf.add_argument("--rank", type=int, default=1, help="which flight pair to refine, by rank in the results (1 = best)")
    rf.add_argument("--spacing", type=float, default=50.0); rf.add_argument("--half", type=float, default=600.0, help="half-width of the fine grid, m")
    rf.add_argument("--harmonics", nargs=2, type=int, default=(1, 1))
    rf.add_argument("--extra", nargs="*", default=[], metavar="EV_JSON", help="further events (from detect or fit) to score locked to the same clock")
    rf.add_argument("--extra-harmonics", nargs="*", type=int, default=[])
    rf.add_argument("--no-dem", action="store_true")
    rf.add_argument("--score", choices=("brightness", "ridge"), default="brightness", help="as for search")

    g = sub.add_parser("search-single", help="rank locations for ONE event whose absolute time is known")
    g.add_argument("spec"); g.add_argument("--event", required=True); g.add_argument("--traces", required=True)
    g.add_argument("--utc", required=True, help="heard CPA time of the event, unix seconds or ISO-8601 UTC")
    g.add_argument("--tol", type=float, default=30.0); g.add_argument("--out", required=True)
    g.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    g.add_argument("--spacing", type=float, default=200.0); g.add_argument("--harmonics", type=int, default=1)
    g.add_argument("--score", choices=("brightness", "ridge"), default="brightness", help="as for search")
    g.add_argument("--v-tol", type=float, default=0.35, help="speed gate, as a fraction of the fitted speed; a faint or turning pass fits badly, so open it (e.g. 1.0)")
    g.add_argument("--d-tol", nargs=2, type=float, default=(0.4, 2.5), metavar=("LO", "HI"), help="distance gate, as multiples of the fitted CPA distance (e.g. 0.05 20)")
    g.add_argument("--no-dem", action="store_true")

    ix = sub.add_parser("index-day", help="build the per-day index a nationwide search runs on (1 Hz segments in one frame + 10 s samples)")
    ix.add_argument("traces", help="traces .npz from extract-region / fetch-day"); ix.add_argument("out", help="index .npz")
    ix.add_argument("--origin", nargs=2, type=float, metavar=("LAT0", "LON0"), help="frame origin (default: the traces' median position)")
    ix.add_argument("--bucket", type=float, default=10.0, help="sample bucket width, s")

    sw = sub.add_parser("search-wide", help="rank candidate locations over a whole day index (continent-scale): pair join, lens lattices, locked scoring")
    sw.add_argument("spec"); sw.add_argument("--events", nargs=2, required=True, metavar=("EV1_JSON", "EV2_JSON"))
    sw.add_argument("--index", required=True, help="index .npz from index-day"); sw.add_argument("--out", required=True)
    sw.add_argument("--cands", metavar="NPZ", help="cache the candidate table here; reused when the index, gates and geometry match")
    sw.add_argument("--spacing", type=float, default=200.0); sw.add_argument("--coarse", type=float, default=1000.0, help="coarse sweep spacing, m")
    sw.add_argument("--harmonics", nargs=2, type=int, default=(1, 1))
    sw.add_argument("--tol", type=float, default=6.0); sw.add_argument("--max-range", type=float, default=12000.0)
    sw.add_argument("--v-tol", type=float, default=0.35, help="as for search"); sw.add_argument("--d-tol", nargs=2, type=float, default=(0.4, 2.5), metavar=("LO", "HI"))
    sw.add_argument("--no-dem", action="store_true"); sw.add_argument("--dem-zoom", type=int, default=9, help="terrain tile zoom for candidate points (9: ~300 m pixels, a few hundred tiles for the whole country)")
    sw.add_argument("--no-silence", action="store_true"); sw.add_argument("--workers", type=int, help="process pool size (default: all cores on Linux, 1 on macOS)")
    sw.add_argument("--score", choices=("brightness", "ridge"), default="brightness", help="as for search")
    sw.add_argument("--top", type=int, default=20)

    a = ap.parse_args(argv)
    if a.cmd == "spectrogram":
        sp = pipeline.make_spectrogram(a.audio, a.out, hop_s=a.hop, fmin=a.fmin, fmax=a.fmax)
        print(f"wrote {a.out}: {sp.S.shape[0]} bins x {sp.S.shape[1]} frames")
    elif a.cmd == "fit":
        sp = Spectrogram.load(a.spec)
        ef = pipeline.fit_event(sp, a.window[0], a.window[1], a.band[0], a.band[1], harmonics=a.harmonics,
                                objective=a.objective)
        js = json.dumps(ef.to_dict(), indent=1)
        print(js)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(js)
    elif a.cmd == "detect":
        from .tracks import detect_events
        sp = Spectrogram.load(a.spec)
        evs = detect_events(sp, a.band[0], a.band[1], harmonics=a.harmonics, top=a.top, window_s=a.window,
                            progress=lambda k, n: print(f"\r  window {k}/{n}", end="", flush=True))
        print()
        print(f"{'rank':>4} {'t_cpa s':>9} {'f0 Hz':>7} {'v m/s':>6} {'d m':>7} {'score':>6}")
        for i, e in enumerate(evs):
            print(f"{i + 1:>4} {e.t_cpa:>9.1f} {e.f0:>7.1f} {e.v:>6.0f} {e.d:>7.0f} {e.score:>6.2f}")
        if a.out:
            with open(a.out, "w") as fh:
                json.dump([e.to_dict() for e in evs], fh, indent=1)
    elif a.cmd == "extract-region":
        pipeline.extract_region(a.archive_dir, tuple(a.bbox), a.out, prop_only=not a.keep_jets)
    elif a.cmd == "fetch-day":
        pipeline.fetch_region(a.date, tuple(a.bbox), a.out, prop_only=not a.keep_jets, pod=a.pod)
    elif a.cmd == "search":
        sp = Spectrogram.load(a.spec)
        ev1, ev2 = (load_event(p) for p in a.events)
        traces = load_traces(a.traces)
        elev = None if a.no_dem else Terrain().elevation
        if a.point:
            el = elev(np.array([a.point[0]]), np.array([a.point[1]])) if elev else np.zeros(1)
            grid = Grid.points([a.point[0]], [a.point[1]], el)
        else:
            grid = Grid.make(*a.bbox, a.spacing, elev)
        res, segs = pipeline.run_search(sp, ev1, ev2, traces, grid, harmonics=tuple(a.harmonics),
                                        silence_d_m=None if a.no_silence else 6000.0, tol_s=a.tol, cpa_cache=a.cpa_cache,
                                        v_tol=a.v_tol, d_tol=tuple(a.d_tol), score_mode=a.score)
        res["lat0"] = grid.lat0; res["lon0"] = grid.lon0
        res.to_csv(a.out, index=False)
        print(res.head(20).to_string())
    elif a.cmd == "refine":
        import pandas as pd
        sp = Spectrogram.load(a.spec)
        ev1, ev2 = (load_event(p) for p in a.events)
        extra = [load_event(p) for p in a.extra]
        xh = list(a.extra_harmonics) + [1] * (len(extra) - len(a.extra_harmonics))
        res = pd.read_csv(a.results)
        best = res.sort_values("score", ascending=False).drop_duplicates(["icao1", "icao2"]).reset_index(drop=True)
        row = best.iloc[a.rank - 1]
        traces = load_traces(a.traces)
        elev = None if a.no_dem else Terrain().elevation
        segs = pipeline.segments_for_search(traces, tuple(a.bbox), float(row.lat0), float(row.lon0))
        row = pipeline.relocate_row(row, segs, ev2.t_cpa - ev1.t_cpa)
        print(f"refining rank {a.rank}: {row.icao1}/{row.icao2} at {row.lat:.5f}, {row.lon:.5f} (score {row.score:.2f})")
        fine = pipeline.refine(sp, ev1, ev2, segs, row, spacing_m=a.spacing, half_m=a.half, harmonics=tuple(a.harmonics),
                               elevation=elev, extra=extra, extra_harmonics=xh, score_mode=a.score)
        fine.to_csv(a.out, index=False)
        b = fine.iloc[0]
        print(f"peak score {b.score:.2f} at {b.lat:.6f}, {b.lon:.6f}" + (f"; extra flights {fine.attrs['extra_icao']}" if extra else ""))
        print(fine.head(10).to_string())
    elif a.cmd == "index-day":
        import time
        from .wide import DayIndex
        t0 = time.time()
        traces = load_traces(a.traces)
        print(f"{len(traces)} traces loaded in {time.time() - t0:.0f} s", flush=True)
        o = a.origin or (None, None)
        idx = DayIndex.build(traces, o[0], o[1], bucket_s=a.bucket)
        idx.save(a.out)
        print(f"wrote {a.out}: {len(idx)} segments, {len(idx.t)} 1 Hz points, {len(idx.sb)} samples, origin {idx.lat0:.4f}, {idx.lon0:.4f}, {time.time() - t0:.0f} s")
    elif a.cmd == "search-wide":
        import time
        from .wide import DayIndex, run_wide_search
        t0 = time.time()
        sp = Spectrogram.load(a.spec)
        ev1, ev2 = (load_event(p) for p in a.events)
        idx = DayIndex.load(a.index)
        print(f"[{time.strftime('%H:%M:%S')}] index: {len(idx)} segments, {len(idx.t)} points, loaded in {time.time() - t0:.0f} s", flush=True)
        elev = None if a.no_dem else Terrain(zoom=a.dem_zoom).elevation
        res = run_wide_search(sp, ev1, ev2, idx, harmonics=tuple(a.harmonics), tol_s=a.tol, v_tol=a.v_tol, d_tol=tuple(a.d_tol),
                              silence_d_m=None if a.no_silence else 6000.0, max_range_m=a.max_range, spacing_m=a.spacing, coarse_m=a.coarse,
                              elevation=elev, score_mode=a.score, workers=a.workers, cand_cache=a.cands)
        res.to_csv(a.out, index=False)
        print(res.head(a.top).to_string())
    elif a.cmd == "search-single":
        from datetime import datetime, timezone
        sp = Spectrogram.load(a.spec)
        ev = load_event(a.event)
        traces = load_traces(a.traces)
        try:
            t_unix = float(a.utc)
        except ValueError:
            t_unix = datetime.fromisoformat(a.utc).astimezone(timezone.utc).timestamp()
        elev = None if a.no_dem else Terrain().elevation
        grid = Grid.make(*a.bbox, a.spacing, elev)
        res, segs = pipeline.run_single_search(sp, ev, traces, grid, t_unix, tol_s=a.tol, harmonics=a.harmonics,
                                               v_tol=a.v_tol, d_tol=tuple(a.d_tol), score_mode=a.score)
        res["lat0"] = grid.lat0; res["lon0"] = grid.lon0
        res.to_csv(a.out, index=False)
        print(res.head(20).to_string())


if __name__ == "__main__":
    main()
