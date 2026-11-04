"""Read adsb.lol / readsb `globe_history` daily archives and classify aircraft."""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from .geo import lla_to_enu

FT = 0.3048
KT = 0.514444
TRACE_PARSE_ERRORS = (OSError, ValueError, IndexError, KeyError, TypeError, zlib.error, EOFError)   # KeyError: a trace row that is a dict
_DOC8643_URL = "https://opensky-network.org/datasets/metadata/doc8643AircraftTypes.csv"


def _cache_dir() -> Path:
    p = Path(os.environ.get("DOPPLERTRACK_CACHE", Path.home() / ".cache" / "dopplertrack"))
    p.mkdir(parents=True, exist_ok=True)
    return p


class AircraftTypes:
    """ICAO Doc 8643 designator -> engine class ('Piston', 'Turboprop', 'Jet', ...)."""

    def __init__(self, path: str | Path | None = None):
        p = Path(path) if path else _cache_dir() / "doc8643AircraftTypes.csv"
        if not p.exists():
            p.write_bytes(requests.get(_DOC8643_URL, timeout=120).content)
        self.types: dict[str, dict] = {}
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.types.setdefault(row["Designator"], row)

    def engine(self, designator: str | None) -> str | None:
        r = self.types.get(designator or "")
        return r["EngineType"] if r else None

    def is_prop(self, designator: str | None, unknown_ok: bool = True) -> bool:
        e = self.engine(designator)
        if e is None:
            return unknown_ok
        return any(k in e for k in ("Piston", "Turboprop", "Electric"))


@dataclass
class Trace:
    icao: str
    reg: str | None
    typ: str | None
    desc: str | None
    t: np.ndarray        # unix seconds
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray      # metres MSL (geometric when available, else barometric)
    gs: np.ndarray       # m/s (nan if missing)
    on_ground: np.ndarray

    def __len__(self):
        return len(self.t)

    def segments(self, max_gap_s: float = 90.0) -> list[slice]:
        """Index slices of contiguous airborne data."""
        out = []
        if len(self.t) == 0:
            return out
        brk = np.where((np.diff(self.t) > max_gap_s) | (self.on_ground[1:] != self.on_ground[:-1]))[0] + 1
        starts = np.r_[0, brk]; ends = np.r_[brk, len(self.t)]
        for s, e in zip(starts, ends):
            if e - s >= 2 and not self.on_ground[s]:
                out.append(slice(s, e))
        return out

    def resample(self, sl: slice, dt: float = 1.0):
        """Uniformly resampled (t, lat, lon, alt) over one segment."""
        t = self.t[sl]
        tq = np.arange(t[0], t[-1], dt)
        return (tq, np.interp(tq, t, self.lat[sl]), np.interp(tq, t, self.lon[sl]), np.interp(tq, t, self.alt[sl]))

    def enu(self, sl: slice, lat0: float, lon0: float, dt: float = 1.0):
        tq, la, lo, al = self.resample(sl, dt)
        return tq, lla_to_enu(la, lo, al, lat0, lon0)


def parse_trace_file(path: str | Path) -> Trace | None:
    return parse_trace_bytes(Path(path).read_bytes(), Path(path).stem.split("_")[-1])


def parse_trace_bytes(raw: bytes, fallback_icao: str = "") -> Trace | None:
    """Parse one readsb trace_full JSON document (gzipped or not)."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    d = json.loads(raw)
    if not isinstance(d, dict):
        return None
    base = float(d.get("timestamp", 0.0))
    rows = d.get("trace") or []
    t, lat, lon, alt, gs, grd = [], [], [], [], [], []
    for r in rows:
        if r[1] is None or r[2] is None:
            continue
        a = r[3]
        on_ground = a == "ground"
        geom = r[10] if len(r) > 10 else None
        if isinstance(geom, (int, float)):
            am = geom * FT
        elif isinstance(a, (int, float)):
            am = a * FT
        elif on_ground:
            am = np.nan
        else:
            continue
        t.append(base + r[0]); lat.append(r[1]); lon.append(r[2]); alt.append(am)
        gs.append(r[4] * KT if isinstance(r[4], (int, float)) else np.nan); grd.append(on_ground)
    if not t:
        return None
    return Trace(d.get("icao", fallback_icao), d.get("r"), d.get("t"), d.get("desc"),
                 np.array(t), np.array(lat), np.array(lon), np.array(alt), np.array(gs), np.array(grd))


def _scan_one(args):
    path, bbox = args
    try:
        tr = parse_trace_file(path)
    except TRACE_PARSE_ERRORS:
        return None
    if tr is None:
        return None
    lat0, lat1, lon0, lon1 = bbox
    m = (tr.lat >= lat0) & (tr.lat <= lat1) & (tr.lon >= lon0) & (tr.lon <= lon1)
    if not m.any():
        return None
    return tr


def load_region(archive_dir: str | Path, bbox: tuple[float, float, float, float],
                workers: int | None = None) -> list[Trace]:
    """All traces with at least one point inside bbox=(lat_min, lat_max, lon_min, lon_max)."""
    files = sorted(Path(archive_dir).glob("traces/*/trace_full_*.json"))
    out = []
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for tr in ex.map(_scan_one, ((str(f), bbox) for f in files), chunksize=256):
            if tr is not None:
                out.append(tr)
    return out


def in_bbox(tr: Trace, bbox) -> bool:
    lat0, lat1, lon0, lon1 = bbox
    return bool(((tr.lat >= lat0) & (tr.lat <= lat1) & (tr.lon >= lon0) & (tr.lon <= lon1)).any())


def save_traces(traces: list[Trace], path: str | Path):
    np.savez_compressed(path, n=len(traces),
                        meta=np.array([json.dumps([t.icao, t.reg, t.typ, t.desc]) for t in traces]),
                        **{f"a{i}": np.stack([t.t, t.lat, t.lon, t.alt, t.gs, t.on_ground.astype(float)]) for i, t in enumerate(traces)})


def load_traces(path: str | Path) -> list[Trace]:
    z = np.load(path, allow_pickle=False)
    meta = z["meta"]                      # read once: indexing z[...] decompresses the member each time
    out = []
    for i in range(int(z["n"])):
        icao, reg, typ, desc = json.loads(str(meta[i]))
        a = z[f"a{i}"]
        out.append(Trace(icao, reg, typ, desc, a[0], a[1], a[2], a[3], a[4], a[5].astype(bool)))
    return out
