import io
import tarfile
from pathlib import Path

from dopplertrack import fetch

FIX = Path(__file__).parent / "data" / "trace_full_a2f38b.json"


def _tar_bytes():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        raw = FIX.read_bytes()
        for name in ["traces/8b/trace_full_a2f38b.json", "traces/8b/trace_full_ffffff.json", "README"]:
            ti = tarfile.TarInfo(name); ti.size = len(raw); tf.addfile(ti, io.BytesIO(raw))
    return buf.getvalue()


def test_iter_tar_traces_filters_by_bbox():
    data = _tar_bytes()
    got = list(fetch.iter_tar_traces(io.BytesIO(data), (40.0, 41.5, -75.0, -72.0)))
    assert len(got) == 2 and all(t.icao == "a2f38b" for t in got)   # icao comes from the JSON, not the name
    assert list(fetch.iter_tar_traces(io.BytesIO(data), (0.0, 1.0, 0.0, 1.0))) == []


def test_chained_stream_resumes_with_range(monkeypatch):
    data = _tar_bytes()
    parts = [data[: len(data) // 2], data[len(data) // 2:]]
    calls = []

    class Resp:
        def __init__(self, body, status):
            self.status_code = status; self._body = body
        def iter_content(self, n):
            # drop the connection after the first chunk of every fresh (non-Range) open
            yield self._body[:n]
            if len(self._body) > n and not calls[-1][1]:
                raise fetch.requests.ConnectionError("dropped")
            for k in range(n, len(self._body), n):
                yield self._body[k:k + n]
        def close(self):
            pass

    def fake_get(url, headers=None, stream=False, timeout=0):
        i = int(url[-1]); start = int(headers["Range"][6:-1]) if headers and "Range" in headers else 0
        calls.append((i, start))
        return Resp(parts[i][start:], 206 if start else 200)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    s = fetch._ChainedHTTP(["http://x/0", "http://x/1"], chunk=1000)
    out = io.BufferedReader(s, 4096).read()
    assert out == data
    assert any(start > 0 for _, start in calls)          # a resume actually happened
    got = list(fetch.iter_tar_traces(io.BytesIO(out), (40.0, 41.5, -75.0, -72.0)))
    assert len(got) == 2


def test_release_asset_urls_orders_split_tar_by_name(monkeypatch):
    assets = [{"name": n, "browser_download_url": f"http://x/{n}"}
              for n in ["day.tar.ab", "day.tar.aa", "notes.txt"]]

    class Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"assets": assets}

    monkeypatch.setattr(fetch.requests, "get", lambda url, timeout=0: Resp())
    assert fetch.release_asset_urls("2026-09-08") == ["http://x/day.tar.aa", "http://x/day.tar.ab"]
