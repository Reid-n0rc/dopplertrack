# dopplertrack

Acoustic geolocation from the Doppler curves of propeller aircraft.

Any outdoor (and many indoor) recordings contain faint, usually inaudible
tones from aircraft flying over: the propeller blade-pass fundamental and its
harmonics, between roughly 60 and 400 Hz. As the aircraft passes, the tone
sweeps down in frequency. The exact shape of that sweep depends on where the
listener was relative to the aircraft's real trajectory. With historical
ADS-B data for every flight that day, each candidate location on the ground
predicts a different curve, and the recording picks out the one that matches.

This is a reimplementation of the technique shown in colsto's video
["I discovered a sound that leaks your exact location"](https://www.youtube.com/watch?v=UytYrhYLOKI).
No code was released with that video; this project is an open, automatic
version of the method. See `docs/superpowers/specs/` for the design.

## Status

Proof of concept, validated on two recordings with known answers. On the
author's reference recording (Oyster Bay, NY, 2026-09-08) a blind search
over a 100 x 150 km region ranks the true location first, 43 m off after
refinement. On a 12-hour public eagle-cam stream (North Fort Myers, FL,
2024-12-28), with the flyovers found automatically, the search ranks the
right pair of flights first of 607 and places the camera within 600 m.
See Results below.

## Pipeline

1. `spectrogram` — decode any audio/video with ffmpeg, high-resolution STFT of
   the 40–450 Hz band, whitened per frequency bin so only *changing* tones stand out.
2. `detect` — find the flyovers without a person: slide a window over the
   whole recording, grid-search an ideal straight-pass Doppler curve
   (f0, speed, CPA slant distance, CPA time) in each, keep the local maxima,
   refine them and rank them. `fit` does the same for one window you choose.
   `detect` scores a sweep-weighted ridge contrast, which ignores bursts and
   steady tones but gives the flat ends of a curve little weight; `fit` scores
   the plain mean brightness under the curve, as the results below were fitted
   (`--objective contrast` switches it to the detector's score).
   The curve parameters only seed a search with coarse ranges.
3. `fetch-day` — stream one day's [adsb.lol](https://github.com/adsblol/globe_history_2026)
   archive straight from GitHub (4 GB, nothing written to disk) and keep the
   flights that touched a bounding box, dropping known jets using ICAO Doc
   8643 engine classes. `extract-region` does the same from an archive you
   already have on disk.
4. `search` — for every grid point (200 m) and every flight, compute the
   closest point of approach (CPA), its slant distance (using a DEM for the
   observer's elevation) and the *heard* CPA time (propagation delay). Keep
   points that hear an event-1-like pass, then exactly Δt later an
   event-2-like pass, with no other prop flight in between. Model the exact
   Doppler curve each survivor would hear from the real trajectory
   (emission-time radial velocity) and score it against the spectrogram.
   Event 1's placement absorbs measurement error; event 2's placement is
   locked to the flight geometry.
5. `refine` — 50 m grid around a winning candidate. Further detected
   flyovers can be added here: each is matched to a flight once, at the
   candidate, then scored at every fine point with the same time shift as
   event 1, so every event is locked to one clock.

## Usage

```bash
uv sync
uv run dopplertrack spectrogram recording.mp4 rec.npz
uv run dopplertrack detect rec.npz --band 60 300 --top 10 --out detected.json
uv run dopplertrack fit rec.npz --window 2090 2200 --band 95 200 --harmonics 2 --out ev1.json
uv run dopplertrack fit rec.npz --window 3020 3130 --band 95 195 --out ev2.json
uv run dopplertrack fetch-day 2026-09-08 traces_ny.npz --bbox 40.4 41.3 -74.4 -72.6
uv run dopplertrack search rec.npz --events ev1.json ev2.json --traces traces_ny.npz \
    --bbox 40.4 41.3 -74.4 -72.6 --spacing 200 --harmonics 2 1 --out results.csv
uv run dopplertrack refine rec.npz results.csv --events ev1.json ev2.json --traces traces_ny.npz \
    --bbox 40.4 41.3 -74.4 -72.6 --harmonics 2 1 --extra ev3.json ev4.json --out fine.csv
```

`search --point LAT LON` scores a single known location instead of a grid,
which is the quickest way to validate the data and physics on a recording
whose location you already know. `detect` prints a ranked table of
candidate flyovers; `--out` writes them as a JSON list whose entries can be
passed to `search --events` as they are.

## Results

Reference recording: colsto's raw one-hour clip
([youtu.be/4CxVxuMVkMc](https://youtu.be/4CxVxuMVkMc)), made 2026-09-08 in
Oyster Bay, NY at 40.876466, -73.524951, the answer given in the video. Only
the recording date and the search region were used as inputs; the clip's
start time (20:17:30 UTC) and the two aircraft fall out of the search.

**Known-point check.** Scoring the true location against every non-jet flight
of the day (1,016 traces touching the region) leaves exactly one pair that
passes the 918 s offset and silence filters: N22NY, a Pilatus PC-12 at 6.1 km,
heard CPA 20:53:25 UTC, and N93CQ, a King Air 350 at 3.6 km, 21:08:43 UTC.
The Doppler curves modelled from their real trajectories lie on the measured
ridges with no free parameters beyond a half-second time shift:

![Modelled curves at the true point](docs/results/oyster_bay_known_point.png)

**Blind search.** A 200 m grid over a 100 x 150 km box around New York
(379,758 points, 3,068 airborne flight segments) gives 86 million
(point, flight) closest approaches and 40,889 candidate (point, pair)
triples after the filters, from 328 distinct flight pairs. Scoring all of them
takes 19 minutes on a laptop.

| | |
|---|---|
| Rank of the true location's flight pair | 1 of 328 |
| Top candidate, 200 m grid | 123 m from truth |
| Refined peak, 50 m grid | 43 m from truth |
| Margin over the best wrong flight pair | 4.9 % (27.1 vs 25.8) |

![Blind search score map](docs/results/oyster_bay_search.png)

The result is right and the 50 m refinement shows a single clean peak. The
4.9 % margin over the runner-up pair (flights 70 km away whose geometry
happens to produce a similar 918 s offset) is thin, because the score only
measures brightness under the modelled curve, so a wrong pair can score well
wherever the spectrogram is busy. Numbers:
`docs/results/oyster_bay_search_best_per_pair.csv` (best score per flight
pair) and `docs/results/oyster_bay_refined_50m.csv`.

**Automatic event detection.** The two windows above were chosen by eye, as
in the video. Running `detect` over the whole hour with no hints returns the
same two flyovers as its top two candidates (CPA 2155.6 s and 3068.4 s,
scores 15.7 and 10.0; the third candidate scores 9.0). Its own fits of them
can't stand in for `fit` yet: its event 2 comes out with f0 11 % above what
the King Air's trajectory needs and the time between the two events 10 s
short, so the search's ±6 s and ±5 % filters reject the true pair at the true
location. Finding the flyovers needs no manual input; measuring them for the
search still uses the two `fit` windows.

### Second recording: a 12-hour eagle cam

To test on something the method was not developed on, we used a public
livestream archive with a published location: the Southwest Florida Eagle
Cam in North Fort Myers (nest LE-026B at 26.702359, -81.847766 in Lee
County's eagle-nest register), the 12-hour archive of 2024-12-28
([youtu.be/zs2o5Rhlp5w](https://youtu.be/zs2o5Rhlp5w)). It is a harder
recording: an outdoor microphone in wind, 12 hours long, and 10 km from a
busy flight-school airport, so prop aircraft pass every few minutes. The
day's ADS-B was streamed with `fetch-day` (557 non-jet flights touching a
130 x 130 km box, 4 GB read, nothing stored) and the flyovers were found
with `detect`: of its top 40 candidates, 20 coincide within 25 s with a
known prop flight passing within 12 km of the nest.

![Known-point check on the eagle cam](docs/results/swfl_known_point.png)

**Blind search** on two detected events 338 s apart (a Beech Baron at
2.5 km and a Pilatus PC-12 at 6 km, as it turns out) over the 130 x 130 km
box: 432,196 points, 1,233 flight segments, 50,682 candidate triples from
607 flight pairs, 8 minutes.

| | |
|---|---|
| Rank of the true flight pair | 1 of 607 |
| Margin over the best wrong pair | 2.5 % |
| Top candidate, 200 m grid, two events | 917 m from the nest |
| Refined peak, 50 m grid, two events | 1.7 km from the nest |
| Refined peak with three more detected flyovers | 592 m from the nest |

![Eagle cam search](docs/results/swfl_search.png)

The right flights are identified, but two events pin the location only to
about 2 km here, and the reason is geometry rather than noise: both
aircraft passed nearly overhead at 2.5 km and 5.7 km altitude, so the slant
distance that shapes the Doppler curve barely changes as the observer moves
a kilometre on the ground. At Oyster Bay the King Air passed 3.5 km to the
side at 1 km altitude, which is why that peak is sharp. Adding three more
flyovers found by `detect` (`refine --extra`), locked to the same clock,
brings the peak to 592 m. Lower, lateral passes are what pin a location,
so the more flyovers a long recording offers, the better. Numbers:
`docs/results/swfl_search_best_per_pair.csv`, `docs/results/swfl_refined_50m.csv`,
and the detector's candidates for both recordings in `docs/results/*_detected.json`.

**What this second recording changed.** The first version of the scorer
averaged brightness only over the part of the window a modelled curve
covered, so a flight that landed or left the data mid-pass was judged on its
bright half alone; on the eagle cam that put a pair 64 km away on top. The
score is now the brightness summed along the curve over the whole measured
window (uncovered parts count as zero), which fixed the ranking here and
left the Oyster Bay result unchanged. Scoring the curve against its
spectral neighbourhood (a contrastive score) was tried as well: it widens
the Oyster Bay margin but does not help the eagle cam, so it is not in.

## Changes from the method in the video

The physics and the search strategy are the author's. What differs:

- **Automatic curve measurement.** The video measures each Doppler curve by
  hand, sliding f0 / speed / distance / CPA-time until the overlay looks
  right, then choosing "generous ranges" by eye. `fit` does this without a
  person: the window is whitened per frequency bin so stationary tones
  (mains hum, machinery) vanish, a vectorised grid search over all four
  parameters maximises spectrogram brightness under the curve, and a
  Nelder-Mead pass refines it. The ideal-curve fit still only seeds coarse
  filters; final ranking uses real trajectories.
- **Harmonic-aware fitting and scoring.** When a flyover shows a second
  harmonic, fundamental and harmonic are scored together, which constrains
  the curve more tightly than one line.
- **Open, documented data path.** The video used a paid/scraped ADS-B feed
  and did not release code. Here every input is a public dataset with a
  loader: adsb.lol daily archives, ICAO Doc 8643 engine classes, AWS
  terrain tiles.
- **Search engineering.** A KD-tree corridor around each flight path limits
  CPA computation to grid points that could possibly hear it; distance math
  is chunked so multi-hour tracks do not exhaust memory; the time-shift and
  f0 sweeps of the locked scorer run as one array operation per candidate.
- **Single-event, known-time mode as a first-class command.** The video
  demonstrates locating from one faint one-minute curve when the recording
  time is known; `search-single` exposes that directly.

Kept unchanged on purpose: emission-time radial velocity (not
observation-time), propagation-delayed CPA times, slant distance with a DEM,
the inter-event offset filter, the silence filter, and locked two-event
scoring where event 1 absorbs measurement error and event 2 is placed purely
by flight geometry.

## Data sources (all free)

| Need | Source |
|---|---|
| Historical ADS-B | adsb.lol `globe_history_YYYY` GitHub releases (~4 GB/day) |
| Aircraft engine class | OpenSky mirror of ICAO Doc 8643 |
| Elevation | AWS Terrain Tiles (Terrarium) |

## Development

```bash
uv run pytest
```

## Releasing

Releases are driven by git tags. Pushing a tag `vX.Y.Z` that matches the
version in `pyproject.toml` runs `.github/workflows/release.yml`, which runs
the tests, builds the sdist and wheel, attaches them to a GitHub Release with
generated notes, and publishes to PyPI through trusted publishing (no API
token stored anywhere).

```bash
uv version 0.2.0            # bump pyproject.toml
git commit -am "Release 0.2.0"
git tag v0.2.0 && git push && git push --tags
```

One-time setup for PyPI: on pypi.org, add a *pending publisher* for project
`dopplertrack` with owner `mjshiggins`, repository `dopplertrack`, workflow
`release.yml`, environment `pypi`, and create a `pypi` environment in the
GitHub repository settings. Until that exists the `pypi` job fails and the
GitHub Release still succeeds.
