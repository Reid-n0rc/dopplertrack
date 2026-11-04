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
Searched nationally, with no box at all, the Oyster Bay recording ranks the
true pair of flights first of 9,669 over the whole continental US, 32 m from
the true location, in about half an hour on a 16-thread desktop.
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
   locked to the flight geometry. `--score ridge` scores how closely the
   modelled curve follows the ridge traced in the window instead of the
   brightness under it (see "Ridge score" below); `--v-tol` and `--d-tol`
   open the speed and distance gates when the fit of a faint or turning
   pass cannot be trusted.
   `search` grids a box, which cannot scale past a region. `index-day` +
   `search-wide` run the same test over a whole day of a continent without
   a grid: the flights are paired first (two events heard Δt apart at slant
   distances d1 and d2 mean the aircraft were within d1 + d2 of each other
   Δt apart, a cheap join over every trajectory), then ground points are laid
   only in each pair's small lens, tested with the same exact CPA values,
   gates and silence filter, and scored with the same locked scorer. The
   candidate table is cached per event set, so a scorer change reruns only
   the scoring. `--extra` then rescores the best pairs with further
   flyovers, each matched to a flight at the pair's best point and scored
   at all of its points on the same clock, as `refine --extra` does.
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
# the same search over the whole continental US, no box: one index per day, reused by every search
uv run dopplertrack fetch-day 2026-09-08 traces_conus.npz --bbox 24 50 -125 -66
uv run dopplertrack index-day traces_conus.npz index_2026-09-08.npz
uv run dopplertrack search-wide rec.npz --events ev1.json ev2.json --index index_2026-09-08.npz \
    --cands cands.npz --harmonics 2 1 --out wide.csv --extra ev3.json ev4.json   # extras rescore the top pairs
# one faint flyover whose heard-CPA time is known (here 20:40:58 UTC), gates opened wide
uv run dopplertrack fit rec.npz --window 1375 1465 --band 70 105 --harmonics 2 --out ev_faint.json
uv run dopplertrack search-single rec.npz --event ev_faint.json --traces traces_ny.npz --utc 1788900058.6 \
    --bbox 40.4 41.3 -74.4 -72.6 --harmonics 2 --v-tol 1.0 --d-tol 0.05 20 --out single.csv
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

### A faint single flyover with a known time

The video's last experiment locates the recording from one faint one-minute
curve, given its time. The Oyster Bay hour has such a curve at 1375–1465 s:
a fundamental sweeping 92 to 73 Hz a few dB above the noise, with its second
harmonic ([`docs/results/oyster_bay_ev_faint.json`](docs/results/oyster_bay_ev_faint.json)).
The only ADS-B flight passing the true point then is N7681Y, a Piper Twin
Comanche, whose heard CPA at the true point is 0.9 s from the fitted one.
Its curve modelled from the real trajectory sits on the faint ridge
([figure](docs/results/oyster_bay_faint_curve.png)), but the straight-pass
fit of that ridge says 41 m/s at 750 m where the aircraft was doing 70 m/s
at 8.2 km: a sharp drop followed by a flat tail reads as a close slow pass,
and the search's default gates (±35 % in speed, 0.4–2.5× in distance) drop
the true flight before scoring. With the gates opened (`--v-tol 1.0
--d-tol 0.05 20`), `search-single` over the same 100 x 150 km box and ±30 s
scores 77,280 (point, flight) pairs from 66 flights:

| score | top point | margin over the next flight | top-20 points on N7681Y | of those within 2 km |
|---|---|---|---|---|
| brightness | 1.7 km from the truth, N7681Y | 31.6 % | 20 | 17 |
| ridge | 2.3 km from the truth, N7681Y | 14.5 % | 20 | 3 |

Tables: [`oyster_bay_single_faint_brightness.csv`](docs/results/oyster_bay_single_faint_brightness.csv),
[`oyster_bay_single_faint_ridge.csv`](docs/results/oyster_bay_single_faint_ridge.csv).
Two lessons. A single faint flyover with a known time does locate the
recording to a couple of kilometres in a box this size, as the video says.
And the ideal-curve fit is the weak link: on a pass that is not straight it
is wrong by a factor of ten in distance, so the gates it seeds must be loose
or, better, replaced by the trajectory-modelled curve itself.

**Ridge score.** For that faint curve the video scores candidates by their
distance from a hand-traced line rather than by brightness. `--score ridge`
is the automatic version: a Viterbi ridge is traced through the event
window once (harmonics are multiples of it), frames are kept only where the
ridge is bright and at least 3 dB brighter than the best path that avoids
it (a bird burst or a second tone offers several equally good paths and says
nothing), and a modelled curve scores the fraction of kept frames it passes
within 1.5 % in frequency of ([tracks and masks](docs/results/ridge_tracks.png)).
On the two blind two-event searches it is not an improvement over
brightness, and it is not the default:

| recording | brightness: true pair rank, margin | ridge: rank, margin |
|---|---|---|
| Oyster Bay (328 pairs) | 1, 4.9 % | 1, 4.0 % |
| eagle cam (607 pairs) | 1, 2.5 % | 4 (1 without the uniqueness mask, 0.6 %) |

Widening the tolerance to 3 % or 5 % or dropping the mask moves the eagle
cam between rank 1 with no margin and rank 4; the pair it loses to is the
one brightness ranks second, two aircraft on the same airway whose curves
differ little. Best-per-pair tables under both scores:
[`oyster_bay_search_best_per_pair_ridge.csv`](docs/results/oyster_bay_search_best_per_pair_ridge.csv),
[`swfl_search_best_per_pair_ridge.csv`](docs/results/swfl_search_best_per_pair_ridge.csv).
What the ridge score does give is a score in [0, 1] with a meaning (the
share of the measured curve the candidate explains) and a per-frame
residual, which is the raw material for an uncertainty estimate.

### Nationwide search

The regional searches above only work because a box was drawn around the
answer. `search-wide` drops the box: one day of prop flights over the whole
continental US (35,427 flights, 149 million one-second positions after
resampling) is indexed once, in one Earth-centred frame so slant distances
are exact anywhere, and the Oyster Bay events are searched against all of
it with the default gates.

| stage | Oyster Bay day, 16 threads (Ryzen 5800X) |
|---|---|
| `fetch-day`, prop flights over CONUS | 2.5 min at 20 MB/s |
| `index-day` (1 Hz segments + 10 s samples, 6.2 GB) | 30 s, once per day of data |
| pair join (speed gates, distance + lag geometry) | 10 s → 66,637 flight pairs |
| lens lattices, exact CPA, gates, silence filter | 71 s → 2,479,072 points on 9,963 pairs |
| locked scoring | 16 min |

A scorer change reruns only the last row (`--cands` caches the candidate
table); a gate change reruns the last three.

Result: the true pair N22NY / N93CQ ranks **1 of 9,669** flight pairs, its
best 200 m lattice point **32 m** from the true location, with a 6.2 %
margin over the runner-up (a pair over Texas). The next pair within 100 km
of the truth ranks 3rd. The best point of every pair is in
[`docs/results/oyster_bay_wide_best_per_pair.csv`](docs/results/oyster_bay_wide_best_per_pair.csv)
(top 1,000 of 9,669). Adding the faint flyover from the single-event
experiment below as an extra (`--extra ev_faint.json --extra-harmonics 2`)
widens the margin to 24 %, and the flight it matches to that event at the
true point is N7681Y, the Twin Comanche found there by hand.

**The eagle cam, nationally.** The same search over 2024-12-28 (13,046 prop
flights; 102,941 pairs joined in 6 s, 1,060,324 points on 13,466 pairs in
29 s, scored in 4.5 min) does *not* locate the camera from its two detected
events alone: the true pair ranks 15 of 13,298, 972 m off, and the winner
pairs the right first flight with a different second flight 200 km away
near Tampa, with a 1.7 % margin
([`docs/results/swfl_wide_best_per_pair.csv`](docs/results/swfl_wide_best_per_pair.csv)).
That is the regional finding writ large: two high, nearly overhead passes
pin a location to about 2 km and a 2.5 % margin in a box, and a continent
offers many pairs that fit two such curves as well.

The extra flyovers that pin the eagle cam regionally do the same
nationally. `--extra` takes the three further detected flyovers that
`refine --extra` used, matches each to a flight at every top pair's best
point, and scores it at all of that pair's points on the same clock. On
the 200 best two-event pairs this takes one second, and the true pair
moves from 15th to **first, 627 m from the nest, with a 13 % margin**
([`docs/results/swfl_wide_extra_best_per_pair.csv`](docs/results/swfl_wide_extra_best_per_pair.csv)).
The rescoring only sees the pairs the two-event search ranked in the top
`--top-pairs` (200 by default), so the two events still have to put the
truth somewhere in that list; here they put it 15th.

Two things made this feasible at all. The join over trajectories replaces
the grid: a box the size of the country at 200 m would be 200 million
points times every flight, while the join touches only pairs whose
geometry allows the two events, and lays points in the lens where both
annuli overlap (about 250 points per surviving pair). And the scorer had to
be rebuilt for a process pool: its original form streamed several megabytes
of temporaries per candidate, which an Apple-silicon laptop hides and a
desktop's memory bus does not (16 workers ran 1.4 times faster than one).
Sampling the spectrogram window along time once per event, so each sweep
sample is two gathers from a 200 KB array, and walking the sweep in small
chunks made it twice as fast on one core and eleven times faster on 16.
The lens stage had the same disease (a dense point-by-track distance
matrix per pair) and the same cure: a KD-tree nearest-sample query gives
the identical CPA with no large temporaries, 12 minutes became 71 seconds.

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
  f0 sweeps of the locked scorer run as one array operation per candidate,
  on a window sampled along time once per event.
- **Continent-scale search by pairing flights, not by laying points along
  one flight.** The video lays points along every flight segment that
  matches event 1's speed, at the fitted slant distance, and tests those
  points for event 2 (73,000 points nationally, with hand-fitted curves and
  tight ranges). `search-wide` joins flight pairs on the geometry of both
  events first and lays points only in each pair's lens, which stays cheap
  under the wider gates an automatic fit needs; with tight gates it reduces
  to the video's method.
- **Single-event, known-time mode as a first-class command.** The video
  demonstrates locating from one faint one-minute curve when the recording
  time is known; `search-single` exposes that directly, and reproduces it on
  the Oyster Bay hour (see above).
- **Ridge scoring as an option.** The video scores its faint curve by
  distance from a hand-traced line; `--score ridge` traces the line itself,
  with a uniqueness mask instead of a person's judgement of where the line
  is real.

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
