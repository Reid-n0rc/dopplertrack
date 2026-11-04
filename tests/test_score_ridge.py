"""Ridge-residual scoring. In mode="ridge" an EventWindow extracts the ridge of the measured
curve once and scores a modelled curve by how closely it follows that ridge, so a faint but
clean curve scores like a bright one, and the score is a fraction of the window in [0, 1]."""
import numpy as np

from dopplertrack.audio import Spectrogram
from dopplertrack.score import EventWindow
from dopplertrack.tracks import EventFit


def _spec(level, noise=1.0, seed=0, blank_after=None):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 200, 0.25); f = np.arange(60, 260, 0.5)
    S = rng.normal(0, noise, (len(f), len(t))).astype(np.float32)
    ridge = 150.0 - 50.0 / (1 + np.exp(-(t - 100) / 8))        # sigmoid sweep, 150 -> 100 Hz
    for k, tk in enumerate(t):
        if blank_after is None or tk <= blank_after:
            j = int(np.argmin(np.abs(f - ridge[k])))
            S[j - 1:j + 2, k] += level * np.array([0.5, 1.0, 0.5], np.float32)
    fit = EventFit(t_cpa=100, t_cpa_emit=97, f0=1.0, v=80, d=1500, t_start=40, t_end=160, score=0,
                   f_pre=150, f_post=100)
    return Spectrogram(t, f, S), fit, t, ridge


def test_true_curve_scores_near_one_and_wrong_placements_do_not():
    sp, fit, t, ridge = _spec(10.0)
    w = EventWindow.from_fit(sp, fit, mode="ridge")
    s = w.curve_score(t, ridge)
    assert s > 0.9
    assert w.sweep_scores(t, ridge, [20.0], [1.0])[0, 0] < 0.5 * s      # 20 s off the sweep
    assert w.sweep_scores(t, ridge, [0.0], [1.05])[0, 0] < 0.3          # f0 5 % high, off everywhere


def test_faint_curve_scores_like_a_bright_one_in_ridge_mode_only():
    bright, faint = _spec(10.0), _spec(6.0)
    rb = EventWindow.from_fit(bright[0], bright[1], mode="ridge").curve_score(bright[2], bright[3])
    rf = EventWindow.from_fit(faint[0], faint[1], mode="ridge").curve_score(faint[2], faint[3])
    bb = EventWindow.from_fit(bright[0], bright[1]).curve_score(bright[2], bright[3])
    bf = EventWindow.from_fit(faint[0], faint[1]).curve_score(faint[2], faint[3])
    assert rf > 0.8 * rb
    assert bf < 0.75 * bb


def test_missing_half_of_the_ridge_halves_the_score():
    sp, fit, t, ridge = _spec(10.0, blank_after=100)
    half = EventWindow.from_fit(sp, fit, mode="ridge").curve_score(t, ridge)
    full = EventWindow.from_fit(*_spec(10.0)[:2], mode="ridge").curve_score(t, ridge)
    assert 0.35 * full < half < 0.65 * full


def test_sweep_matches_curve_score_in_ridge_mode():
    sp, fit, t, ridge = _spec(10.0)
    w = EventWindow.from_fit(sp, fit, mode="ridge")
    s = w.sweep_scores(t, ridge, [0.0, 3.0], [1.0, 1.01])
    assert np.isclose(s[0, 0], w.curve_score(t, ridge))
    assert np.isclose(s[0, 1], w.curve_score(t, ridge, f0_scale=1.01))
