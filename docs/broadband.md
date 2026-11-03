# Broadband (ground-reflection) events

Status: 0.3.0, experimental and opt-in: `dopplergeo detect|locate --broadband`, `broadband: true` in the `doppler_detect` / `doppler_locate` tools, `detect_events(x, sr, broadband=True)` in Python, or `--det broadband=true` in `scripts/eval_batch.py`. Code: `dopplergeo/broadband.py`, plus the broadband branch in `dopplergeo/score.py`.

## Why

Most close jet passes in the field and livestream recordings have no usable tonal line, so the tonal detector rejects them ("constant tone", "no harmonic line", "degenerate CPA"). Their spectrograms do show a strong, time-varying interference pattern from the sound reflected off the ground (Lloyd's mirror). That pattern encodes the source elevation, and so the pass geometry.

## Model

A microphone at height h above flat reflecting ground hears the direct sound and a ground-reflected copy, as if from an image receiver at −h. When the source range is much larger than h, the path difference is Δ ≈ 2·h·sin θ, where θ is the source elevation, so the **multipath delay** is

    τ = Δ / c ≈ 2 h sin θ / c.

The received spectrum is multiplied by |1 + R·e^{−i2πfτ}|², so the log spectrum is periodic in f with period 1/τ (fringe spacing c/Δ). The power cepstrum therefore peaks at quefrency τ. Typical values: h = 1.5 m overhead gives τ ≈ 8.7 ms (fringes about 114 Hz apart). A balcony mount at 5–10 m gives 30–60 ms.

For a level, straight pass with CPA slant range S0 = √(H² + R²), speed v and CPA time t_c, sin θ(t) = H / √(S0² + v²(t − t_c)²), so

    τ(t) = τ_max / √(1 + ((t − t_c)/T)²),   τ_max = 2 h H / (c S0),   T = S0 / v.

With one microphone of unknown height, a pass gives t_c, T and τ_max (which equals 2h·sin θ_CPA / c). For a given ADS-B candidate and grid point, the whole curve h·sin θ(t) is fixed up to the scale h. The scorer uses the same "unknown scale" structure as the tonal rest frequency.

## Detector (`detect_broadband`)

1. **Cepstrogram** (`cepstrogram`): 0.5 s Hann frames, 0.25 s hop. The log power is taken in 150–2000 Hz, a 4th-order polynomial trend (the source spectrum) is removed, and the result is projected onto cos(2π f q) for q = 1.5–60 ms in 0.05 ms steps. Each frame is z-scored with median/MAD (CFAR-style).
2. **Tracking** (`track_segments`): Viterbi over quefrency with an off state, a bounded jump per hop (about 1.5 ms), birth cost 8, z threshold 4, at least 16 frames. Peaks get parabolic sub-bin refinement.
3. **Fit** (`fit_pass`): bounded least squares of the τ(t) model above.
4. **Gates.** Every one must pass, and the reasons for rejection go into `diag["rejected"]`:

| gate | value |
| --- | --- |
| track span | ≥ 4 s |
| median cepstral z | ≥ 5 |
| F-test vs a constant delay | p ≤ 1e-4 (steady tones and their rahmonics are flat in time) |
| delay contrast max/min of the fit | ≥ 1.15 |
| CPA inside the track | t_start ≤ t_c ≤ t_end |
| T = S0/v | 0.8–150 s (0.8 s ≈ a 70 m/s jet 60 m overhead) |
| relative rms residual | ≤ 0.08 |

Accepted tracks become `Event(kind="broadband", tau_max_s, T_s, t_cpa, ratio=τ/τ_max, sigma_ratio=rel rms)`. `f0_hz`, `speed_mps` and `cpa_slant_m` are NaN or 0, because one microphone does not separate S0 from v. In `detect_events(..., broadband=True)`, a broadband event within 15 s of a tonal event is dropped: tonal events win.

## Scorer

For a broadband event, `score.py` predicts sin θ(t) at the grid point from the real ADS-B track (emission time solved, DEM ground height), then fits the scale 1/sin θ_CPA within log-uniform bounds (1, 60) with no prior. Model sigma is `bb_sigma_model = 0.06`. Clock slack, robust residuals, marginalisation over aircraft and the CPA-time search are shared with tonal events. ADS-B jets are scored for broadband events only. Tonal events still drop jets unless `include_jets`, so tonal behaviour, tonal gates and silence scoring are unchanged.

## Synthetic test

`dopplergeo.synth.render_broadband(rec_start, dur, sr, observer, tracks, mic_h=1.5, refl=0.9, ...)` emits band-limited (80–1800 Hz) noise from each track. The direct path and the image-receiver path (−mic_h) are each rendered with correct emission times, so the audio carries both Doppler and the fringes. `tests/test_broadband.py`:

- A pass at 500 m altitude, 150 m to the side, 75 m/s, heard with mic heights 1.5 m and 5 m: exactly one event. τ_max is within 12 % of 2h·sin θ_CPA/c, T is within 20 % of S0/v (7.0 s), and t_c is within 3 s.
- 300 s of white noise: no events. A steady 110 Hz harmonic tone in noise: no events.
- Scorer: the true point beats points 2.0 and 2.5 km away.
- Tonal events ignore jets unless they are enabled.

Quick check, from a checkout (about 10 s, under 200 MB):

```bash
python3 - <<'PY'
import numpy as np
from dopplergeo import synth, detect
t = np.arange(0, 200, 1.0)
la, lo, al = synth.track(t, 53.44, -2.10, speed=75.0, heading_deg=60.0, alt_m=500.0,
                         through=(300.0, 0.0), t_through=100)
x = synth.render_broadband(0, 200, 4000, (53.44, -2.10, 0.0), [(t, la, lo, al)], mic_h=1.5)
ev, _ = detect.detect_events(x, 4000, broadband=True)
print([(e.kind, round(e.t_cpa, 1), round(e.tau_max_s * 1e3, 2), round(e.T_s, 2)) for e in ev])
PY
# -> [('broadband', 99.0, 8.38, 7.08)]   (true: tau_max 8.4 ms, T 7.0 s)
```

## Results and caveats

See the README ("Broadband events") and [evaluation.md](evaluation.md). On tune, two Manchester field recordings went from 10.5 km and no events to 0.19 km and 0.33 km. On held-out (18 clips, one frozen run), 2 runs were under 3 km against 1 of 14 tonal-only, with 11 solved against 8. The median of solved runs went from 12.1 to 13.2 km, and ads_0821b and syd regressed; see [evaluation.md](evaluation.md). The one gate change made after seeing real data was lowering T's floor from 2 s to 0.8 s and the span floor to 4 s, for very low passes in the AMS livestream, decided on tune only. Most clips still give no broadband events: the method needs hard ground, a close pass and a clear fringe pattern.

## References

- K. W. Lo, B. G. Ferguson, "Aircraft flight parameter estimation using acoustic multipath delays", IEEE TAES 39(1), 2003. doi:10.1109/TAES.2003.1188908
- B. G. Ferguson, K. W. Lo, B. G. Quinn, "Flight parameter identification from cepstrum tracks", IDC 2002. doi:10.1109/IDC.2002.995436
- K. W. Lo, S. W. Perry, B. G. Ferguson, "Aircraft flight parameter estimation using acoustical Lloyd's mirror effect", IEEE TAES 38(1), 2002. doi:10.1109/7.993235
- K. W. Lo, B. G. Ferguson, "Passive estimation of aircraft motion parameters using destructive interference between direct and ground-reflected sound waves", IDC 1999. doi:10.1109/IDC.1999.754147
- B. G. Ferguson, "Time-delay estimation techniques applied to the acoustic detection of jet aircraft transits", JASA 106(1):255, 1999.
- "Source parameter estimation of aero-acoustic emitters using non-linear least squares and conventional methods", IET Radar Sonar Navig., 2017. doi:10.1049/iet-rsn.2016.0147
- R. L. Streit, R. F. Barrett, "Frequency line tracking using hidden Markov models", IEEE TASSP 38(4), 1990.
