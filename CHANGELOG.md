# Changelog

## 0.3.0 — unreleased (feat/broadband)

- Broadband (jet) pass detection from the ground-reflection (Lloyd's mirror) multipath delay: band-limited cepstrogram, Viterbi quefrency track, straight-pass fit τ(t) = τ_max/√(1+((t−t_c)/T)²), and gates (`dopplergeo/broadband.py`). Opt-in with `detect_events(..., broadband=True)` or `eval_batch.py --det broadband=true`; not yet a CLI or tool option.
- `Event.kind` (`tonal` / `broadband`), `tau_max_s`, `T_s`. A broadband event within 15 s of a tonal event is dropped.
- Scorer: broadband events match predicted sin(elevation) with the unknown mic height as a scale (`bb_scale_bounds=(1, 60)`, `bb_sigma_model=0.06`). ADS-B jets are scored for broadband events only; tonal scoring is unchanged.
- `synth.render_broadband` (direct + image-receiver path) and `tests/test_broadband.py`.
- Gate change made on tune data: T lower bound 0.8 s and minimum span 4 s (very low passes).
- Docs: README restructured (CLI flags, tool parameters with example output, how to read 95 % regions, data and eval licences, harness), new `docs/broadband.md` and `docs/evaluation.md`. Tune results: man_hg0903 10.5 → 0.19 km, man_hg0909 none → 0.33 km. Held-out results pending.

## 0.2.0 — unreleased

- New `dopplergeo` package, CLI (`dopplergeo detect|locate`), and Hermes tools `doppler_detect` / `doppler_locate`.
- Automatic detection: whitening, harmonic sum, CFAR, multi-ridge Viterbi, harmonic-family fit, segment linking, acceptance test.
- Data fetching with cache and `--offline`: adsb.lol daily releases (streamed crop), tar1090-db types (jet filter), Copernicus GLO-30 DEM, Open-Meteo temperature.
- Scoring against real track geometry, joint likelihood with silence penalty, coarse-to-fine posterior heatmap with credible regions, unknown-time handling via metadata and sliding offsets.
- Tests: synthetic turning tracks, harmonic crossings, AAC/Opus round-trips, noise negatives; network tests behind `DOPPLER_GEO_NETWORK_TESTS=1`.
- Plugin now makes network calls in `locate` (disclosed in README and SECURITY.md).
- `best.confidence` (`ok` / `low`) and `best.n_consistent_events`; low-confidence results add a `warning` and report the whole search disc as their 95 % area.
- Calibration: `score.Options.temper_extra` (default 4) multiplies the Birge temperature so 95 % areas are not overconfident. New defaults `sigma_model=0.008`, `detect_prob=0.15`; sub-harmonic f0 relabel; CPA-distance consistency term.
- Atomic cache writes (temp file + `os.replace`) for DEM tiles, weather, type table and ADS-B crops.
- Real-world evaluation harness `scripts/eval_batch.py` (manifest-driven, ground-truth error, credible-area coverage) and results in the README.

## 0.1.1 — 2026-11-02

- `doppler_track.py` now gates its fit. The JSON gains `ridge_snr_z`, `detected`, and `rejected_because`; before this, the optimiser returned a plausible-looking fit even on noise or broadband jet roar.
- New options `--min-snr` (default 2.5) and `--max-slope` (Hz/s, default 8) for the ridge tracker.
- The fitter tries several closest-approach start times, not just one.
- New test: pure noise is not reported as a flyover (6 tests).
- README: real-world test on YouTube audio, with two real-world screenshots.

## 0.1.0 — 2026-11-02

- First public release of the aircraft-doppler-geolocation Hermes plugin.
- Skill `aircraft-doppler-geolocation`: find flyover Doppler curves in a spectrogram, fit speed / closest-approach distance, and score candidate locations against ADS-B tracks with emission-time-corrected modelling.
- Offline synthetic self-test and unit tests. No network calls, credentials, or telemetry.
