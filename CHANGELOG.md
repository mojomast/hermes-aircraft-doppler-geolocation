# Changelog

## 0.2.0 — unreleased

- New `dopplergeo` package, CLI (`dopplergeo detect|locate`), and Hermes tools `doppler_detect` / `doppler_locate`.
- Automatic detection: whitening, harmonic sum, CFAR, multi-ridge Viterbi, harmonic-family fit, segment linking, acceptance test.
- Data fetching with cache and `--offline`: adsb.lol daily releases (streamed crop), tar1090-db types (jet filter), Copernicus GLO-30 DEM, Open-Meteo temperature.
- Scoring against real track geometry, joint likelihood with silence penalty, coarse-to-fine posterior heatmap with credible regions, unknown-time handling via metadata and sliding offsets.
- Tests: synthetic turning tracks, harmonic crossings, AAC/Opus round-trips, noise negatives; network tests behind `DOPPLER_GEO_NETWORK_TESTS=1`.
- Plugin now makes network calls in `locate` (disclosed in README and SECURITY.md).

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
