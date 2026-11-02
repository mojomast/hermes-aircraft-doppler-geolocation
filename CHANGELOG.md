# Changelog

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
