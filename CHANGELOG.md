# Changelog

## Unreleased

- CI: GitHub Actions workflow `checks` (pull requests and pushes to `main`): FFmpeg from apt, `pip install -e ".[dev]"`, `ruff check`, `ruff format --check`, offline test suite (network tests stay off). `pytest` and `ruff` (pinned 0.17.3) added to the `dev` extra; local commands in README "Development / checks".

Efficiency pass; the changes below only remove repeated work. On the 7 tune clips checked (see PR #5), events, every search level's log-likelihoods, start times, matches, best point, credible areas and coverage are bit-identical to 0.3.0 (max observed difference 0). An earlier revision of this branch differed on man_hg0903 (events up to ~3e-5 relative): that came from the vectorised broadband Viterbi accumulating float32 scores in float32, not from the detrend, and is fixed (float64 accumulation as in 0.3.0).

- Event cache shared by `detect`, `locate` and the eval harness (`dopplergeo/evcache.py`): keyed by audio content hash, decode rate, detector settings and detector source hash; atomic writes, corrupt or mismatched entries are misses, LRU pruning. Changing region, date, tolerance or scorer options no longer repeats detection. `--no-cache` / `cache: false` to bypass. Reports carry `detection_cache.hit`.
- `--no-plots` (CLI) / `plots: false` (tools) / `plots=False` (Python) skips `detection.png` and `heatmap.png`. Default unchanged (plots on).
- `locate` validates region and date before decoding audio. A missing hint now returns its error without `events` in the report.
- Scoring: event arrays prepared once per search and option-dependent values once per level; one shared pass-geometry slice per (point, pass) for the span prefilter and every event's prediction; plane velocities from one full-track gradient (slice ends patched to match `np.gradient`); `find_passes` evaluates only track samples near each chunk and builds slant ranges without a 3-D temporary; row-wise nanmedian without NumPy's masked-array path.
- Detection: Viterbi jump search vectorised (tonal and broadband, same tie-breaking); masked score updated in place; frame-major backpointers. Broadband polynomial detrend uses the (nf × 5) factors instead of a dense nf × nf projection (isolated 256 × 876 benchmark: ~3.6× faster than the precomputed dense projection, max diff 2.4e-15). The vectorised broadband Viterbi accumulates in float64 as 0.3.0 did.
- DEM: parsed tiles and decoded blocks reused across `elevation()` calls (bounded LRU, 4 tiles × 16 blocks). ADS-B: cached crops stream-filtered line by line; crop headers memoised.
- Event-cache entries are validated in full (payload and every event); malformed entries are recomputed.
- Eval harness: events from the event cache (old per-run pickles were keyed on the audio path); `run_key` per row, checked on `--resume`; `--jobs 0` (new default) sizes workers from available memory (`--mem-per-job-gb`, default 2); spawned workers get `OMP/OPENBLAS/MKL_NUM_THREADS = cpu // jobs`.
- `scripts/profile_clip.py`: stage timings, peak RSS and a result pickle for before/after comparisons.
- Benchmark limitations: equivalence was checked on the 7 tune clips only; held-out equivalence is unverified (held-out was not re-run, per the evaluation policy). The warm-cache timings in PR #5 were measured before the stricter event-cache validation was added.
- Skill script `score_candidates.py`: new opt-in `--smooth-s` option, a centred moving average on ADS-B positions before taking velocities, so 1 Hz position jitter no longer turns into Doppler noise. Contributed by @Bortlesboat in [#3](https://github.com/mojomast/hermes-aircraft-doppler-geolocation/pull/3); centred-window and track-end fixes in [#6](https://github.com/mojomast/hermes-aircraft-doppler-geolocation/pull/6). Off by default (`--smooth-s 0`) and confined to the standalone skill script; `dopplergeo/score.py` is unchanged.

## 0.3.0 — 2026-11-03

- Broadband (jet) pass detection from the ground-reflection (Lloyd's mirror) multipath delay: band-limited cepstrogram, Viterbi quefrency track, straight-pass fit τ(t) = τ_max/√(1+((t−t_c)/T)²), and gates (`dopplergeo/broadband.py`). Opt-in with `detect_events(..., broadband=True)` or `eval_batch.py --det broadband=true`; enabled from the CLI with `--broadband` (detect and locate) and from the tools with `broadband: true`.
- CLI `locate --temper-extra K` and tool parameter `temper_extra` (95 % area calibration, default 4).
- `--offline` "no cached ADS-B crop" error prints plain lat/lon numbers, not `np.float64(...)`. `detection.png` now draws broadband events (it crashed on them).
- `Event.kind` (`tonal` / `broadband`), `tau_max_s`, `T_s`. A broadband event within 15 s of a tonal event is dropped.
- Scorer: broadband events match predicted sin(elevation) with the unknown mic height as a scale (`bb_scale_bounds=(1, 60)`, `bb_sigma_model=0.06`). ADS-B jets are scored for broadband events only; tonal scoring is unchanged.
- `synth.render_broadband` (direct + image-receiver path) and `tests/test_broadband.py`.
- Gate change made on tune data: T lower bound 0.8 s and minimum span 4 s (very low passes).
- Docs: README restructured (CLI flags, tool parameters with example output, how to read 95 % regions, data and eval licences, harness), new `docs/broadband.md` and `docs/evaluation.md`. Tune results: man_hg0903 10.5 → 0.19 km, man_hg0909 none → 0.33 km. Held-out (18 clips, one frozen run): 2/18 under 3 km (dfw 1.96, lgb 2.38) vs 1/14 tonal-only; 11 solved vs 8; median of solved 13.2 km vs 12.1; regressions ads_0821b 7.9 → 17.2 km and syd 11.0 → 18.6 km.

## 0.2.0 — unreleased (folded into 0.3.0)

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
