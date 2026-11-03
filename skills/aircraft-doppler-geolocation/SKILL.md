---
name: aircraft-doppler-geolocation
description: >-
  Geolocate an audio or video recording from faint aircraft flyover Doppler
  curves matched against ADS-B flight tracks. Use when the user wants to find
  where or when a clip was recorded from background plane sound, or to audit
  their own recording for location leakage.
version: "0.3.0"
author: mojomast
license: MIT
compatibility: >-
  Python 3.11+ with numpy, scipy, soundfile, matplotlib, and ffmpeg on PATH.
  Since 0.2.0 the `doppler_locate` tool / `dopplergeo locate` CLI fetches ADS-B (adsb.lol, ODbL), types, Copernicus DEM and Open-Meteo temperature itself (cached, `--offline` for cache only). `doppler_detect` and the scripts make no network calls.
metadata:
  hermes:
    tags: [OSINT, Geolocation, Audio, Doppler, ADS-B, Spectrogram, Forensics]
    category: research
---

# Aircraft Doppler Geolocation Skill

Locates where (and optionally when) an audio/video recording was made by finding faint
aircraft flyover Doppler curves in its spectrogram and matching them against real ADS-B
flight tracks. Method follows the colsto video "I discovered a sound that leaks your exact
location" (https://www.youtube.com/watch?v=UytYrhYLOKI). It is NOT mains-hum/ENF analysis,
and it needs either a propeller/engine tonal (audible or not) or, with the experimental
broadband mode, a jet pass whose ground reflection makes a clear fringe pattern.

## Fast path (0.2.0+)

1. `doppler_detect(recording)`: automatic flyover detection; inspect `detection.png`. Zero events means no usable tonal pass. By default the tools detect tonal events only. If the spectrogram shows broadband jet passes with ground-reflection fringes (evenly spaced bands that converge and widen around closest approach), rerun with `broadband: true` (see below); otherwise stop and say so.
2. `doppler_locate(recording, date_hint, region_hint="lat,lon,radius_km")`: needs a date (or file metadata) and a region. Returns best point, 68/95 % areas, matched aircraft, `heatmap.png`. If metadata contains GPS it is reported directly.
3. Read `best.confidence` first. `ok` needs at least 2 events that consistently match an aircraft (`best.n_consistent_events`). With `low`, the report carries a `warning` and `credible_area_km2["95_low_conf"]` (the whole search disc): report the search region, not the point. Only with `ok` is `credible_area_km2["95"]` a usable 95 % region, and even then it is roughly calibrated (on held-out clips the only `ok` result was 13 km off).
4. Report honestly: a broad posterior or one event is weak evidence. Use only on recordings the user has a right to analyse.

CLI equivalent: `python3 -m dopplergeo detect REC [--out DIR]` and `python3 -m dopplergeo locate REC --date D --region lat,lon,km [--time-tolerance S] [--offline] [--include-jets] [--broadband] [--temper-extra K] [--out DIR]`. Cache: `~/.cache/doppler-geo` or `$DOPPLER_GEO_CACHE`. Give an ISO start time with a small `--time-tolerance` (e.g. 30 s) whenever you know it; a date alone searches the whole UTC day.

## Broadband mode (0.3.0, experimental)

Jets often give no tonal line. `detect_events(x, sr, broadband=True)` also finds passes from the ground-reflection (Lloyd's mirror) multipath delay τ(t) = τ_max/√(1+((t−t_c)/T)²), T = S0/v, measured with a cepstrogram, Viterbi-tracked and gated (span ≥ 4 s, cepstral z ≥ 5, F-test vs constant, contrast ≥ 1.15, CPA inside, 0.8 ≤ T ≤ 150 s, rel rms ≤ 0.08). Broadband events have `kind="broadband"`, `tau_max_s`, `T_s`, and no speed or distance of their own. The scorer matches them to ADS-B jets as well as props. Enable it with `broadband: true` on `doppler_detect` / `doppler_locate`, or `--broadband` on the CLI. Results: on tune it fixed two Manchester field recordings (0.19 and 0.33 km). On held-out (18 clips) it gave 2 runs under 3 km against 1 of 14 without it, and more solved runs, but a slightly worse median (13.2 km against 12.1) and two regressions. So run it when the tonal pass finds nothing, and present broadband-based fixes as experimental. `temper_extra` (default 4) widens or narrows the 68/95 % areas; do not lower it to make a result look sharper. Method and maths: `docs/broadband.md` in the plugin repo.

## When to Use

- Forensic/OSINT geolocation of a recording with few visual clues, where you have a date
  (and ideally a time window or region) and a flyover tone is visible in the spectrogram.
- Auditing your OWN recordings for location leakage before publishing.
- Do not use to locate private individuals without consent or lawful basis.

## Prerequisites

- `ffmpeg`, Python 3.11+, `pip install numpy scipy soundfile matplotlib tifffile`. The manual path below needs your own ADS-B; the tools fetch it themselves.
- ADS-B history for the date/area (the video scraped a full day for the continental US):
  OpenSky Network historical data via `pyopensky` (https://github.com/junzis/pyopensky) or the
  `traffic` library (https://github.com/xoolive/traffic); community feeds such as
  https://adsb.lol or https://www.adsbexchange.com. Need per-aircraft time, lat, lon,
  altitude (MSL), and ICAO type code.
- An aircraft type lookup to drop known jet types (the video filtered out jets rather than
  filtering in props, because the prop table had gaps).
- A DEM for ground elevation (e.g. https://portal.opentopography.org) — ADS-B altitude is
  above sea level, so slant distance needs ground height.

## How to Run

Run via `terminal` from the skill directory:

```
python scripts/doppler_track.py rec.mp4 --fmin 60 --fmax 250 --t0 40 --t1 160 --out ev1
python scripts/score_candidates.py track.csv candidates.csv ev1.csv --f0 <fit f0> \
       --cpa-rec <fit cpa_receive_time_s> [--rec-start <track-clock time of rec t=0>] --out ranked.csv --plot scores.png
python scripts/make_synthetic_test.py   # self-test, no network, exit 0 = pass
```

## Quick Reference

| Output | Meaning |
|---|---|
| `ev1.csv` | time_s, freq_hz, power_db of the tracked ridge |
| `ev1.png` | band spectrogram with ridge + fitted curve (inspect with `vision_analyze`) |
| `ev1.json` | `f0_hz` (rest tone), `speed_mps`, `cpa_slant_m`, `cpa_receive_time_s`, stderr, `ridge_snr_z`, `detected`, `rejected_because` |
| `ranked.csv` | candidates sorted by RMS Hz misfit (lower = better) |
| `--plot scores.png` | heatmap of misfit, track, top ranks (`--true lat,lon` marks a known point) |

Physics: heard f = f0·c/(c − v_r), where v_r is the radial velocity at EMISSION time
τ and the sound arrives at t = τ + r(τ)/c. Frequency drop size ⇒ speed; steepness ⇒ CPA
slant distance. c ≈ 331.3·√(1+T/273.15) m/s.

## Procedure

1. Spectrogram: `doppler_track.py` decodes with ffmpeg to 4 kHz mono and tracks the ridge in
   a narrow band with fine (0.5 Hz) resolution. Zoom/boost; curves are often inaudible. Note
   every flyover's receive-CPA time, and also long stretches with NO flyover (silence is a constraint).
2. Check `detected` first. If it is false, read `rejected_because`, then narrow the band to one harmonic or the window to one pass, or treat it as no flyover. Real tones score about 2.7 to 3.6 on `ridge_snr_z`; noise scores below 2.5 (`--min-snr`). For steep passes close by, raise `--max-slope`.
   Measure each event: read `f0`, speed and CPA slant distance from the JSON. Real planes turn
   and change speed, so treat the fit as a generous range (e.g. ±20 m/s, ±30% distance).
3. Candidate flights: from ADS-B for the date, drop known jet type codes (if the tone sounds
   like a prop), keep segments whose ground speed is within the event's speed range.
4. Candidate points: lay a 200 m grid around each matching flight out to the measured slant
   distance band (≤10 km max audibility). Compute slant distance using DEM elevation, and the
   DELAYED CPA time (closest-approach time + slant distance / c).
5. Time-offset filter (2+ events): keep points that hear an event-1 flight then an event-2
   flight at the observed offset (±~5 s). Silence filter: drop points that would have heard
   another flight between events. Near airports many points survive; silence prunes them.
6. Model and score: for each surviving point (refine to 100 m) and its flight, model the exact
   received curve from real geometry with `score_candidates.py`. Sweep f0 and a few seconds of
   CPA timing for event 1 only; LOCK event 2 to its physics-derived offset from event 1 (run
   event 2 with `--rec-start` set from event 1's best alignment). Sum/compare both scores; take
   the best point per flight pair. With one event, a known recording time (`--rec-start`, small
   `--dt-tol`) is required; phone calls carry timestamps.
7. Confirm visually: open the top coordinate in maps/street view and match the scene. Report
   the top-k with scores, margin over 2nd place, and assumptions.

## Pitfalls

- Using radial velocity at receive time instead of emission time yields a wrong curve shape,
  not just a delay — the search silently fails.
- One flyover with unknown time has an along-track ambiguity, and a straight track has a
  left/right mirror ambiguity; turns, a second event, or a known time break these.
- ADS-B altitude is barometric/MSL; use geometric altitude when available plus DEM.
- Raw 1 Hz ADS-B positions jitter by tens of metres, which `np.gradient` turns into several Hz of modelled Doppler. Pass `--smooth-s 10` to `score_candidates.py` for jittery feeds (centred odd window; ends extrapolated so straight tracks are unchanged).
- 200 m grids can put the true point ~141 m off and score poorly — refine to 100 m.
- Harmonics that cross each other make the ridge hop between tones. Use a band holding a single harmonic. A turning plane fails the straight-pass fit. Both are expected rejections.
- Wind/temperature shift c and arrival times; MP3/AAC can hide but not remove low tones.
- Livestream audio (host commentary, gated ATC radio, high-passed at 80–120 Hz, compressed) usually gives no events in either mode. Do not loosen gates to force a result.
- Do not report the best point of a `confidence: low` result as the location.

## Verification

`scripts/make_synthetic_test.py` synthesizes a curved 80 m/s flyover at 110 Hz buried in noise
(CPA tone amplitude 0.29 vs noise σ 1.0), recovers f0 = 110.0 Hz, speed 80.4 m/s, slant 1039 m
(true 1040 m), then with known start time ranks the true grid point first (0 m error). Without
time it lands on the right range but ~1.4 km along-track — the documented ambiguity.

## Limitations and Ethics

- Needs a date, ADS-B coverage (sparse in remote areas, gaps for non-transponder aircraft),
  heavy compute for large areas (the author reported months of work, 5,000+ lines, 50 GB files).
- Results are probabilistic; state confidence as score margin and require independent visual
  confirmation. Never present as proof by itself.
- Consent/privacy: use for forensics, journalism, research, or self-audit only; locating a person
  can endanger them. Follow local law.
- Protecting your own recordings (from the video): a high-pass filter does not reliably work
  (tones sit in the voice band and resurfaced after MP3 export + gain); talking over it does not
  drown it out; phone compression/noise reduction can preserve it. What helps: cutting segments
  (breaks the inter-event offset), stronger noise cancellation, removing timestamps/metadata, and
  checking your own spectrogram with `doppler_track.py` before posting.
