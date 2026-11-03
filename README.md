# aircraft-doppler-geolocation

Hermes plugin that locates where (and optionally when) a recording was made from the Doppler curves of aircraft flying over, often too faint to hear, matched against ADS-B flight tracks. It ships one skill, two Hermes tools (`doppler_detect`, `doppler_locate`), a `dopplergeo` CLI, and three standalone scripts.

It is a community plugin by [Kyle Durepos](https://github.com/mojomast). It is not a Nous Research product. The method comes from colsto's video [I discovered a sound that leaks your exact location (and it's everywhere)](https://www.youtube.com/watch?v=UytYrhYLOKI). The author released no code; these scripts are an independent implementation of the steps he describes, not his pipeline.

![Catalog banner](docs/catalog-banner.png)

## New in 0.2.0: automatic pipeline

```bash
dopplergeo detect clip.wav --out ev/                          # find flyovers automatically
dopplergeo locate clip.wav --date 2026-09-21 --region 43.8,-79.3,8 --out run/
dopplergeo locate clip.wav --date 2026-09-21 --region 43.8,-79.3,8 --offline   # cache only
```

Python / Hermes tool: `locate(recording, date_hint=None, region_hint=None)`.

- **Detection** (no hand-picked bands): whitened 1 s spectrogram, harmonic sum (k ≤ 6), CFAR, Viterbi multi-ridge tracking with an off state, a joint harmonic-family fit, segment linking across dropouts, and an acceptance test (F-test against a constant tone, frequency drop, plausible speed, closest approach inside the tone, non-degenerate distance).
- **Data**: ADS-B from adsb.lol daily `globe_history` releases, streamed and cropped to your region and time window; only the crop is cached (`~/.cache/doppler-geo` or `$DOPPLER_GEO_CACHE`). Aircraft types from tar1090-db (jets dropped by default, since they rarely give a tonal line). Copernicus GLO-30 elevation (flat-terrain fallback). Open-Meteo temperature sets the speed of sound.
- **Scoring**: predicts each event's curve from the real track (turns included), with emission-time correction, a robust f0-scale fit, joint likelihood across events, and a penalty for close passes that were not heard. Coarse-to-fine grid (1000/200/50 m) gives a posterior heatmap with 68 %/95 % regions. Unknown recording time: file metadata (ffprobe creation_time, GPS tags, yt-dlp `.info.json`) or sliding offsets.

![Synthetic heatmap](docs/screenshots/v02-heatmap-synthetic.png)
![Automatic detection, real Cessna 421 clip](docs/screenshots/v02-auto-detect-cessna421.png)

**Real-clip detection (0.2.0, automatic):** Cessna 421 clip: 1 event (49 m/s, 159 m CPA). "Planes overhead / birds" clip: 2 events (57 m/s at 1.4 km; 67 m/s at 1.1 km). Amsterdam city ambience, a 23 s jet clip, and the Conquest tower flyby: 0 events (expected: no clean tonal pass). An end-to-end real location with a known place and time has **not** been verified yet; the full pipeline is verified on synthetic data only (0 m grid error, start time within 0.3 s).

### Data credits

- ADS-B: [adsb.lol](https://adsb.lol), Open Database License (ODbL 1.0). Derived crops stay under ODbL.
- Aircraft types: [wiedehopf/tar1090-db](https://github.com/wiedehopf/tar1090-db).
- Elevation: Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
- Weather: [Open-Meteo](https://open-meteo.com) (CC BY 4.0).

## What you get

One skill, `aircraft-doppler-geolocation:aircraft-doppler-geolocation`, and three scripts in `skills/aircraft-doppler-geolocation/scripts/`:

| Script | Job |
| --- | --- |
| `doppler_track.py` | Decode audio/video with ffmpeg, trace the flyover tone on a narrow-band spectrogram (Viterbi ridge), and fit base tone, speed, closest-approach slant distance, and closest-approach time. Writes CSV, PNG, JSON with a `detected` flag, `ridge_snr_z`, and `rejected_because`. Options `--min-snr` and `--max-slope`. |
| `score_candidates.py` | For a real flight track (ADS-B CSV) and candidate points (lat, lon, ground elevation), model the exact curve each point would hear using the velocity at emission time, and rank points by misfit in Hz. `--plot` writes a heatmap. |
| `make_synthetic_test.py` | Offline self-test: synthetic curved flyover buried in noise, fit, then locate the observer on a grid. |

## Disclosure

- `register()` calls `ctx.register_skill` and `ctx.register_tool` for `doppler_detect` and `doppler_locate`. No hooks or middleware.
- `doppler_track.py` runs `ffmpeg` (no shell) on the path you pass. Scripts read only the paths you pass and write the outputs you name.
- **Network (0.2.0, `locate` only, skipped with `--offline`)**: HTTPS GET to api.github.com / github.com (adsb.lol release assets, streamed; up to ~3 GB read per day, only the crop is stored), raw.githubusercontent.com (type table), copernicus-dem-30m.s3.amazonaws.com (DEM tiles), archive-api.open-meteo.com (temperature). No credentials, uploads, telemetry, or self-updater. `detect` and the scripts make no network calls.
- Enabling the plugin asks Hermes to install numpy, scipy, soundfile, matplotlib, and tifffile (pinned ranges in `pyproject.toml`). Install `ffmpeg` yourself.

Details are in [SECURITY.md](SECURITY.md).

## Install

Until a maintainer lists it in the Hermes catalog:

```bash
hermes plugins install mojomast/hermes-aircraft-doppler-geolocation --no-enable
hermes plugins enable aircraft-doppler-geolocation
```

After catalog acceptance, the reviewed install is `hermes plugins install aircraft-doppler-geolocation`.

## Use it from a shell

```bash
S=skills/aircraft-doppler-geolocation/scripts
python $S/doppler_track.py clip.mp4 --fmin 60 --fmax 250 --out ev1
python $S/score_candidates.py track.csv candidates.csv ev1.csv --f0 110 --rec-start 1725827820 --dt-tol 2 --plot scores.png
python $S/make_synthetic_test.py
```

## Self-test result

Images below come from real runs of the scripts on the synthetic self-test, not mockups.

**`doppler_track.py`:** band spectrogram with the traced ridge (cyan) and the fitted flyover model (green).

![Synthetic spectrogram with traced ridge and fit](docs/screenshots/spectrogram-fit.png)

**`score_candidates.py --plot`:** misfit per candidate on a 200 m grid, the curving ADS-B track, the top 5 ranks, and the true observer (star). Rank 1 is the true point; the mirror-image cluster on the other side of the track is ranks 3 to 5.

![Candidate scoring heatmap](docs/screenshots/candidate-scoring.png)

**Terminal:** the self-test, pytest, and `hermes plugins validate .` output captured from the box and rendered to PNG.

![Self-test, pytest and hermes validate passing](docs/screenshots/terminal-tests.png)

A synthetic propeller tone (110 Hz, 80 m/s, 900 m altitude, gently turning) with tone amplitude 0.29 against noise σ 1.0 is recovered as f0 110.0 Hz, 80.4 m/s, slant 1039 m (true 1040 m). With the recording start known to ±2 s, the true grid point ranks first (0 m error on a 200 m grid). Without a known time, a single flyover lands at the right distance but about 1.4 km off along the track, which is the ambiguity the video describes. A second flyover at a fixed offset, or a known time, is required.

## Real-world geolocation results (v0.2.0)

The full pipeline (events → historical ADS-B → grid search) was evaluated on public recordings with a known location and start time: the original 12 clips plus 25 livestream VODs filmed at known spotting locations. Clips were split by location before any tuning, giving 23 tune clips and 14 held-out clips. Runs use a ±30 s start-time tolerance (the original tune clips also keep a ±600 s run for crq; dab_part1 runs at ±9000 s). Each run searched a 15 km radius around a centre offset 5 km from the truth. Settings were tuned on the tune split only. The held-out split was then run once with the frozen defaults (`sigma_model=0.008`, `detect_prob=0.15`, sub-harmonic relabel, CPA-distance consistency term, `temper_extra=4`).

![Per-clip error, tune vs held-out](docs/screenshots/summary.png)

| | runs | no events / no ADS-B | solved | error < 3 km | median error (solved) | truth inside 95% area* |
|---|---|---|---|---|---|---|
| Tune | 24 | 11 / 1 | 12 | 5 | 7.2 km | 9 / 12 |
| Held-out, previous (5 clips, ±30 s and ±600 s) | 9 | 0 | 9 | 1 | 10.8 km | 3 / 9 |
| **Held-out, final (14 clips, ±30 s)** | 14 | 6 / 0 | 8 | **1** | **12.1 km** | 6 / 8 |

*"95% area" is the 95% credible area padded by the truth location's own uncertainty, using the shipped calibration. Point estimates on the 5 original held-out clips at ±30 s are unchanged from the previous run; the median moved because the new held-out clips are harder, not because the method got worse.

**Livestream clips mostly fail.** 12 of the 16 new tune clips and 6 of the 9 new held-out clips produced no usable events. Spectrograms show why: the stream audio is dominated by host commentary or gated ATC radio, is high-passed (around 80–120 Hz) and compressed, and jet passes appear as broadband noise without tonal engine lines. ADS-B had plenty of close passes for these clips, so the data side is not the problem. Four of these tune clips were diagnosed individually and marked excluded in the manifest; the rest are counted as failures above. Fixing this would need a broadband Doppler model rather than looser detector gates.

**Calibration (`temper_extra=4`).** Without extra tempering, the 95% areas were badly overconfident (the unpadded area contained the truth in 0 of 23 earlier runs). The search now multiplies the Birge temperature by `temper_extra` (default 4, chosen on the tune split). On tune this covers 9 of 12 solved runs; at 64 it covers 11 of 12 but the area grows to the whole 15 km search disc, which is useless. With only 12 tune rows, this calibration is rough.

**Confidence flag.** A result is flagged `confidence: low` when fewer than 2 matched events fit well and agree with the CPA distance. Low-confidence results report the whole search area as their 95% region, so the 95% region is only meaningful when confidence is `ok`.

| | right (< 3 km) | wrong |
|---|---|---|
| tune: ok | 3 | 0 |
| tune: low | 2 | 7 |
| held-out: ok | 0 | 1 |
| held-out: low | 1 | 6 |

**What works:** clips with several clean propeller passes near an airfield (crq, lgb_0810) locate to about 1 km and are flagged ok.

**Limitations:** on unseen clips the method mostly fails (1 of 14 held-out runs under 3 km, 6 with no events at all). An `ok` flag is not a guarantee: osh_0719 was the only held-out `ok` result, it was 13.2 km off, and its 95% area missed the truth. One flyover leaves the along-track position ambiguous, and the likelihood then prefers the wrong aircraft. With these small samples the numbers are indicative, not precise. Treat any output as a hypothesis to check, not as a location.

## Real-world test (v0.1.1)

Run on 5 public YouTube clips (audio only), 2026-11-02:

| Clip | Result |
| --- | --- |
| [Planes Flying Overhead While the Birds Sing](https://www.youtube.com/watch?v=G0fjcT-SP1o) (City Limits Sound) | 2 flyovers detected: 94.9 Hz, 59 m/s, 1.53 km, 1.27 Hz rms; and 247.6 Hz (2nd harmonic), 97 m/s, 2.08 km, 0.61 Hz rms. The two harmonics of pass 2 disagree on speed by about 8%, so treat the stderr as optimistic. |
| [Cessna 421 takeoff, Puerto Princesa](https://www.youtube.com/watch?v=DqZ2411R1ys) (Palawan Skies Aviation) | Detected in 85-145 Hz: 109.4 Hz, 63 m/s (~122 kt), 295 m. A higher band failed: the tracker hops between crossing harmonics. |
| [Cessna Conquest 425 flyby, Hannover](https://www.youtube.com/watch?v=yltKJzPAcb4) (HD1080ide) | Rejected. The plane was turning, so the straight-pass model does not apply. |
| [Evening City Ambience in Amsterdam](https://www.youtube.com/watch?v=VZDnXSN6Cgg) (Jeronimo Ambience) | No detections. |
| [Outdoor atmosphere with plane flying over](https://www.youtube.com/watch?v=aeC-UfLcxqk) (Your Film Sounds) | Broadband roar with no tone. 0.1.0 reported a false fit here; 0.1.1 rejects it (`ridge_snr_z` 1.7). |

![Real-world fit: City Limits Sound clip, pass 2](docs/screenshots/realworld-g0-pass2.png)

*Spectrogram of audio from City Limits Sound's video [Planes Flying Overhead While the Birds Sing in the Distance](https://www.youtube.com/watch?v=G0fjcT-SP1o), with the traced ridge and fit.*

![Real-world fit: Cessna 421 takeoff](docs/screenshots/realworld-cessna421.png)

*Spectrogram of audio from Palawan Skies Aviation's video [Cessna 421 Golden Eagle takeoff at Puerto Princesa](https://www.youtube.com/watch?v=DqZ2411R1ys).*

The tracker works on clean real tones. You still pick the frequency band and time window by eye from a spectrogram, and the `detected` flag is the guard against noise. Real tones scored 2.7 to 3.6 on `ridge_snr_z` and noise scored 1.7 to 2.0, so the margin is narrow. Geolocation scoring could not be tested: none of the clips gives a recording date or time, and no free historical ADS-B was used.

## Limitations

- ADS-B fetch, jet-type filtering, DEM lookup, the time-offset and silence filters, and grid generation are described in the skill but **not scripted** yet.
- Detection is tested on real audio (above), but you choose the band by hand. Location scoring was evaluated on 12 real clips (see above): about 1 km on clean multi-pass clips, but only 1 of 9 held-out runs was under 3 km, and the 95% areas are overconfident.
- Harmonics that cross each other and turning flights break the tracker or the straight-pass fit.
- Needs ADS-B coverage, a date, and serious compute for large search areas.

## Ethics

Use for forensics, journalism, research, or auditing your own recordings. Locating a person without consent can put them in danger; follow local law. Results are probabilistic and need independent visual confirmation. To protect your own clips: high-pass filters and talking over the tone do not reliably remove it, and it can survive phone compression; cutting segments, stronger noise suppression, and stripping timestamps help. Check your own spectrogram with `doppler_track.py` before publishing.

## Tests

```bash
pip install -e ".[dev]"   # needs ffmpeg on PATH
python -m unittest discover -s tests -v   # or: python -m pytest tests -v
```

## License

MIT
