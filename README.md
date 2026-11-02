# aircraft-doppler-geolocation

Hermes plugin that locates where (and optionally when) a recording was made from the Doppler curves of aircraft flying over, often too faint to hear, matched against ADS-B flight tracks. It ships one skill and three local scripts.

It is a community plugin by [Kyle Durepos](https://github.com/mojomast). It is not a Nous Research product. The method comes from colsto's video [I discovered a sound that leaks your exact location (and it's everywhere)](https://www.youtube.com/watch?v=UytYrhYLOKI). The author released no code; these scripts are an independent implementation of the steps he describes, not his pipeline.

![Catalog banner](docs/catalog-banner.png)

## What you get

One skill, `aircraft-doppler-geolocation:aircraft-doppler-geolocation`, and three scripts in `skills/aircraft-doppler-geolocation/scripts/`:

| Script | Job |
| --- | --- |
| `doppler_track.py` | Decode audio/video with ffmpeg, trace the flyover tone on a narrow-band spectrogram (Viterbi ridge), and fit base tone, speed, closest-approach slant distance, and closest-approach time. Writes CSV, PNG, JSON. |
| `score_candidates.py` | For a real flight track (ADS-B CSV) and candidate points (lat, lon, ground elevation), model the exact curve each point would hear using the velocity at emission time, and rank points by misfit in Hz. `--plot` writes a heatmap. |
| `make_synthetic_test.py` | Offline self-test: synthetic curved flyover buried in noise, fit, then locate the observer on a grid. |

## Disclosure

- `register()` only calls `ctx.register_skill`. No tools, hooks, or middleware.
- `doppler_track.py` runs `ffmpeg` (no shell) on the path you pass. Scripts read only the paths you pass and write the outputs you name.
- No network calls, credentials, telemetry, or self-updater. The plugin does **not** fetch ADS-B or elevation data; the skill names sources (pyopensky, traffic, adsb.lol, ADS-B Exchange, OpenTopography) for you to use.
- Enabling the plugin asks Hermes to install numpy, scipy, soundfile, and matplotlib (pinned ranges in `pyproject.toml`). Install `ffmpeg` yourself.

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

## Limitations

- ADS-B fetch, jet-type filtering, DEM lookup, the time-offset and silence filters, and grid generation are described in the skill but **not scripted** yet.
- Tested on synthetic data only. Not validated on a real recording plus real ADS-B.
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
