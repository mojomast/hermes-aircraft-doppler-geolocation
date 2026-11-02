# Security policy

Report vulnerabilities through GitHub private vulnerability reporting if it is enabled. Otherwise open a short issue that asks for a private channel. Do not include exploit details in a public issue.

This plugin is trusted in-process Python, the same as any other Hermes plugin. Review the source before you enable it.

What it does:

- `register()` calls `ctx.register_skill` and registers two tools, `doppler_detect` and `doppler_locate`. No hooks or middleware. The only environment variables read are `DOPPLER_GEO_CACHE` (cache location) and `DOPPLER_GEO_NETWORK_TESTS` (tests only).
- `doppler_track.py` runs `ffmpeg` as a subprocess (argument list, no shell) to decode the audio or video path you pass into a temporary WAV. The scripts otherwise read only the CSV/audio paths you pass and write the output paths you pass.
- Since 0.2.0, `locate` / `doppler_locate` makes unauthenticated HTTPS GET requests (https only) to api.github.com and github.com (adsb.lol release assets, streamed and cropped; only the crop is written to the cache), raw.githubusercontent.com (tar1090-db type table), copernicus-dem-30m.s3.amazonaws.com (DEM tiles), and archive-api.open-meteo.com (temperature). `--offline` / `offline=true` uses the cache only. Nothing is uploaded. `detect` and the scripts make no network calls.
- Nothing reads `.env` files or Hermes credential stores. There is no telemetry and no self-updater.

Enabling the plugin asks Hermes to install the Python packages listed in `pyproject.toml` from PyPI. `ffmpeg` is not installed by the plugin.
