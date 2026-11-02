# Security policy

Report vulnerabilities through GitHub private vulnerability reporting if it is enabled. Otherwise open a short issue that asks for a private channel. Do not include exploit details in a public issue.

This plugin is trusted in-process Python, the same as any other Hermes plugin. Review the source before you enable it.

What it does:

- `register()` only calls `ctx.register_skill`. It does not register tools, hooks, or middleware, and it does not read the environment.
- `doppler_track.py` runs `ffmpeg` as a subprocess (argument list, no shell) to decode the audio or video path you pass into a temporary WAV. The scripts otherwise read only the CSV/audio paths you pass and write the output paths you pass.
- No script opens a socket, fetches ADS-B data, reads `.env` files, or reads Hermes credential stores. There is no telemetry and no self-updater.
- ADS-B history, aircraft type tables, and elevation data are fetched by you (or by the agent with your approval) from the services named in the skill; this plugin does not do it.

Enabling the plugin asks Hermes to install the Python packages listed in `pyproject.toml` from PyPI. `ffmpeg` is not installed by the plugin.
