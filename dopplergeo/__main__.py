"""CLI: python -m dopplergeo {detect,locate} ..."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dopplergeo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="find flyover Doppler events (offline)")
    d.add_argument("recording")
    d.add_argument("--out")
    lc = sub.add_parser(
        "locate", help="detect events and score locations against ADS-B"
    )
    lc.add_argument("recording")
    lc.add_argument(
        "--date",
        help="YYYY-MM-DD (time unknown) or ISO datetime of recording start (UTC unless offset given)",
    )
    lc.add_argument("--region", help="lat,lon,radius_km")
    lc.add_argument(
        "--offline", action="store_true", help="use only cached ADS-B, DEM and weather"
    )
    lc.add_argument("--time-tolerance", type=float, default=600.0)
    lc.add_argument("--include-jets", action="store_true")
    lc.add_argument("--out")
    a = ap.parse_args(argv)
    from .locate import detect_only, locate

    if a.cmd == "detect":
        rep = detect_only(a.recording, a.out)
    else:
        rep = locate(
            a.recording,
            a.date,
            a.region,
            offline=a.offline,
            out_dir=a.out,
            time_tolerance_s=a.time_tolerance,
            include_jets=a.include_jets,
            progress=lambda *p: print("progress", *p, file=sys.stderr),
        )
    print(json.dumps(rep, indent=2))
    return 0 if rep.get("ok", True) else 2


if __name__ == "__main__":
    sys.exit(main())
