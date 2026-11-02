"""Hermes plugin entrypoint. Registers the skill and the doppler_* tools."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def register(ctx) -> None:
    """Register skills/<name>/SKILL.md and the doppler_detect / doppler_locate tools.

    Tools use the keyword form of register_tool (name, toolset, schema, handler),
    as in photo-to-part 0.1.1. numpy/scipy load when a tool runs, not here.
    """
    skills_dir = _ROOT / "skills"
    if skills_dir.is_dir():
        for child in sorted(path for path in skills_dir.iterdir() if path.is_dir()):
            skill_md = child / "SKILL.md"
            if skill_md.is_file():
                ctx.register_skill(child.name, skill_md)
    try:
        from dopplergeo.tools import register_tools
    except (
        ImportError
    ):  # plugin checkout not installed as a package: import from the plugin dir
        sys.path.insert(0, str(_ROOT))
        from dopplergeo.tools import register_tools

    register_tools(ctx)
