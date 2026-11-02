"""Hermes plugin entrypoint. Registers bundled skills and nothing else."""

from pathlib import Path


def register(ctx) -> None:
    """Register each skills/<name>/SKILL.md under this plugin's namespace.

    The call is the two-argument form, which both current and older Hermes
    builds accept. Hermes reads the description from the skill file.
    """
    skills_dir = Path(__file__).resolve().parent / "skills"
    if not skills_dir.is_dir():
        return
    for child in sorted(path for path in skills_dir.iterdir() if path.is_dir()):
        skill_md = child / "SKILL.md"
        if skill_md.is_file():
            ctx.register_skill(child.name, skill_md)
