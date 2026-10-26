"""Rebuild notes/speaker_notes.md from the speaker notes in the scene files.

The notes live next to the visuals, in each slide's ``begin_slide(notes=...)``
call. This script reads them straight from the source (no rendering needed),
so the Markdown file can never drift from what the presenter view shows.

    python notes/build_notes.py          # rewrite notes/speaker_notes.md
    python notes/build_notes.py --check  # exit 1 if the file is out of date
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

SCENE_FILES = ["s1_problem.py", "s2_solution.py", "s3_sampler.py", "s4_wrapup.py"]
OUTPUT = ROOT / "notes" / "speaker_notes.md"


def _string_value(node: ast.AST) -> str | None:
    """The text of a plain string, or of an f-string evaluated against config.

    Scene notes may say f"{cfg.N_STATS:,} samples"; evaluating gives the same
    words the presenter view shows.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        code = compile(ast.Expression(node), "<notes>", "eval")
        return eval(code, {"cfg": config})
    return None


def _clean(notes: str) -> str:
    return " ".join(textwrap.dedent(notes).split())


def _slide_heading(method: ast.FunctionDef, section_name: str) -> str:
    """'1.3  What we want': number from the method name, words from the slide.

    The words come from the slide_title(...) call, or name the section on a
    section slide, or fall back to the method name.
    """
    _, major, minor, *words = method.name.split("_")
    number = f"{major.upper()}.{minor}"  # "a_1" (the appendix) becomes "A.1"
    for node in ast.walk(method):
        if isinstance(node, ast.Call):
            called = getattr(node.func, "attr", "")
            if called == "slide_title":
                return f"{number}  {_string_value(node.args[0])}"
            if called == "show_section_slide":
                return f"{number}  Section slide: {section_name}"
    return f"{number}  " + " ".join(words).capitalize()


def _section_name(cls: ast.ClassDef) -> str:
    for node in cls.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "section_name":
            return _string_value(node.value) or ""
    return ""


def _slide_notes(method: ast.FunctionDef) -> str | None:
    """Notes passed to begin_slide(...) or next_slide(...) at the top of a slide."""
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in (
            "begin_slide",
            "next_slide",
        ):
            for keyword in node.keywords:
                if keyword.arg == "notes" and _string_value(keyword.value):
                    return _clean(_string_value(keyword.value))
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "notes":
            if _string_value(node.value):
                return _clean(_string_value(node.value))
    return None


def build() -> str:
    lines = [
        "# Speaker notes",
        "",
        "Generated from the scene files by `python notes/build_notes.py`. "
        "Edit the notes in the scenes, then rerun it (or `make notes`).",
        "",
    ]
    for filename in SCENE_FILES:
        tree = ast.parse((ROOT / "scenes" / filename).read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            slides = [
                m
                for m in cls.body
                if isinstance(m, ast.FunctionDef) and m.name.startswith("slide_")
            ]
            if not slides:  # a helper class, not a section
                continue
            lines += [f"## {cls.name}", ""]
            section_name = _section_name(cls)
            for method in slides:
                notes = _slide_notes(method)
                heading = _slide_heading(method, section_name)
                lines += [f"### {heading}", "", notes or "(no notes)", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    content = build()
    if "--check" in sys.argv:
        sys.exit(0 if OUTPUT.read_text() == content else 1)
    OUTPUT.write_text(content)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
