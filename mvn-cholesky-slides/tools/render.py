"""Render the deck with a clean progress display instead of Manim's log spam.

    python tools/render.py                      # all sections, 480p (fast preview)
    python tools/render.py --quality h          # all sections, 1080p60
    python tools/render.py --quality h --fps 30 # all sections, 1080p30 (about 2x faster)
    python tools/render.py s2 s4 --quality h    # only some sections

Shows one bar for the whole run and one for the current section, with
elapsed time and an estimate of what is left. Manim's full output goes to
media/render.log; if a render fails, the end of that log is printed.

The bars count Manim animations. The total for each section comes from the
last time it was rendered (Manim's partial_movie_file_list.txt), so the very
first render of a section shows a count without a percentage.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

# Short name -> (scene file, Slide class), in presentation order.
SECTIONS = {
    "s1": ("scenes/s1_problem.py", "S1Problem"),
    "s2": ("scenes/s2_solution.py", "S2Solution"),
    "s3": ("scenes/s3_sampler.py", "S3Sampler"),
    "s3b": ("scenes/s3_sampler.py", "S3bThreeD"),
    "s4": ("scenes/s4_wrapup.py", "S4WrapUp"),
}
ANIMATION_LINE = re.compile(rb"Animation (\d+)\s*:")  # "Animation 12:" or "... 12 : Using cached"
FINISHING_LINE = re.compile(rb"Concatenating animations|Combining to Movie file")
LOG_FILE = ROOT / "media" / "render.log"

console = Console()


def default_sections() -> list[str]:
    names = ["s1", "s2", "s3", "s4"]
    if config.ENABLE_3D_BONUS:
        names.insert(3, "s3b")
    return names


def previous_animation_count(scene_file: str, scene: str) -> int | None:
    """How many animations the scene had in its most recent render, if any."""
    stem = Path(scene_file).stem
    lists = (ROOT / "media" / "videos" / stem).glob(
        f"*/partial_movie_files/{scene}/partial_movie_file_list.txt"
    )
    newest = max(lists, key=lambda p: p.stat().st_mtime, default=None)
    if newest is None:
        return None
    return sum(1 for line in newest.read_text().splitlines() if line.startswith("file"))


def render_section(name: str, args: argparse.Namespace, progress: Progress, overall, log) -> bool:
    """Run manim-slides for one section, moving both bars as animations finish."""
    scene_file, scene = SECTIONS[name]
    expected = previous_animation_count(scene_file, scene)
    section_bar = progress.add_task(f"{scene:<11}", total=expected)
    command = [
        str(ROOT / ".venv" / "bin" / "manim-slides"),
        "render",
        f"--quality={args.quality}",  # not "-qh": manim-slides reads that "h" as --help
        *(["--fps", str(args.fps)] if args.fps else []),
        scene_file,
        scene,
    ]
    log.write(f"\n===== {' '.join(command)}\n".encode())
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
    )
    done = 0
    buffer = b""
    while chunk := process.stdout.read(4096):
        log.write(chunk)
        buffer += chunk
        # Manim redraws its own bars with "\r", so split on both kinds of line end.
        *lines, buffer = re.split(rb"[\r\n]", buffer)
        for line in lines:
            match = ANIMATION_LINE.search(line)
            # "Animation N" is printed when animation N starts, so N animations are done.
            if match and int(match.group(1)) > done:
                step = int(match.group(1)) - done
                done += step
                if expected is not None and done > expected:  # the slide grew since last time
                    progress.update(section_bar, total=done)
                progress.update(section_bar, completed=done)
                progress.advance(overall, step)
            elif FINISHING_LINE.search(line):
                progress.update(section_bar, description=f"{scene:<11} (saving)")
    ok = process.wait() == 0
    # Help text instead of a render means bad arguments; treat it as a failure.
    ok = ok and done > 0
    if ok:
        done += 1  # the last animation, which no later "Animation N" line announced
        progress.advance(overall, 1)
        progress.update(section_bar, total=done, completed=done, description=f"{scene:<11} done")
        if expected is not None and done < expected:  # the section shrank: drop the extra
            progress.update(overall, total=progress.tasks[overall].total - (expected - done))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the deck with a progress bar.")
    parser.add_argument("sections", nargs="*", help=f"any of {', '.join(SECTIONS)} (default: all)")
    parser.add_argument("--quality", choices=["l", "m", "h"], default="l")
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()
    names = args.sections or default_sections()
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")

    expected = [previous_animation_count(*SECTIONS[n]) or 0 for n in names]
    label = {"l": "480p", "m": "720p", "h": "1080p"}[args.quality]
    label += f"{args.fps}" if args.fps else {"l": "15", "m": "30", "h": "60"}[args.quality]
    console.print(
        f"[bold]Rendering {', '.join(names)} at {label}[/]  (log: {LOG_FILE.relative_to(ROOT)})"
    )

    LOG_FILE.parent.mkdir(exist_ok=True)
    start = time.monotonic()
    columns = (
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("animations"),
        TimeElapsedColumn(),
        TextColumn("left"),
        TimeRemainingColumn(),
    )
    with LOG_FILE.open("wb") as log, Progress(*columns, console=console) as progress:
        overall = progress.add_task("[bold]All sections", total=sum(expected) or None)
        for name in names:
            if not render_section(name, args, progress, overall, log):
                progress.stop()
                log.flush()
                tail = LOG_FILE.read_text(errors="replace").splitlines()[-30:]
                console.print("\n".join(tail), markup=False, highlight=False)
                console.print(f"[bold red]FAILED[/] while rendering {SECTIONS[name][1]}.")
                return 1
    minutes, seconds = divmod(int(time.monotonic() - start), 60)
    console.print(f"[bold green]DONE[/] rendered {', '.join(names)} in {minutes}m {seconds:02d}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
