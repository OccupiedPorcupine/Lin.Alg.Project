# Simulating the Multivariate Normal with Cholesky

A [Manim Slides](https://manim-slides.eertmans.be/) deck for the project
**Simulation of Multivariate Normal Distribution with a Given Covariance Matrix**:

- **(a)** simulate N(μ, Σ) from a given Σ with the Cholesky decomposition;
- **(b)** vary Σ and show how it changes the shape and spread of the data.

The story in one line: computers give us independent noise N(0, I), we want
correlated data N(μ, Σ), and the Cholesky factor G of Σ (with G Gᵀ = Σ) is the
"matrix square root" that turns one into the other via x = μ + G z.

Notation used everywhere: **G** is the Cholesky factor, **L** only appears on
slide 2.4 as the unit lower-triangular factor from elimination, and **A** is a
general matrix on slides 2.1 to 2.3.

## Setup

Tested with Python 3.12, Manim Community 0.21.0, manim-slides 5.7.0, numpy 2.5,
MacTeX 2026 and ffmpeg 8 on macOS (Apple Silicon).

Keep Manim at 0.21.x for now: Manim 0.22 removed the `write_to_movie` setting,
and manim-slides 5.7.0 still sets it, so rendering crashes with
`AttributeError: ... no attribute 'write_to_movie'`. `requirements.txt` pins it.

```bash
# System dependencies (macOS)
brew install --cask mactex-no-gui      # LaTeX, needed for every formula
brew install ffmpeg cairo pango pkgconf

# Python environment, local to this folder
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The Makefile uses `.venv/bin` and `/Library/TeX/texbin` automatically, so there
is no need to activate the environment for `make` commands.

## Run it

| Command | What it does |
|---|---|
| `make test` | Run the math tests (`tests/test_mvn_math.py`). |
| `make render-low` | Render all four sections at 480p. Fast; use while editing. |
| `make render-fast` | Render all four sections at 1080p, 30 fps. About 2x faster than `make render`. |
| `make render` | Render all four sections at 1080p, 60 fps for the real talk. |
| `make s3` | Render one section (`s1` to `s4`); add `QUALITY=h` for 1080p and `FPS=30` for 30 fps. |
| `make present` | Open the presenter. Right arrow or space: next. Left: back. |
| `make html` | Export `slides.html`, one self-contained file as a backup. |
| `make notes` | Rebuild `notes/speaker_notes.md` from the scenes. |
| `make lint` | Run ruff checks and the formatter check. |

Every render command shows a progress bar for the whole run and for the
current section, with time elapsed and left, and prints **DONE** or **FAILED**
at the end. Manim's full output goes to `media/render.log`; on failure the end
of it is printed. The bars count animations, using each section's count from
its previous render, so a section's very first render shows a count but no
percentage. The script is `tools/render.py`.

Under the hood a section is rendered with

```bash
manim-slides render --quality=l scenes/s1_problem.py S1Problem
```

(`--quality=h` for 1080p). Write the quality as `--quality=h`, not `-qh`:
manim-slides has its own `-h` flag and treats `-qh` as a request for help.
The deck is presented with
`manim-slides present S1Problem S2Solution S3Sampler S4WrapUp`. Rendering writes
videos to `media/` and the slide index to `slides/`; both can be deleted with
`make clean`.

In the presenter, press `H` to hide or show the controls window. In the HTML
export, press `S` for the speaker view with notes.

## Project layout

```
config.py        every tweakable number: matrices, sample sizes, timings
theme.py         colors, font sizes, layout constants, title and caption helpers
mvn_math.py      pure numpy math, no Manim; every number on screen comes from here
components.py    reusable Manim pieces (axes, clouds, ellipses, matrices, panels)
                 and DeckSlide, the shared base class for the sections
scenes/          one Slide class per section: S1Problem, S2Solution, S3Sampler, S4WrapUp
tests/           pytest suite for mvn_math (plus a speaker-notes sync check)
notes/           speaker_notes.md and build_notes.py, which generates it from the scenes
tools/render.py  the progress-bar renderer behind the make render targets
ruff.toml        lint and format settings (`make lint`)
SLIDE_FLOW.md    every slide and click in order; edit it to plan changes to the flow
```

Each slide is one method named after its outline number, for example
`slide_2_5_algorithm_and_hero`, and each section's `construct()` is just the
ordered list of those methods. Slide titles on screen carry no numbers;
instead each section opens with a section slide (`slide_2_0_section`, and so
on) showing the section number, name and a one-line subtitle. The name comes
from `section_name` at the top of the section class. There is no footer.
The numbers still appear in `notes/speaker_notes.md` for rehearsal.

## How to change things

**Change Σ or μ.** Edit `MAIN_SIGMA` and `MAIN_MU` in `config.py`. Every
matrix, readout, ellipse and axis range is computed from them, so just
re-render. Σ must be symmetric positive definite; `make test` will tell you if
it is not. If you change `MAIN_SIGMA`, two tests that pin the default numbers
(the factor `[[2, 0], [1, 2]]` and the worked point `[2, 0.5]`) will fail by
design: update or delete them. In the hero animation (slide 2.5) the moving
grid covers z in ±`HERO_GRID_HALF_WIDTH`; lower it if a large Σ pushes the
grid off the axes. The other matrices each have their own block in
`config.py`: `EXAMPLE_3X3_SIGMA` (the hand computation on 2.4 and the 3×3
check on 3.1), `DEMO_STRUCTURES` with `DEMO_CAPTIONS` (3.2),
`EIGEN_EXAMPLE_SIGMA` and `EIGEN_EXAMPLE_SIGMA_2` (3.3), `RHO_SWEEP_PATH`,
`RHO_ZERO_PIVOT` and `RHO_NEGATIVE_PIVOT` (3.4), and the hook's
`PORTFOLIO_*`, `GP_*` and `SYNTH_*` values (1.1, replayed on 4.2 and 4.3).
Some tests pin the spec's exact numbers (for example the 3×3 elimination
steps); they fail by design when you change those matrices.

**Change the sample count.** `N_DISPLAY` is how many dots are drawn (keep it
under about 1500 or rendering gets slow). `N_STATS` is how many samples are used
for the measured mean and covariance; it is cheap, so it can be large.
`RANDOM_SEED` changes which samples you get.

**Change colors.** Edit the semantic colors at the top of `theme.py`
(`COLOR_Z`, `COLOR_X`, `COLOR_SIGMA`, `COLOR_G`, ...). Each has one meaning across
the whole deck, so one edit recolors every slide consistently. Font sizes and
the background color are in the same file.

**Skip or reorder a slide.** In the section's `construct()`, comment out or
move the `self.slide_...()` line. Every slide builds its own objects and starts
by fading out whatever is on screen, so slides do not depend on each other.
Then re-render that section and run `make notes`. To drop a whole section,
remove it from `SCENES` in the Makefile.

If the talk runs long, cut in this order (each is one line in `construct()`):

1. Appendix A.1 (`slide_a_1_derivation` in `S4WrapUp`) sits after the closing
   slide already; only show it if asked.
2. 1.4 Reading the formula. Sampling never uses the density, and 3.1's
   contour ellipses still work without it.
3. 2.4e The shortcut. Keep only its final "general pattern" reveal, or drop it.
4. 3.3 The eigen view. If you cut it, remove "(eigen view)" on slide 4.3.
5. 3.5 (the 3D bonus) is already off by default.

Two slides are deliberately linked: slide 2.5 ends on exactly the frame that
Section 3 opens with (both build it with `HeroFrame` in `components.py`), so
the orange cloud carries across the section boundary. If you cut or change
2.5's ending, keep that in mind.

**Edit speaker notes.** The notes live at the top of each slide method, in
`self.begin_slide(notes="""...""")`. Edit them there, re-render the section,
then run `make notes` to refresh `notes/speaker_notes.md` (`make test` fails if
you forget). Keep them free of em dashes; a test checks that too.

**Re-render just one section.** `make s2` (480p) or `make s2 QUALITY=h` (1080p).
`make present` always uses whatever was rendered last for each section.

**Change timing.** `T_SHORT`, `T_MEDIUM`, `T_LONG` and
`RHO_SWEEP_SECONDS_PER_LEG` in `config.py`.

**Turn on the 3D bonus slide (3.5).** Set `ENABLE_3D_BONUS = True` in
`config.py`. The Makefile then renders `S3bThreeD` (the 3D cloud of
`EXAMPLE_3X3_SIGMA`) and puts it between sections 3 and 4. It is off by
default because it is slow to render.

**Other switches.** `ENABLE_EIGEN_SECOND_EXAMPLE` (the second matrix on 3.3) and
`ENABLE_NEGATIVE_PIVOT_REVEAL` (the ρ = 1.2 step on 3.4) can each be turned off
in `config.py`.

## Notes on a few design choices

- Slide transitions: each slide starts by fading out everything from the
  previous slide.
- `next_slide(notes=...)` attaches notes to the slide that *follows* the call
  in manim-slides 5, so the call sits at the start of each slide, inside
  `DeckSlide.begin_slide`. Reveal pauses inside a slide (`self.reveal()`) repeat
  the same notes so the presenter view never goes blank.
- The 2.5 hero animation maps points through data coordinates
  (screen → `plane.p2c` → apply G → `plane.c2p`) instead of `ApplyMatrix`,
  because the plane is not unit scale. The scene asserts that the tracked point
  lands exactly on μ + G z.
- The ρ sweep on 3.4 never calls `np.linalg.cholesky` at ρ ≥ 1: the pivot
  1 − ρ² and the factor come from closed forms in `mvn_math.py`
  (`unit_corr_pivot`, `unit_corr_factor`).
