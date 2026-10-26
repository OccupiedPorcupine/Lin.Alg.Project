"""Colors, font sizes, layout constants, and the title and caption helpers.

Colors have a fixed meaning across the whole deck (see the semantic block
below). Change a color here and every slide follows.
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    UL,
    MarkupText,
    MathTex,
    Text,
    TexTemplate,
    VGroup,
    config,
)

# --- Canvas -----------------------------------------------------------------

BACKGROUND = "#0f1115"
config.background_color = BACKGROUND  # set once, before any scene is built
FRAME_WIDTH = config.frame_width  # 16:9 frame: about 14.2 by 8 Manim units
FRAME_HEIGHT = config.frame_height

# Words inside formulas (\text{...}) use a sans font so they sit closer to the
# slide font; the math itself stays in the standard LaTeX math font.
TEX_TEMPLATE = TexTemplate()
TEX_TEMPLATE.add_to_preamble(r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}")
config.tex_template = TEX_TEMPLATE

# --- Semantic colors (same meaning on every slide) ---------------------------

COLOR_Z = "#4ea1ff"  # independent samples z and N(0, I): blue
COLOR_X = "#ff9f43"  # correlated samples x and N(mu, Sigma): orange
COLOR_SIGMA = "#2ec4b6"  # Sigma, the blueprint: teal
COLOR_G = "#ffd23f"  # G, the Cholesky factor: yellow
COLOR_DIST = "#ff6fb5"  # distance term of the density formula (1.4): pink
COLOR_NORM = "#a3a9b3"  # normalizer term of the density formula (1.4): grey
COLOR_Z1 = COLOR_Z  # z1 where z1 and z2 must be told apart: blue
COLOR_Z2 = "#b48cff"  # z2 where z1 and z2 must be told apart: purple
COLOR_BAD = "#ff5c5c"  # errors, failures, negative pivots: red
COLOR_WARN = "#ffb627"  # warnings and zero pivots: amber
COLOR_GOOD = "#4cd964"  # confirmations, positive pivots: green

# --- Neutral colors ----------------------------------------------------------

COLOR_TEXT = "#e8e8e8"  # default text
COLOR_MUTED = "#7d838c"  # axes, greyed-out zeros, secondary text
COLOR_GRID = "#2a2f38"  # faint background grid lines
COLOR_PANEL = "#181b22"  # fill behind boxes (algorithm, code)

# --- Fonts -------------------------------------------------------------------

FONT = "Avenir Next"  # sans font for all plain text
CODE_FONT = "Menlo"  # monospace font for the code reveal
QUESTION_SIZE = 80  # the big "?" on slide 1.5
SECTION_NUMBER_SIZE = 110  # the big number on each section slide
DECK_TITLE_SIZE = 56  # only the title slide and the closing slide
TITLE_SIZE = 44
BODY_SIZE = 32
CAPTION_SIZE = 26
MIN_SIZE = 24  # nothing on screen is smaller than this
# LaTeX glyphs look about 20% smaller than Avenir at the same font_size, so
# every formula is scaled up by this factor to read at the same size as text.
MATH_SCALE = 1.2

# --- Layout ------------------------------------------------------------------

EDGE_BUFF = 0.45  # distance from the frame edge for titles and side panels
TITLE_GAP = 0.35  # gap between the title and the content below it
CAPTION_Y = -3.05  # vertical position of the caption line at the bottom
CONTENT_TOP = 2.6  # content should stay below this y
CONTENT_BOTTOM = -2.6  # and above this y, so it never meets the caption
MAX_CONTENT_WIDTH = FRAME_WIDTH - 2 * EDGE_BUFF

PANEL_LARGE = 4.4  # side of a single main scatter plot
PANEL_MEDIUM = 3.5  # side of each plot in a side-by-side pair
PANEL_SMALL = 2.0  # side of each plot in the 2x2 grid (slide 3.2)
PANEL_STRIP = 3.0  # side of each plot in the recap strip (4.1)
DENSITY_WIDTH = 11.5  # 1D bell-curve axes (slide 1.6)
DENSITY_HEIGHT = 2.9

# Gaussian process panel (1.1 and its replay on 4.2): same size on both slides.
GP_WIDTH = 10.0
GP_HEIGHT = 3.6
GP_DOT_RADIUS = 0.06

# Hero animation (2.5). Section 3 opens on this exact frame, so both sections
# build it from these values (see components.hero_final_frame).
HERO_MAX_WIDTH = 7.0  # box the hero NumberPlane must fit in
HERO_MAX_HEIGHT = 6.5
HERO_PLANE_CENTER = (-3.2, -0.45)
HERO_CODE_Y = 0.3  # vertical position of the NumPy code block beside the plane

# Pivot gauge (3.4, thumbnail on 4.3): a vertical bar for the second pivot.
GAUGE_HEIGHT = 2.6  # screen height of a pivot of 1
GAUGE_WIDTH = 0.55
GAUGE_MIN = -0.5  # lowest pivot the gauge can show (1.2^2 = 1.44 gives -0.44)

# --- Section 1 layout (scenes/s1_problem.py) ---------------------------------

HIST_WIDTH = 10.0  # portfolio return histograms (1.1)
HIST_HEIGHT = 3.2
HEATMAP_CELL = 0.8  # side of one cell of the correlation heatmaps (1.1)
PANEL_CODE_CLOUD = 4.0  # the N(0, I) cloud beside the code (1.2)
PANEL_THUMB = 1.9  # the shrunken blue cloud in the corner (1.3)
PANEL_FORMULA = 4.4  # the zoomed cloud with points A and B (1.4)
FORMULA_BELL_WIDTH = 3.4  # the small bell curve beside the 1D formula (1.4)
FORMULA_BELL_HEIGHT = 1.4
PANEL_ROW = 1.8  # side of each plot in the two attempt rows (1.7)

# --- Section 2 layout (scenes/s2_solution.py) --------------------------------

PANEL_ROTATION = 3.2  # the rotating round cloud (2.2), below the legend
ROTATION_PANEL_CENTER = (4.3, -1.2)
TRIANGLE_SOLVE_X = -1.2  # left edge of the solving steps on 2.3
TRIANGLE_KEY_Y = -2.1  # the boxed "Cholesky: Sigma = G G^T" line on 2.3
HERO_SIDE_CENTER = (3.6, 1.1)  # x = mu + G z with G and mu, beside the hero plane (2.5)

# --- Section 3 layout (scenes/s3_sampler.py) ---------------------------------

PANEL_SWEEP = 5.0  # the single large plot of the rho sweep (slide 3.4)
VERIFY_PLANE_HEIGHT = 5.2  # 3.1 shrinks the hero plane to this height to make room
VERIFY_PLANE_CENTER = (-3.7, -0.15)
DEMO_PANEL = 2.2  # side of each plot in the row of four on slide 3.2
EIGEN_PANEL = 4.6  # the cloud on the eigen slide (3.3)
EIGEN_TEST_PANEL = 3.2  # the small plane where Sigma is applied to test arrows (3.3)
EIGEN_TEST_CENTER = (4.9, 0.75)  # top right, so the readouts fit underneath it on the left
EIGEN_READOUT_Y = -0.75  # first Sigma-times-arrow readout on 3.3
SWEEP_GAUGE_CENTER = (6.1, -0.55)  # the pivot gauge at the right edge of 3.4

# --- Section 4 layout (scenes/s4_wrapup.py) ----------------------------------

LIMITS_THUMB_SCALE = 0.5  # pivot gauge thumbnail on 4.3, relative to the 3.4 gauge
LIMITS_THUMB_X = 5.0  # x position of the thumbnails on 4.3
LIMITS_DIM_OPACITY = 0.5  # earlier limits dim when the next one appears
BOOKEND_DIM_OPACITY = 0.25  # earlier curves on 4.2 dim when the next one is drawn
LIMITS_THUMB_WIDTH = 2.4  # portfolio histogram thumbnail on 4.3
LIMITS_THUMB_HEIGHT = 1.0

# --- Points and lines --------------------------------------------------------

DOT_RADIUS = 0.028  # radius of each sample dot at full panel size
DOT_OPACITY = 0.75  # dots overlap, so partial opacity shows density
WORKED_DOT_RADIUS = 0.1  # highlighted single points (worked point, rotation marker)
WORKED_RING_RADIUS = 0.17  # ring drawn around the worked point
BULLET_RADIUS = 0.07
DOT3D_RADIUS = 0.04  # sphere radius in the 3D bonus
AXIS_STROKE = 2.0
ELLIPSE_STROKE = 4.0


def text(content: str, size: float = BODY_SIZE, color: str = COLOR_TEXT) -> Text:
    """Plain sans text in the deck font."""
    return Text(content, font=FONT, font_size=size, color=color)


def markup(content: str, size: float = BODY_SIZE, color: str = COLOR_TEXT) -> MarkupText:
    """Text with Pango markup, for bold or colored words inside a sentence."""
    return MarkupText(content, font=FONT, font_size=size, color=color)


def math(tex: str, size: float = BODY_SIZE, color: str = COLOR_TEXT, **kwargs) -> MathTex:
    """A LaTeX formula at body size by default.

    Extra keyword arguments go to MathTex, for example
    ``substrings_to_isolate`` so one symbol can be recolored.
    """
    return MathTex(tex, font_size=size * MATH_SCALE, color=color, **kwargs)


def slide_title(content: str) -> Text:
    """Title in the top-left corner, for example 'What we want'.

    Slide numbers are not shown; each section opens with its own section
    slide instead (see ``section_card``).
    """
    title = text(content, TITLE_SIZE)
    # A very long title shrinks to fit the frame instead of running off it.
    if title.width > MAX_CONTENT_WIDTH:
        title.scale_to_fit_width(MAX_CONTENT_WIDTH)
    return title.to_corner(UL, buff=EDGE_BUFF)


def section_card(number: int, name: str, subtitle: str) -> VGroup:
    """Centered section slide: a big muted number, the section name, a subtitle."""
    return VGroup(
        text(str(number), SECTION_NUMBER_SIZE, COLOR_MUTED),
        text(name, DECK_TITLE_SIZE),
        text(subtitle, BODY_SIZE, COLOR_SIGMA),
    ).arrange(DOWN, buff=0.35)


def caption(content: str, color: str = COLOR_TEXT) -> MarkupText:
    """One caption line at the bottom of the slide. Accepts Pango markup."""
    line = markup(content, CAPTION_SIZE, color)
    line.scale_to_fit_width(min(line.width, MAX_CONTENT_WIDTH))
    return line.move_to([0, CAPTION_Y, 0])


def below_title(mobject: VGroup, title: VGroup) -> VGroup:
    """Place content just under a title, left-aligned with it."""
    return mobject.next_to(title, DOWN, buff=TITLE_GAP, aligned_edge=LEFT)


def stack(*mobjects: VGroup, buff: float = 0.35) -> VGroup:
    """Left-aligned vertical stack, used for short revealed lists."""
    return VGroup(*mobjects).arrange(DOWN, buff=buff, aligned_edge=LEFT)
