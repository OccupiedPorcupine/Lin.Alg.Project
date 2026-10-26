"""Colors, font sizes, layout constants, and the title, caption and footer helpers.

Colors have a fixed meaning across the whole deck (see the semantic block
below). Change a color here and every slide follows.
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
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
COLOR_L = "#ffd23f"  # L, the Cholesky factor: yellow
COLOR_Z1 = COLOR_Z  # z1 component on the mixing slide: blue
COLOR_Z2 = "#b48cff"  # z2 component on the mixing slide: purple
COLOR_BAD = "#ff5c5c"  # errors and failures: red
COLOR_GOOD = "#4cd964"  # confirmations: green

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
FOOTER_SIZE = MIN_SIZE

# --- Layout ------------------------------------------------------------------

EDGE_BUFF = 0.45  # distance from the frame edge for titles and footers
TITLE_GAP = 0.35  # gap between the title and the content below it
CAPTION_Y = -3.05  # vertical position of the caption line at the bottom
CONTENT_TOP = 2.6  # content should stay below this y
CONTENT_BOTTOM = -2.6  # and above this y, so it never meets the caption
MAX_CONTENT_WIDTH = FRAME_WIDTH - 2 * EDGE_BUFF

PANEL_LARGE = 4.4  # side of a single main scatter plot
PANEL_MEDIUM = 3.5  # side of each plot in a side-by-side pair
PANEL_SMALL = 2.0  # side of each plot in the 2x2 grid (slide 3.2)
PANEL_TARGET = 4.1  # the target cloud beside the density formula (slide 1.3)
PANEL_ROTATION = 3.8  # the rotating round cloud (slide 2.2)
PANEL_SWEEP = 5.0  # the single large plot of the rho sweep (slide 3.3)
HERO_MAX_WIDTH = 7.0  # box the hero NumberPlane must fit in (slide 2.6)
HERO_MAX_HEIGHT = 6.5
DENSITY_WIDTH = 11.5  # 1D bell-curve axes (slide 1.6)
DENSITY_HEIGHT = 2.9
PANEL_ROW = 1.8  # side of each plot in the two attempt rows on slide 1.8
PANEL_STRIP = 3.0  # side of each plot in the recap strip
BAR_WIDTH = 8.6  # full width of the stacked variance bar (slide 2.5)
BAR_HEIGHT = 0.55

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


def footer(content: str) -> Text:
    """Small section indicator for the bottom-right corner (kept on the canvas)."""
    label = text(content, FOOTER_SIZE, COLOR_MUTED)
    return label.to_edge(DOWN, buff=0.2).to_edge(RIGHT, buff=0.3)


def below_title(mobject: VGroup, title: VGroup) -> VGroup:
    """Place content just under a title, left-aligned with it."""
    return mobject.next_to(title, DOWN, buff=TITLE_GAP, aligned_edge=LEFT)


def stack(*mobjects: VGroup, buff: float = 0.35) -> VGroup:
    """Left-aligned vertical stack, used for short revealed lists."""
    return VGroup(*mobjects).arrange(DOWN, buff=buff, aligned_edge=LEFT)
