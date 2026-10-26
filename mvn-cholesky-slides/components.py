"""Reusable Manim builders: axes, point clouds, ellipses, matrices, panels, marks.

Also holds ``DeckSlide``, the shared base class that gives every section the
same footer, speaker-note handling and slide-to-slide transition.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from manim import *  # noqa: F403
from manim_slides import Slide

import config as cfg
import mvn_math as mm
import theme as th

# --- Number formatting ----------------------------------------------------------


def fmt(value: float, decimals: int = cfg.DISPLAY_DECIMALS, fixed: bool = False) -> str:
    """Format a number for a slide.

    Whole numbers print without decimals ("4", not "4.00") and trailing zeros
    are dropped ("0.5"), so exact values look clean. ``fixed=True`` always
    prints ``decimals`` places, which suits measured values like a sample
    covariance.
    """
    if not fixed and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    shown = f"{value:.{decimals}f}"
    if not fixed:
        shown = shown.rstrip("0")  # "0.5", not "0.50"
    return shown[1:] if shown.startswith("-") and float(shown) == 0 else shown


def fmt_vector(values: Sequence[float], fixed: bool = False) -> str:
    """Format a vector as '(a, b)' for labels."""
    return "(" + ", ".join(fmt(v, fixed=fixed) for v in values) + ")"


def sqrt_tex(value: float) -> str:
    """LaTeX for the square root of a number: '2' for 4, '\\sqrt{2}' for 2.

    Keeps the sign, matching ``mm.elementwise_sqrt``.
    """
    root = abs(value) ** 0.5
    sign = "-" if value < 0 else ""
    if abs(root - round(root)) < 1e-9:
        return sign + fmt(root)
    return sign + r"\sqrt{" + fmt(abs(value)) + "}"


# --- Matrices -------------------------------------------------------------------


def make_matrix(
    values: Sequence[Sequence[float]] | np.ndarray,
    color: str = th.COLOR_TEXT,
    size: float = th.BODY_SIZE,
    fixed: bool = False,
) -> Matrix:
    """A bracketed numeric matrix, entries formatted with ``fmt``.

    Uses Manim's ``Matrix`` (not a single MathTex) so individual entries can be
    highlighted or greyed out with ``get_entries()``.
    """
    rows = [[fmt(v, fixed=fixed) for v in row] for row in np.asarray(values, dtype=float)]
    return make_symbolic_matrix(rows, color, size)


def make_labeled_matrix(
    symbol: str,
    values: Sequence[Sequence[float]] | np.ndarray,
    color: str,
    size: float = th.BODY_SIZE,
    fixed: bool = False,
) -> VGroup:
    """'symbol = [matrix]', for example a teal Sigma or a yellow L."""
    label = th.math(symbol + "=", size, color)
    matrix = make_matrix(values, color, size, fixed)
    return VGroup(label, matrix).arrange(RIGHT, buff=0.2)


def make_symbolic_matrix(
    entries: Sequence[Sequence[str]], color: str, size: float = th.BODY_SIZE
) -> Matrix:
    """A matrix whose entries are LaTeX strings, for formulas like L = [[1, 0], [rho, ...]].

    Manim spaces columns center to center, so the spacing is set from the widest
    entry; a fixed value leaves single digits floating far apart.
    """
    font_size = size * th.MATH_SCALE
    scale = size / th.BODY_SIZE
    widest = max(MathTex(tex, font_size=font_size).width for row in entries for tex in row)
    matrix = Matrix(
        entries,
        element_to_mobject_config={"font_size": font_size},
        h_buff=widest + 0.55 * scale,
        v_buff=0.62 * scale,
        bracket_h_buff=0.18,
        element_alignment_corner=ORIGIN,  # center entries in their column
    )
    return matrix.set_color(color)


# --- Axes and clouds ---------------------------------------------------------------


def make_axes(
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    max_width: float,
    max_height: float,
    step: float = cfg.AXIS_ROUND_STEP,
) -> Axes:
    """Axes with the same unit length on x and y, as large as the box allows.

    Equal scaling matters: with unequal scaling a round N(0, I) cloud would
    look oval and every ellipse angle would be wrong.
    """
    span_x, span_y = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
    unit = min(max_width / span_x, max_height / span_y)
    return Axes(
        x_range=[x_lim[0], x_lim[1], step],
        y_range=[y_lim[0], y_lim[1], step],
        x_length=unit * span_x,
        y_length=unit * span_y,
        tips=False,
        axis_config={"color": th.COLOR_MUTED, "stroke_width": th.AXIS_STROKE, "tick_size": 0.04},
    )


def make_square_axes(center: Sequence[float], half_width: float, size: float) -> Axes:
    """Square axes of side ``size`` covering center +/- half_width on both axes."""
    x_lim = (center[0] - half_width, center[0] + half_width)
    y_lim = (center[1] - half_width, center[1] + half_width)
    return make_axes(x_lim, y_lim, size, size)


def make_plane(
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    max_width: float,
    max_height: float,
    line_color: str = th.COLOR_GRID,
) -> NumberPlane:
    """A NumberPlane with equal x and y units (unit grid spacing), sized to fit the box."""
    span_x, span_y = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
    unit = min(max_width / span_x, max_height / span_y)
    return NumberPlane(
        x_range=[x_lim[0], x_lim[1], 1],
        y_range=[y_lim[0], y_lim[1], 1],
        x_length=unit * span_x,
        y_length=unit * span_y,
        background_line_style={"stroke_color": line_color, "stroke_width": 1.5},
        axis_config={"color": th.COLOR_MUTED, "stroke_width": th.AXIS_STROKE},
        faded_line_ratio=1,
    )


def make_density_axes(
    x_lim: tuple[float, float], y_max: float, width: float, height: float
) -> Axes:
    """Axes for 1D bell curves: numbered x axis, bare y axis (heights are not the point)."""
    axes = Axes(
        x_range=[x_lim[0], x_lim[1], 1],
        y_range=[0, y_max, y_max],
        x_length=width,
        y_length=height,
        tips=False,
        axis_config={"color": th.COLOR_MUTED, "stroke_width": th.AXIS_STROKE},
        y_axis_config={"include_ticks": False},
    )
    even_ticks = [x for x in range(int(x_lim[0]), int(x_lim[1]) + 1) if x % 2 == 0]
    axes.x_axis.add_numbers(even_ticks, font_size=th.MIN_SIZE * th.MATH_SCALE, color=th.COLOR_MUTED)
    return axes


def axis_labels(axes: Axes, x_label: str, y_label: str) -> VGroup:
    """Small muted names under the x axis and beside the y axis."""
    x_text = th.text(x_label, th.MIN_SIZE, th.COLOR_MUTED).next_to(axes, DOWN, buff=0.1)
    y_text = th.text(y_label, th.MIN_SIZE, th.COLOR_MUTED).rotate(PI / 2)
    return VGroup(x_text, y_text.next_to(axes, LEFT, buff=0.1))


def unit_length(axes: Axes) -> float:
    """Screen length of one data unit (equal on both axes by construction)."""
    return float(axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0])


def make_point_cloud(
    samples: np.ndarray,
    axes: Axes,
    color: str,
    radius: float = th.DOT_RADIUS,
) -> VGroup:
    """One Dot per row of ``samples`` (only the first two columns are used).

    Callers pass at most ``N_DISPLAY`` rows; thousands of Dots make Manim slow.
    """
    return VGroup(
        *[
            Dot(axes.c2p(x, y), radius=radius, color=color, fill_opacity=th.DOT_OPACITY)
            for x, y in samples[:, :2]
        ]
    )


def make_cov_ellipse(
    sigma: Sequence[Sequence[float]],
    mu: Sequence[float],
    k: float,
    axes: Axes,
    color: str = th.COLOR_SIGMA,
) -> Ellipse:
    """The k-sigma contour of N(mu, Sigma), drawn on ``axes``."""
    width, height, angle = mm.covariance_ellipse_params(sigma, k)
    unit = unit_length(axes)
    ellipse = Ellipse(width=width * unit, height=height * unit, color=color)
    ellipse.set_stroke(width=th.ELLIPSE_STROKE)
    return ellipse.rotate(angle).move_to(axes.c2p(*mu[:2]))


def make_bell_curve(axes: Axes, mu: float, sd: float, color: str) -> ParametricFunction:
    """Graph of the N(mu, sd^2) density across the full x range of ``axes``."""
    return axes.plot(
        lambda x: float(mm.normal_pdf_1d(x, mu, sd)),
        x_range=[axes.x_range[0], axes.x_range[1], 0.05],
        color=color,
        stroke_width=5,
    )


def make_mean_sd_guides(
    axes: Axes,
    mu: float,
    sd: float,
    color: str = th.COLOR_MUTED,
    show_mean: bool = True,
) -> VGroup:
    """Mean and one-standard-deviation markers for a bell curve on ``axes``.

    A dashed line at the mean (labeled mu = ...), dashed lines at mu - sd and
    mu + sd up to the curve, and a double arrow from mu to mu + sd labeled
    sigma = .... Rebuild with new values and Transform to move them with the curve.
    ``show_mean=False`` drops the mean line, for a second curve with the same mean.
    """
    peak = float(mm.normal_pdf_1d(mu, mu, sd))
    at_sd = float(mm.normal_pdf_1d(mu + sd, mu, sd))  # curve height one sd from the mean
    mean_line = DashedLine(axes.c2p(mu, 0), axes.c2p(mu, peak), color=th.COLOR_TEXT)
    sd_lines = VGroup(
        *[DashedLine(axes.c2p(x, 0), axes.c2p(x, at_sd), color=color) for x in (mu - sd, mu + sd)]
    )
    arrow = DoubleArrow(
        axes.c2p(mu, at_sd),
        axes.c2p(mu + sd, at_sd),
        buff=0,
        color=color,
        stroke_width=3,
        tip_length=0.12,
    )
    # Up and to the right, so the y axis (which the line sits on when mu = 0)
    # does not run through the label.
    mean_label = th.math(rf"\mu = {fmt(mu)}", th.MIN_SIZE)
    mean_label.next_to(mean_line.get_end(), UR, buff=0.08)
    # Below the arrow: above it the label would touch the curve once sd grows.
    sd_text_color = th.COLOR_TEXT if color == th.COLOR_MUTED else color
    sd_label = th.math(rf"\sigma = {fmt(sd)}", th.MIN_SIZE, sd_text_color)
    sd_label.next_to(arrow, DOWN, buff=0.08)
    for label in (mean_label, sd_label):  # stay readable when another curve crosses them
        label.add_background_rectangle(color=th.BACKGROUND, opacity=0.9, buff=0.03)
    if not show_mean:
        return VGroup(sd_lines, arrow, sd_label)
    return VGroup(mean_line, sd_lines, arrow, mean_label, sd_label)


class Panel(VGroup):
    """A small labeled scatter plot: title, square axes, cloud, and its Sigma.

    ``z`` is passed in (not drawn here) so several panels can share the same
    standard normal samples and differ only in Sigma.
    """

    def __init__(
        self,
        label: str,
        sigma: Sequence[Sequence[float]],
        z: np.ndarray,
        half_width: float,
        size: float,
        color: str = th.COLOR_X,
        matrix_side: np.ndarray | None = DOWN,
    ) -> None:
        super().__init__()
        self.label = th.text(label, th.CAPTION_SIZE)
        self.axes = make_square_axes((0.0, 0.0), half_width, size)
        samples = mm.transform(z, np.zeros(2), mm.cholesky_factor(sigma))
        self.cloud = make_point_cloud(samples, self.axes, color, panel_dot_radius(size))
        self.label.next_to(self.axes, UP, buff=0.15)
        self.add(self.label, self.axes, self.cloud)
        if matrix_side is not None:
            self.matrix = make_labeled_matrix(r"\Sigma", sigma, th.COLOR_SIGMA, th.MIN_SIZE)
            self.matrix.next_to(self.axes, matrix_side, buff=0.25)
            self.add(self.matrix)


def panel_dot_radius(size: float) -> float:
    """Shrink dots a little on small plots so the cloud does not turn into a blob."""
    return th.DOT_RADIUS * min(1.0, max(0.6, size / th.PANEL_LARGE))


def fade_in_cloud(cloud: VGroup, run_time: float = cfg.T_MEDIUM) -> Animation:
    """Dots appear in a quick ripple rather than all at once."""
    return FadeIn(cloud, lag_ratio=0.002, run_time=run_time)


# --- Code -----------------------------------------------------------------------


def make_code(snippet: str) -> Code:
    """A Python snippet in the deck's code style: monospace on a dark panel."""
    return Code(
        code_string=snippet,
        language="python",
        formatter_style="monokai",
        add_line_numbers=False,
        background="rectangle",
        background_config={"fill_color": th.COLOR_PANEL, "stroke_color": th.COLOR_MUTED},
        paragraph_config={"font": th.CODE_FONT, "font_size": th.MIN_SIZE},
    )


# --- Marks and arrows ------------------------------------------------------------------


def make_cross_mark(size: float = 0.5, color: str = th.COLOR_BAD) -> VGroup:
    """A red X drawn with lines (font-independent, unlike a glyph)."""
    half = size / 2
    return VGroup(
        Line([-half, -half, 0], [half, half, 0]),
        Line([-half, half, 0], [half, -half, 0]),
    ).set_stroke(color, width=8)


def make_check_mark(size: float = 0.5, color: str = th.COLOR_GOOD) -> VMobject:
    """A green check mark drawn with lines."""
    mark = VMobject().set_points_as_corners(
        [[-0.5 * size, 0, 0], [-0.15 * size, -0.35 * size, 0], [0.5 * size, 0.4 * size, 0]]
    )
    return mark.set_stroke(color, width=8)


def labeled_arrow(start: np.ndarray, end: np.ndarray, label: Mobject, color: str) -> VGroup:
    """An arrow with a label sitting just above its midpoint."""
    arrow = Arrow(start, end, color=color, buff=0.1, stroke_width=6)
    label.next_to(arrow, UP, buff=0.1)
    return VGroup(arrow, label)


def boxed(content: Mobject, color: str = th.COLOR_MUTED, buff: float = 0.3) -> VGroup:
    """Content on a rounded, filled box, used for the algorithm and summaries."""
    box = SurroundingRectangle(content, buff=buff, corner_radius=0.15, color=color)
    box.set_fill(th.COLOR_PANEL, opacity=1.0).set_stroke(width=2)
    return VGroup(box, content)


# --- Shared slide base ---------------------------------------------------------------


class DeckSlide(Slide):
    """Base class for the four sections.

    Gives every section the same four things:
      * a footer on the manim-slides canvas, so it survives transitions;
      * ``show_section_slide(subtitle)``, the opening slide of the section;
      * ``begin_slide(notes)``, which starts a slide, attaches its notes and
        fades out the previous slide (the one transition used everywhere);
      * ``reveal()``, which adds a pause inside a slide, keeping the notes.
    """

    section_number: int = 0  # each section sets these two, for example 1 and "Problem";
    section_name: str = ""  # they feed both the footer and the section slide

    def setup(self) -> None:
        super().setup()
        # Hold the final frame briefly so each slide ends on the exact final
        # state (e.g. rho = -0.900, not the -0.898 of the last in-between frame).
        self.wait_time_between_slides = cfg.T_SETTLE
        self.current_notes = ""
        # One shared draw of N(0, I). Displayed dots are the first N_DISPLAY rows
        # of the N_STATS sample, so the statistics describe the dots we see.
        self.z_stats = mm.standard_normal_samples(cfg.N_STATS, 2, cfg.RANDOM_SEED)
        self.z_display = self.z_stats[: cfg.N_DISPLAY]
        self.main_chol = mm.cholesky_factor(cfg.MAIN_SIGMA)
        if self.section_name:
            self.add_to_canvas(footer=th.footer(f"{self.section_number}  {self.section_name}"))
            self.add(self.canvas["footer"])

    def begin_slide(self, notes: str) -> None:
        """Start a new slide.

        In manim-slides, the options passed to ``next_slide`` (including notes)
        belong to the slide that starts after the call. So the call goes at the
        start of each slide, not the end, or every note would land one slide late.
        """
        self.next_slide(notes=notes)
        self.current_notes = notes
        leftovers = self.mobjects_without_canvas
        if leftovers:
            self.play(FadeOut(*leftovers), run_time=cfg.T_SHORT)

    def show_section_slide(self, subtitle: str) -> None:
        """Draw the section card (number, name, subtitle) for the current slide."""
        number, name, line = th.section_card(self.section_number, self.section_name, subtitle)
        self.play(FadeIn(number, shift=0.3 * DOWN), Write(name), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(line, shift=0.2 * UP), run_time=cfg.T_SHORT)

    def reveal(self) -> None:
        """Pause for a keypress inside the current slide (same speaker notes)."""
        self.next_slide(notes=self.current_notes)
