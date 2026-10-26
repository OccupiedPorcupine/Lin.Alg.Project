"""Reusable Manim builders: axes, point clouds, ellipses, matrices, panels, marks,
plus the pieces shared between sections (Gaussian process curves, histograms,
the pivot gauge and the hero frame).

Also holds ``DeckSlide``, the shared base class that gives every section the
same section slide, speaker-note handling and slide-to-slide transition.
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
    """'symbol = [matrix]', for example a teal Sigma or a yellow G."""
    label = th.math(symbol + "=", size, color)
    matrix = make_matrix(values, color, size, fixed)
    return VGroup(label, matrix).arrange(RIGHT, buff=0.2)


def make_symbolic_matrix(
    entries: Sequence[Sequence[str]], color: str, size: float = th.BODY_SIZE
) -> Matrix:
    """A matrix whose entries are LaTeX strings, for formulas like G = [[1, 0], [rho, ...]].

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


def swap_caption(old: Mobject, new: Mobject) -> Succession:
    """Fade the old caption out, then the new one in, in the same total time.

    Captions sit in the same spot, so fading both at once stacks two
    sentences on top of each other for the whole fade.
    """
    return Succession(FadeOut(old), FadeIn(new))


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


def make_dim_overlay(opacity: float = 0.85) -> FullScreenRectangle:
    """Near-black sheet over the whole frame. Manim has no real blur, so this
    fakes one: everything behind it dims and new text on top stands out."""
    overlay = FullScreenRectangle()
    return overlay.set_fill(th.BACKGROUND, opacity=opacity).set_stroke(width=0)


# --- Gaussian process panel (1.1, replayed on 4.2) --------------------------------


def make_gp_axes() -> Axes:
    """Axes for the Gaussian process curves: x over GP_X_RANGE, y over N(0, 1) values.

    The y range holds standard normal draws out to AXIS_N_SD, so the jagged
    dots (which are N(0, 1)) never leave the axes, and neither do the curves.
    """
    y_lim = mm.axis_limits_1d([0.0], [1.0], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
    return Axes(
        x_range=[cfg.GP_X_RANGE[0], cfg.GP_X_RANGE[1], 1],
        y_range=[y_lim[0], y_lim[1], 1],
        x_length=th.GP_WIDTH,
        y_length=th.GP_HEIGHT,
        tips=False,
        axis_config={"color": th.COLOR_MUTED, "stroke_width": th.AXIS_STROKE},
        x_axis_config={"include_ticks": False},
        y_axis_config={"include_ticks": False},
    )


def gp_points(values: np.ndarray, axes: Axes) -> list[np.ndarray]:
    """Screen positions of one curve's values at the GP grid's x positions."""
    xs = mm.gp_grid(cfg.GP_X_RANGE, cfg.GP_N_POINTS)
    return [axes.c2p(x, y) for x, y in zip(xs, values, strict=True)]


def make_gp_dots(values: np.ndarray, axes: Axes, color: str = th.COLOR_Z) -> VGroup:
    """One dot per grid point: the 60 values of one draw, before or after G."""
    return VGroup(*[Dot(p, radius=th.GP_DOT_RADIUS, color=color) for p in gp_points(values, axes)])


def make_gp_jagged_line(values: np.ndarray, axes: Axes, color: str = th.COLOR_Z) -> VMobject:
    """Straight segments joining the dots, which shows how jagged independent draws are."""
    line = VMobject().set_points_as_corners(gp_points(values, axes))
    return line.set_stroke(color, width=2, opacity=0.6)


def make_gp_curve(values: np.ndarray, axes: Axes, color: str = th.COLOR_X) -> VMobject:
    """A smooth curve through the values (used for x = G z, which is smooth)."""
    curve = VMobject().set_points_smoothly(gp_points(values, axes))
    return curve.set_stroke(color, width=5)


def make_gp_states(z_row: np.ndarray, x_row: np.ndarray, axes: Axes) -> tuple[VGroup, VGroup]:
    """Start and end states of one jagged-to-smooth morph: Transform(start, end).

    Start: the 60 independent values z (blue dots joined by a faint jagged
    line). End: the same dots moved to x = G z (orange) on a smooth curve.
    Slides 1.1 and 4.2 both animate exactly this, so 4.2 is a true replay.
    """
    start = VGroup(make_gp_jagged_line(z_row, axes), make_gp_dots(z_row, axes, th.COLOR_Z))
    end = VGroup(make_gp_curve(x_row, axes), make_gp_dots(x_row, axes, th.COLOR_X))
    return start, end


# --- Histograms (1.1 portfolio panel, 4.3 thumbnail) --------------------------------


def make_histogram(
    values: np.ndarray, axes: Axes, bins: int, color: str, opacity: float = 0.45
) -> VGroup:
    """Density-scaled histogram drawn as bars on ``axes`` (over its full x range).

    Density scaling (area 1) lets two histograms of different spread share
    one y axis fairly, the same way two bell curves would.
    """
    heights, edges = np.histogram(values, bins=bins, range=axes.x_range[:2], density=True)
    bars = VGroup()
    for left, right, height in zip(edges[:-1], edges[1:], heights, strict=True):
        bottom_left, top_right = axes.c2p(left, 0), axes.c2p(right, height)
        bar = Rectangle(
            width=top_right[0] - bottom_left[0], height=max(top_right[1] - bottom_left[1], 1e-3)
        )
        bar.move_to(bottom_left, aligned_edge=DL)
        bars.add(bar)
    return bars.set_fill(color, opacity).set_stroke(color, width=1.5, opacity=0.9)


# --- Pivot gauge (3.4, thumbnail on 4.3) ------------------------------------------------


def pivot_color(pivot: float) -> str:
    """Green while the pivot is positive, amber at zero, red below zero."""
    if abs(pivot) < 1e-9:
        return th.COLOR_WARN
    return th.COLOR_GOOD if pivot > 0 else th.COLOR_BAD


def make_pivot_gauge(pivot: float, center: np.ndarray = ORIGIN, scale: float = 1.0) -> VGroup:
    """A vertical bar gauge for the second pivot 1 - rho^2.

    The frame always spans GAUGE_MIN to 1, so its position never moves while
    the bar inside grows, shrinks or drops below the zero line. Rebuild it
    each frame (always_redraw) with the current pivot and the same ``center``.
    Returns VGroup(frame, zero_line, bar, labels).
    """
    unit = th.GAUGE_HEIGHT * scale
    width = th.GAUGE_WIDTH * scale
    frame = Rectangle(width=width, height=(1 - th.GAUGE_MIN) * unit)
    frame.set_stroke(th.COLOR_MUTED, width=2)
    frame.move_to(center)
    zero_y = frame.get_bottom()[1] - th.GAUGE_MIN * unit
    zero_line = Line(
        [frame.get_left()[0] - 0.12 * scale, zero_y, 0],
        [frame.get_right()[0] + 0.12 * scale, zero_y, 0],
    ).set_stroke(th.COLOR_TEXT, width=3)
    shown = float(np.clip(pivot, th.GAUGE_MIN, 1.0))
    bar = Rectangle(width=0.8 * width, height=max(abs(shown) * unit, 0.02 * scale))
    bar.set_fill(pivot_color(pivot), opacity=0.9).set_stroke(width=0)
    bar.move_to([center[0], zero_y, 0], aligned_edge=DOWN if shown >= 0 else UP)
    labels = VGroup(
        th.math("1", th.MIN_SIZE * scale, th.COLOR_MUTED).next_to(
            frame.get_corner(UL), LEFT, buff=0.1
        ),
        th.math("0", th.MIN_SIZE * scale, th.COLOR_MUTED).next_to(zero_line, LEFT, buff=0.1),
    )
    return VGroup(frame, zero_line, bar, labels)


# --- Hero frame (2.5 ends on it, 3.1 starts from it) ---------------------------------

HERO_NUMPY_SNIPPET = """\
G = np.linalg.cholesky(Sigma)
Z = rng.standard_normal((n, 2))
X = mu + Z @ G.T"""


def make_hero_plane() -> NumberPlane:
    """The fixed background plane of the hero animation.

    Its range covers all three clouds of the animation: z at 0, G z at 0
    (wider), and mu + G z at mu.
    """
    x_lim, y_lim = mm.axis_limits(
        [np.zeros(2), np.zeros(2), cfg.MAIN_MU],
        [np.eye(2), cfg.MAIN_SIGMA, cfg.MAIN_SIGMA],
        cfg.AXIS_N_SD,
        cfg.AXIS_ROUND_STEP,
    )
    plane = make_plane(x_lim, y_lim, th.HERO_MAX_WIDTH, th.HERO_MAX_HEIGHT)
    return plane.move_to([*th.HERO_PLANE_CENTER, 0])


def make_hero_grid(plane: NumberPlane) -> NumberPlane:
    """The bright grid over z in [-HERO_GRID_HALF_WIDTH, HERO_GRID_HALF_WIDTH]^2 that G moves."""
    extent = cfg.HERO_GRID_HALF_WIDTH
    unit = plane.get_x_unit_size()
    grid = NumberPlane(
        x_range=[-extent, extent, 1],
        y_range=[-extent, extent, 1],
        x_length=2 * extent * unit,
        y_length=2 * extent * unit,
        background_line_style={
            "stroke_color": th.COLOR_TEXT,
            "stroke_opacity": 0.6,
            "stroke_width": 2,
        },
        axis_config={"stroke_color": th.COLOR_TEXT, "stroke_opacity": 0.6},
    )
    return grid.shift(plane.c2p(0, 0) - grid.c2p(0, 0))


def through_data(plane: NumberPlane, fn):
    """Turn a map on data coordinates into a map on screen points.

    The hero plane is not unit-scale, so ApplyMatrix (which acts on screen
    coordinates) would be wrong. Instead: screen -> plane.p2c -> fn -> plane.c2p.
    """
    return lambda point: plane.c2p(*fn(np.array(plane.p2c(point))))


def make_worked_marker(position: np.ndarray) -> VGroup:
    """The highlighted worked point: a white dot with a ring around it."""
    dot = Dot(position, radius=th.WORKED_DOT_RADIUS, color=th.COLOR_TEXT)
    ring = Circle(radius=th.WORKED_RING_RADIUS, color=th.COLOR_TEXT, stroke_width=3)
    return VGroup(dot, ring.move_to(position))


def make_point_label(tex: str, color: str, anchor: Mobject) -> VGroup:
    """A math label on a dark backing so it stays readable over the dots."""
    label = th.math(tex, th.CAPTION_SIZE, color)
    label.add_background_rectangle(color=th.BACKGROUND, opacity=0.95, buff=0.06)
    return label.next_to(anchor, UR, buff=0.1)


def make_coordinate_guides(plane: NumberPlane, point: np.ndarray) -> VGroup:
    """Dashed lines from the point to both axes, labeled with its coordinates."""
    to_x_axis = DashedLine(plane.c2p(*point), plane.c2p(point[0], 0), color=th.COLOR_X)
    to_y_axis = DashedLine(plane.c2p(*point), plane.c2p(0, point[1]), color=th.COLOR_X)
    x_value = th.math(fmt(point[0]), th.MIN_SIZE, th.COLOR_X)
    x_value.next_to(plane.c2p(point[0], 0), DOWN, buff=0.12)
    y_value = th.math(fmt(point[1]), th.MIN_SIZE, th.COLOR_X)
    y_value.next_to(plane.c2p(0, point[1]), LEFT, buff=0.12)
    for value in (x_value, y_value):
        value.add_background_rectangle(color=th.BACKGROUND, opacity=0.85, buff=0.04)
    return VGroup(to_x_axis, to_y_axis, x_value, y_value)


def make_hero_code() -> VGroup:
    """'In NumPy' and the three-line sampler, to the right of the hero plane."""
    heading = th.text("In NumPy", th.CAPTION_SIZE, th.COLOR_MUTED)
    block = VGroup(heading, make_code(HERO_NUMPY_SNIPPET)).arrange(
        DOWN, buff=0.25, aligned_edge=LEFT
    )
    return block.to_edge(RIGHT, buff=th.EDGE_BUFF).set_y(th.HERO_CODE_Y)


class HeroFrame(VGroup):
    """The last frame of slide 2.5, rebuilt from scratch.

    Slide 2.5 animates its way to this state; Section 3 opens by adding it
    instantly, so the orange cloud carries across the section boundary. Both
    use this class, so the two frames cannot drift apart. Draw order is the
    order of the submobjects: plane, grid, cloud, marker, label, guides, code.
    """

    def __init__(self, z_display: np.ndarray, chol: np.ndarray) -> None:
        super().__init__()
        mu = np.asarray(cfg.MAIN_MU, dtype=float)
        self.plane = make_hero_plane()
        self.grid = make_hero_grid(self.plane)
        self.grid.apply_function(through_data(self.plane, lambda data: chol @ data + mu))
        self.cloud = make_point_cloud(mm.transform(z_display, mu, chol), self.plane, th.COLOR_X)
        self.worked_x = mm.transform(cfg.WORKED_Z, mu, chol)
        self.marker = make_worked_marker(self.plane.c2p(*self.worked_x))
        self.label = make_point_label(
            r"\mathbf{x} = " + fmt_vector(self.worked_x), th.COLOR_X, self.marker[0]
        )
        self.guides = make_coordinate_guides(self.plane, self.worked_x)
        self.code = make_hero_code()
        self.add(self.plane, self.grid, self.cloud, self.marker, self.label, self.guides, self.code)


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

    Gives every section the same three things:
      * ``show_section_slide(subtitle)``, the opening slide of the section;
      * ``begin_slide(notes)``, which starts a slide, attaches its notes and
        fades out the previous slide (the one transition used everywhere);
      * ``reveal()``, which adds a pause inside a slide, keeping the notes.
    """

    section_number: int = 0  # each section sets these two, for example 1 and "Problem";
    section_name: str = ""  # they appear on the section's opening slide

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
        self.main_chol = mm.cholesky_factor(cfg.MAIN_SIGMA)  # G for the main example
        # Gaussian process draws: each row of gp_z is 60 independent values and
        # the same row of gp_x is the smooth curve G z. Sections 1 and 4 both
        # use these, so 4.2 replays exactly the curves of 1.1.
        gp_xs = mm.gp_grid(cfg.GP_X_RANGE, cfg.GP_N_POINTS)
        self.gp_z = mm.standard_normal_samples(cfg.GP_N_CURVES, cfg.GP_N_POINTS, cfg.RANDOM_SEED)
        self.gp_x = mm.gp_curves(self.gp_z, gp_xs, cfg.GP_LENGTH_SCALE, cfg.GP_JITTER)

    def begin_slide(self, notes: str) -> None:
        """Start a new slide.

        In manim-slides, the options passed to ``next_slide`` (including notes)
        belong to the slide that starts after the call. So the call goes at the
        start of each slide, not the end, or every note would land one slide late.
        """
        self.next_slide(notes=notes)
        self.current_notes = notes
        leftovers = self.mobjects
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
