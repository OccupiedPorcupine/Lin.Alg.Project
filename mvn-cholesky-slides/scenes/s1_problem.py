"""Section 1: the problem.

Computers give us independent noise N(0, I); we want correlated data
N(mu, Sigma). This section shows where that matters, reads the density
formula, and shows why the obvious "square roots" fail.
"""

import sys
from pathlib import Path

# manim-slides loads this file by path, so the project root is not importable
# by default. Put it first on sys.path so config, theme, ... resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from manim import *

import config as cfg
import mvn_math as mm
import theme as th
from components import (
    DeckSlide,
    Panel,
    axis_labels,
    fade_in_cloud,
    fmt,
    make_bell_curve,
    make_check_mark,
    make_code,
    make_cov_ellipse,
    make_cross_mark,
    make_density_axes,
    make_dim_overlay,
    make_histogram,
    make_labeled_matrix,
    make_matrix,
    make_point_cloud,
    make_square_axes,
    make_symbolic_matrix,
    panel_dot_radius,
    sqrt_tex,
    swap_caption,
)

IDENTITY = np.eye(2)

# Pieces of the density formula on slide 1.4, with the role that colors them.
# Each row is (1D rearranged, nD); they sit in the same column on screen.
FORMULA_1D = [
    "f(x) =",
    r"\frac{1}{\sqrt{2\pi\sigma^2}}",
    r"\exp\Big(",
    r"-\frac{(x-\mu)^2}{2\sigma^2}",
    r"\Big)",
]
FORMULA_1D_REARRANGED = [
    "f(x) =",
    r"(2\pi)^{-1/2}",
    r"(\sigma^2)^{-1/2}",
    r"\exp\Big(",
    r"-\tfrac{1}{2}",
    r"(x-\mu)",
    r"(\sigma^2)^{-1}",
    r"(x-\mu)",
    r"\Big)",
]
FORMULA_ND = [
    r"f(\mathbf{x}) =",
    r"(2\pi)^{-k/2}",
    r"|\Sigma|^{-1/2}",
    r"\exp\Big(",
    r"-\tfrac{1}{2}",
    r"(\mathbf{x}-\boldsymbol{\mu})^T",
    r"\Sigma^{-1}",
    r"(\mathbf{x}-\boldsymbol{\mu})",
    r"\Big)",
]
# Role of each piece of the rearranged and nD formulas (same order as above).
FORMULA_ROLES = [None, "norm", "norm", None, "dist", "dist", "spread", "dist", None]
ROLE_COLORS = {"norm": th.COLOR_NORM, "dist": th.COLOR_DIST, "spread": th.COLOR_SIGMA}


class S1Problem(DeckSlide):
    """Section 1: where it matters, what we have, what we want, and why naive fixes fail."""

    section_number = 1
    section_name = "Problem"

    def construct(self) -> None:
        self.slide_1_0_title()
        self.slide_1_0_section()
        self.slide_1_1_hook()
        self.slide_1_2_what_computers_give()
        self.slide_1_3_what_we_want()
        self.slide_1_4_reading_the_formula()
        self.slide_1_5_the_question()
        self.slide_1_6_one_d_warmup()
        self.slide_1_7_nd_catch()

    # --- 1.0 -----------------------------------------------------------------

    def slide_1_0_title(self) -> None:
        """Title slide. Point: name the topic and the method."""
        self.begin_slide(
            notes="""
            Hi everyone. Our project is about simulating random vectors that
            follow a multivariate normal distribution with a covariance matrix
            we choose. The tool that makes it work is the Cholesky decomposition,
            and at the end we will play with different covariance structures.
            """
        )
        title = VGroup(
            th.text("Simulating the Multivariate Normal", th.DECK_TITLE_SIZE),
            th.text("with a Given Covariance Matrix", th.DECK_TITLE_SIZE),
        ).arrange(DOWN, buff=0.25)
        subtitle = th.text(
            "Cholesky decomposition and covariance structures", th.BODY_SIZE, th.COLOR_SIGMA
        )
        presenter = th.text(cfg.PRESENTER_NAME, th.CAPTION_SIZE, th.COLOR_MUTED)
        VGroup(title, subtitle, presenter).arrange(DOWN, buff=0.55)

        self.play(Write(title), run_time=cfg.T_LONG)
        self.play(FadeIn(subtitle, shift=0.2 * UP), FadeIn(presenter), run_time=cfg.T_MEDIUM)

    def slide_1_0_section(self) -> None:
        """Section slide: number, name and a one-line preview of the section."""
        self.begin_slide(
            notes="""
            Let's start with the problem. We will see where correlated
            randomness matters, what a computer can give us, what we actually
            want, and why the obvious fixes fail.
            """
        )
        self.show_section_slide("What we have, what we want, and why it is not obvious")

    # --- 1.1 -----------------------------------------------------------------

    def slide_1_1_hook(self) -> None:
        """Two uses, one per click. Point: both need random numbers that move together."""
        self.begin_slide(
            notes="""
            Before any math, here are two places where correlated randomness
            matters. A bank simulating a portfolio underestimates its risk if it
            treats the stocks as independent, and synthetic data is only useful
            if it keeps the real correlations. Both need random numbers that
            move together.
            """
        )
        self._hook_portfolio()
        self.reveal()
        self.play(FadeOut(*self.mobjects), run_time=cfg.T_SHORT)
        self._hook_synthetic_data()
        self.reveal()

        message = VGroup(
            th.text("Both need random numbers", th.TITLE_SIZE),
            th.text("that move together.", th.TITLE_SIZE),
        ).arrange(DOWN, buff=0.25)
        self.play(FadeIn(make_dim_overlay()), run_time=cfg.T_SHORT)
        self.play(Write(message), run_time=cfg.T_MEDIUM)

    def _hook_portfolio(self) -> None:
        """Two scatter plots of stock returns morph into portfolio return histograms."""
        title = th.slide_title("Monte Carlo portfolio risk")
        heading = th.text("Stock A vs Stock B returns (simulated)", th.CAPTION_SIZE, th.COLOR_MUTED)
        heading.next_to(title, DOWN, buff=th.TITLE_GAP).set_x(0)

        rhos = (cfg.PORTFOLIO_RHO_INDEPENDENT, cfg.PORTFOLIO_RHO_CORRELATED)
        colors = (th.COLOR_Z, th.COLOR_X)
        names = ("Independent", "Correlated")
        sigmas = [mm.correlation_matrix(rho) for rho in rhos]
        half_width = mm.shared_half_width(sigmas, cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        panels = VGroup()
        for name, rho, sigma, color in zip(names, rhos, sigmas, colors, strict=True):
            panel = Panel(
                f"{name}: ρ = {fmt(rho)}",
                sigma,
                self.z_display,
                half_width,
                th.PANEL_MEDIUM,
                color,
                None,
            )
            panel.labels = axis_labels(panel.axes, "Stock A", "Stock B")
            panel.add(panel.labels)
            panels.add(panel)
        panels.arrange(RIGHT, buff=1.8).next_to(heading, DOWN, buff=0.25)

        self.play(FadeIn(title), FadeIn(heading), run_time=cfg.T_SHORT)
        self.play(
            *[FadeIn(VGroup(p.axes, p.label, p.labels)) for p in panels], run_time=cfg.T_SHORT
        )
        self.play(*[fade_in_cloud(p.cloud) for p in panels], run_time=cfg.T_MEDIUM)
        self.reveal()

        # Same z for both rho values, so the histograms differ only by correlation.
        returns = [
            mm.portfolio_returns(cfg.PORTFOLIO_WEIGHTS, rho, cfg.N_STATS, cfg.RANDOM_SEED)
            for rho in rhos
        ]
        spreads = [float(np.std(r)) for r in returns]
        x_lim = mm.axis_limits_1d([0.0, 0.0], spreads, cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        tallest = max(
            np.histogram(r, bins=cfg.PORTFOLIO_HIST_BINS, range=x_lim, density=True)[0].max()
            for r in returns
        )
        axes = make_density_axes(x_lim, 1.15 * tallest, th.HIST_WIDTH, th.HIST_HEIGHT)
        axes.y_axis.set_stroke(opacity=0)  # bar heights are not the point, only the spread
        level = f"{cfg.VAR_LEVEL:.0%}".replace("%", r"\%")
        var_values = [mm.value_at_risk(r, cfg.VAR_LEVEL) for r in returns]
        legend = VGroup(
            *[
                th.math(
                    rf"\rho = {fmt(rho)}:\ \ \text{{sd }} {fmt(spread, fixed=True)},"
                    rf"\ \ {level}\ \text{{VaR }} {{{fmt(var, fixed=True)}}}",
                    th.CAPTION_SIZE,
                    color,
                )
                for rho, spread, var, color in zip(rhos, spreads, var_values, colors, strict=True)
            ]
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        legend.next_to(heading, DOWN, buff=0.3)
        axes.next_to(legend, DOWN, buff=0.3).set_x(0)
        histograms = [
            make_histogram(r, axes, cfg.PORTFOLIO_HIST_BINS, color)
            for r, color in zip(returns, colors, strict=True)
        ]
        var_lines = VGroup(
            *[
                DashedLine(axes.c2p(var, 0), axes.c2p(var, 0.8 * tallest), color=th.COLOR_BAD)
                for var in var_values
            ]
        )
        var_label = th.text(f"{cfg.VAR_LEVEL:.0%} worst case", th.MIN_SIZE, th.COLOR_BAD).next_to(
            var_lines, UP, buff=0.12
        )
        caption = th.caption("Ignore correlation and you underestimate risk.")

        weights = "/".join(f"{w:.0%}".rstrip("%") for w in cfg.PORTFOLIO_WEIGHTS)
        new_heading = th.text(
            f"Portfolio return ({weights} weights, simulated)", th.CAPTION_SIZE, th.COLOR_MUTED
        ).move_to(heading)
        self.play(
            FadeOut(VGroup(*[VGroup(p.axes, p.label, p.labels) for p in panels])),
            Transform(heading, new_heading),
            Create(axes),
            run_time=cfg.T_SHORT,
        )
        self.play(
            *[Transform(panel.cloud, hist) for panel, hist in zip(panels, histograms, strict=True)],
            run_time=cfg.T_LONG,
        )
        self.play(FadeIn(legend), run_time=cfg.T_SHORT)
        self.play(Create(var_lines), FadeIn(var_label), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    def _hook_synthetic_data(self) -> None:
        """Original and synthetic correlation heatmaps, then synthetic rows ticking in."""
        title = th.slide_title("Synthetic data generation")
        original = np.asarray(cfg.SYNTH_CORRELATION)
        z = mm.standard_normal_samples(cfg.N_STATS, len(original), cfg.RANDOM_SEED)
        rows = mm.transform(z, np.zeros(len(original)), mm.cholesky_factor(original))
        measured = mm.sample_corr(rows)

        left = self._heatmap(original, "Original data (illustrative)", fixed=False)
        right = self._heatmap(measured, "Synthetic copy", fixed=True)
        arrow = Arrow(LEFT, RIGHT, color=th.COLOR_MUTED).scale(0.8)
        arrow_label = th.text(f"{cfg.N_STATS:,} simulated rows", th.MIN_SIZE, th.COLOR_MUTED)
        middle = VGroup(arrow_label, arrow).arrange(DOWN, buff=0.12)
        maps = VGroup(left, middle, right).arrange(RIGHT, buff=0.5)
        middle.align_to(left[1], DOWN).shift(0.9 * UP)
        maps.next_to(title, DOWN, buff=0.35).set_x(0)

        scores = cfg.SYNTH_SCORE_MEAN + cfg.SYNTH_SCORE_SD * rows[: cfg.SYNTH_TABLE_ROWS]
        table = VGroup(
            *[th.text(name, th.MIN_SIZE, th.COLOR_MUTED) for name in cfg.SYNTH_VARIABLES],
            *[th.text(f"{value:.0f}", th.MIN_SIZE) for row in scores for value in row],
        ).arrange_in_grid(
            rows=1 + cfg.SYNTH_TABLE_ROWS, cols=len(cfg.SYNTH_VARIABLES), buff=(0.9, 0.14)
        )
        table.next_to(maps, DOWN, buff=0.3).set_x(right.get_x())
        n_cols = len(cfg.SYNTH_VARIABLES)
        caption = th.caption(
            "Fake data with real structure: safe to share, useful for testing models."
        )

        self.play(FadeIn(title), FadeIn(left), run_time=cfg.T_MEDIUM)
        self.play(GrowArrow(arrow), FadeIn(arrow_label), run_time=cfg.T_SHORT)
        self.play(FadeIn(right, shift=0.2 * LEFT), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(table[:n_cols]), run_time=cfg.T_SHORT)
        self.play(
            LaggedStart(
                *[
                    FadeIn(table[start : start + n_cols], shift=0.15 * DOWN)
                    for start in range(n_cols, len(table), n_cols)
                ],
                lag_ratio=0.5,
            ),
            run_time=cfg.T_MEDIUM,
        )
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    def _heatmap(self, corr: np.ndarray, label: str, fixed: bool) -> VGroup:
        """A correlation heatmap: label, then a grid of teal cells with their values.

        Rows are named with SYNTH_VARIABLES; columns follow the same order.
        Returns VGroup(label, grid) where grid holds cells, numbers and row names.
        """
        cell = th.HEATMAP_CELL
        cells, numbers = VGroup(), VGroup()
        for i, row in enumerate(np.asarray(corr)):
            for j, value in enumerate(row):
                square = Square(cell).move_to([j * cell, -i * cell, 0])
                strength = float(np.clip(value, 0.0, 1.0))
                square.set_fill(
                    interpolate_color(
                        ManimColor(th.COLOR_PANEL), ManimColor(th.COLOR_SIGMA), strength
                    ),
                    1.0,
                )
                square.set_stroke(th.BACKGROUND, width=3)
                ink = th.BACKGROUND if strength > 0.55 else th.COLOR_TEXT
                numbers.add(th.math(fmt(value, fixed=fixed), th.MIN_SIZE, ink).move_to(square))
                cells.add(square)
        names = VGroup(
            *[
                th.text(name, th.MIN_SIZE, th.COLOR_MUTED).next_to(cells[i * len(corr)], LEFT)
                for i, name in enumerate(cfg.SYNTH_VARIABLES)
            ]
        )
        grid = VGroup(cells, numbers, names)
        title = th.text(label, th.CAPTION_SIZE).next_to(cells, UP, buff=0.2)
        return VGroup(title, grid)

    # --- 1.2 -----------------------------------------------------------------

    def slide_1_2_what_computers_give(self) -> None:
        """Code rows fly to dots of a round cloud. Point: computers give us independent noise."""
        self.begin_slide(
            notes=f"""
            Here's what a computer gives us for free. One line of numpy and we
            get {cfg.N_DISPLAY} rows, and each row is a point. Every draw is independent,
            and the two columns have nothing to do with each other, so the
            correlation is zero and the cloud is round. The rest of this talk is
            about mixing those two columns together.
            """
        )
        title = th.slide_title("What computers give us")
        shown = self.z_display[: cfg.N_PRINTED_ROWS]
        printed = [f"[{a:5.2f}, {b:5.2f}]" for a, b in shown]
        code = make_code(
            f"rng = np.random.default_rng({cfg.RANDOM_SEED})\n"
            f"Z = rng.standard_normal(({cfg.N_DISPLAY}, 2))\n"
            f"# {cfg.N_DISPLAY} points (rows), 2 variables\n"
            "\n"
            f"Z[:{len(shown)}]\n"
            f"# [{printed[0]},\n"
            + "".join(f"#  {row},\n" for row in printed[1:-1])
            + f"#  {printed[-1]}]"
        )
        code.next_to(title, DOWN, buff=0.35).align_to(title, LEFT)
        first_row_line = 5  # code line index of the first printed row
        # Each value comes from the 1D standard normal; say which one, in full.
        one_d = th.math(
            r"\text{each value} \sim \mathcal{N}(0, 1):\quad"
            r" \varphi(z) = \frac{1}{\sqrt{2\pi}}\, e^{-z^2/2}",
            th.CAPTION_SIZE,
            th.COLOR_Z,
        ).next_to(code, DOWN, buff=0.35, aligned_edge=LEFT)

        half_width = mm.shared_half_width([IDENTITY], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        axes = make_square_axes((0.0, 0.0), half_width, th.PANEL_CODE_CLOUD)
        axes.to_edge(RIGHT, buff=0.9).set_y(0.35)
        names = VGroup(
            th.math("z_1", th.MIN_SIZE, th.COLOR_MUTED).next_to(axes, DOWN, buff=0.08),
            th.math("z_2", th.MIN_SIZE, th.COLOR_MUTED).next_to(axes, LEFT, buff=0.08),
        )
        plot_label = th.math(
            r"\mathbf{z} = (z_1, z_2) \sim \mathcal{N}(\mathbf{0}, I)", th.CAPTION_SIZE, th.COLOR_Z
        ).next_to(axes, UP, buff=0.12)
        rest = make_point_cloud(self.z_display[len(shown) :], axes, th.COLOR_Z)
        correlation = mm.sample_corr(self.z_stats)[0, 1]
        readout = th.math(
            rf"\text{{correlation}}(z_1, z_2) \approx"
            rf" {fmt(correlation, fixed=True)}",
            th.CAPTION_SIZE,
        ).next_to(names[0], DOWN, buff=0.15)
        caption = VGroup(
            th.text("Each draw is independent of the others.", th.CAPTION_SIZE),
            th.text("The two variables are not correlated.", th.CAPTION_SIZE),
        ).arrange(DOWN, buff=0.12)
        caption.move_to([0, th.CAPTION_Y + 0.05, 0])

        self.play(FadeIn(title), FadeIn(code, shift=0.2 * DOWN), run_time=cfg.T_MEDIUM)
        self.play(Create(axes), FadeIn(names), FadeIn(plot_label), run_time=cfg.T_SHORT)
        # One printed row is one point: fly each row to where it lands.
        flown = VGroup()
        for index, point in enumerate(shown):
            line = code.code_lines[first_row_line + index]
            box = SurroundingRectangle(line, color=th.COLOR_Z, buff=0.05)
            dot = Dot(axes.c2p(*point), radius=th.WORKED_DOT_RADIUS, color=th.COLOR_Z)
            self.play(Create(box), run_time=cfg.T_SHORT / 2)
            self.play(ReplacementTransform(line.copy(), dot), FadeOut(box), run_time=cfg.T_SHORT)
            flown.add(dot)
        regular = make_point_cloud(shown, axes, th.COLOR_Z)
        self.play(Transform(flown, regular), fade_in_cloud(rest), run_time=cfg.T_MEDIUM)
        self.play(Write(one_d), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(readout), run_time=cfg.T_SHORT)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    # --- 1.3 -----------------------------------------------------------------

    def slide_1_3_what_we_want(self) -> None:
        """Orange tilted cloud, then mu and Sigma annotated. Point: Sigma is the blueprint."""
        self.begin_slide(
            notes="""
            What we actually want is a multivariate normal with a mean vector mu
            and a covariance matrix Sigma. Mu is the center of the cloud. The
            diagonal of Sigma holds the variances, how spread out each variable
            is, and the off-diagonal entry is the covariance, how the two move
            together. How we made this cloud is the rest of the talk.
            """
        )
        title = th.slide_title("What we want")
        half_width = mm.shared_half_width(
            [IDENTITY, cfg.MAIN_SIGMA], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        target_axes = make_square_axes(cfg.MAIN_MU, half_width, th.PANEL_LARGE)
        target_axes.move_to([3.5, 0.05, 0])
        target_cloud = make_point_cloud(
            mm.transform(self.z_display, cfg.MAIN_MU, self.main_chol), target_axes, th.COLOR_X
        )
        target_label = th.math(
            r"\mathcal{N}(\boldsymbol{\mu}, \Sigma)", th.BODY_SIZE, th.COLOR_X
        ).next_to(target_axes, DOWN, buff=0.15)

        # The blue cloud from 1.2 starts where the target will be, then shrinks away.
        blue_axes = make_square_axes((0.0, 0.0), half_width, th.PANEL_LARGE).move_to(target_axes)
        blue = VGroup(blue_axes, make_point_cloud(self.z_display, blue_axes, th.COLOR_Z))
        thumb_target = blue.copy().scale(th.PANEL_THUMB / th.PANEL_LARGE)
        thumb_target.next_to(title, DOWN, buff=0.35).align_to(title, LEFT).shift(0.3 * RIGHT)
        blue_label = th.math(r"\mathcal{N}(\mathbf{0}, I)", th.CAPTION_SIZE, th.COLOR_Z)
        blue_label.next_to(thumb_target, RIGHT, buff=0.25)

        self.play(FadeIn(title), FadeIn(blue), run_time=cfg.T_MEDIUM)
        self.play(Transform(blue, thumb_target), FadeIn(blue_label), run_time=cfg.T_MEDIUM)
        self.play(Create(target_axes), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(target_cloud), FadeIn(target_label), run_time=cfg.T_MEDIUM)
        self.reveal()

        sigma = np.asarray(cfg.MAIN_SIGMA)
        mu_block = make_labeled_matrix(
            r"\boldsymbol{\mu}", [[m] for m in cfg.MAIN_MU], th.COLOR_TEXT
        )
        sigma_block = make_labeled_matrix(r"\Sigma", sigma, th.COLOR_SIGMA)
        blocks = VGroup(mu_block, sigma_block).arrange(RIGHT, buff=0.7)
        notes_text = VGroup(
            th.text(
                f"Diagonal {fmt(sigma[0, 0])}, {fmt(sigma[1, 1])}: variances (spread)",
                th.CAPTION_SIZE,
            ),
            th.text(
                f"Off-diagonal {fmt(sigma[0, 1])}: covariance (moving together)",
                th.CAPTION_SIZE,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        left_column = VGroup(blocks, notes_text).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        left_column.next_to(thumb_target, DOWN, buff=0.55).align_to(title, LEFT)
        # Boxes are drawn after layout so they sit on the entries' final positions.
        entries = sigma_block[1].get_entries()
        diagonal_boxes = VGroup(
            *[SurroundingRectangle(entries[i], color=th.COLOR_TEXT, buff=0.08) for i in (0, 3)]
        )
        off_diagonal_boxes = VGroup(
            *[
                DashedVMobject(SurroundingRectangle(entries[i], color=th.COLOR_TEXT, buff=0.08))
                for i in (1, 2)
            ]
        )
        center = Dot(target_axes.c2p(*cfg.MAIN_MU), radius=th.WORKED_DOT_RADIUS)
        center_label = th.math(r"\boldsymbol{\mu}", th.CAPTION_SIZE)
        center_label.add_background_rectangle(color=th.BACKGROUND, opacity=0.9, buff=0.04)
        center_label.next_to(center, UR, buff=0.05)
        caption = th.caption(
            "μ says where the cloud sits. Σ says how wide it is and which way it tilts."
        )

        self.play(
            FadeIn(mu_block), FadeIn(center, scale=2), FadeIn(center_label), run_time=cfg.T_SHORT
        )
        self.play(FadeIn(sigma_block), run_time=cfg.T_SHORT)
        self.play(Create(diagonal_boxes), FadeIn(notes_text[0]), run_time=cfg.T_MEDIUM)
        self.play(Create(off_diagonal_boxes), FadeIn(notes_text[1]), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    # --- 1.4 -----------------------------------------------------------------

    def slide_1_4_reading_the_formula(self) -> None:
        """1D formula rearranged into the nD one, term by term, then the distance term."""
        self.begin_slide(
            notes="""
            This formula looks scary, but you already know it. Here's the 1D
            bell curve, and if I rearrange it a little, it has exactly the same
            shape as the multivariate one. Numbers become vectors, σ² becomes Σ,
            and dividing becomes multiplying by an inverse. The only genuinely
            new idea is the distance term: it measures distance in units of
            spread, so a point along the long axis counts as closer than one the
            same ruler distance along the short axis.
            """
        )
        title = th.slide_title("Reading the formula: same idea as 1D")
        size = th.BODY_SIZE
        original = self._colored_formula(FORMULA_1D, [None, "norm", None, "dist", None], size)
        rearranged = self._colored_formula(FORMULA_1D_REARRANGED, FORMULA_ROLES, size)
        n_d = self._colored_formula(FORMULA_ND, FORMULA_ROLES, size)
        self._align_columns(rearranged, n_d)
        rows = VGroup(rearranged, n_d)
        rows.next_to(title, DOWN, buff=0.6).set_x(0.5)
        row_names = VGroup(
            th.text("1D", th.CAPTION_SIZE, th.COLOR_MUTED).next_to(rearranged, LEFT, buff=0.5),
            th.text("nD", th.CAPTION_SIZE, th.COLOR_MUTED).next_to(n_d, LEFT, buff=0.5),
        )
        row_names.align_to(row_names[0], LEFT)
        original.move_to(rearranged, aligned_edge=LEFT)

        bell_axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[0, 0.45, 0.45],
            x_length=th.FORMULA_BELL_WIDTH,
            y_length=th.FORMULA_BELL_HEIGHT,
            tips=False,
            axis_config={"color": th.COLOR_MUTED, "stroke_width": th.AXIS_STROKE},
            y_axis_config={"include_ticks": False},
        ).next_to(original, RIGHT, buff=0.8)
        bell = VGroup(bell_axes, make_bell_curve(bell_axes, 0.0, 1.0, th.COLOR_Z))

        self.play(FadeIn(title), FadeIn(row_names[0]), run_time=cfg.T_SHORT)
        self.play(Write(original), FadeIn(bell), run_time=cfg.T_MEDIUM)
        self.reveal()

        caption = th.caption("Same formula, just rearranged.")
        self.play(FadeOut(bell), run_time=cfg.T_SHORT)
        self.play(TransformMatchingTex(original, rearranged), run_time=cfg.T_LONG)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)
        self.reveal()

        self.play(FadeIn(row_names[1]), FadeOut(caption), run_time=cfg.T_SHORT)
        self.play(
            LaggedStart(
                *[TransformFromCopy(top, bottom) for top, bottom in zip(rearranged, n_d)],
                lag_ratio=0.15,
            ),
            run_time=cfg.T_LONG,
        )

        table = self._translation_table().next_to(rows, DOWN, buff=0.45).set_x(0)
        header, *table_rows = table
        for index, row in enumerate(table_rows):
            self.reveal()
            if index == 0:
                self.play(FadeIn(header), run_time=cfg.T_SHORT)
            self.play(FadeIn(row, shift=0.15 * DOWN), run_time=cfg.T_SHORT)
        self.reveal()

        # Distance term: drop the 1D row and the table, keep the nD formula on top.
        self.play(FadeOut(table), FadeOut(rearranged), FadeOut(row_names), run_time=cfg.T_SHORT)
        self.play(n_d.animate.next_to(title, DOWN, buff=0.4).set_x(0), run_time=cfg.T_SHORT)
        distance_box = SurroundingRectangle(
            n_d[4:8], color=th.COLOR_DIST, buff=0.1, corner_radius=0.12
        )
        self._distance_points(distance_box)

    def _distance_points(self, distance_box: Mobject) -> None:
        """Points A and B, the same ruler distance from mu, with different Sigma distance."""
        mu = np.asarray(cfg.MAIN_MU)
        # A zoomed window around mu: the points of interest are only 2 units away.
        half_width = cfg.FORMULA_ZOOM_HALF_WIDTH
        axes = make_square_axes(mu, half_width, th.PANEL_FORMULA)
        axes.next_to(distance_box, DOWN, buff=0.3).set_x(-3.0)
        samples = mm.transform(self.z_display, mu, self.main_chol)
        inside = np.all(np.abs(samples - mu) <= half_width, axis=1)  # only dots in the window
        cloud = make_point_cloud(samples[inside], axes, th.COLOR_X).set_opacity(0.35)
        _, vectors = mm.eigen_axes(cfg.MAIN_SIGMA)
        distance = cfg.FORMULA_POINT_DISTANCE
        points = [mu + distance * vectors[:, 0], mu + distance * vectors[:, 1]]
        squared = [mm.mahalanobis_sq(p, mu, cfg.MAIN_SIGMA) for p in points]

        ruler = DashedVMobject(
            Circle(radius=distance * axes.x_axis.get_unit_size(), color=th.COLOR_TEXT)
        ).move_to(axes.c2p(*mu))
        center = Dot(axes.c2p(*mu), radius=0.06, color=th.COLOR_TEXT)
        markers, spokes = VGroup(), VGroup()
        for name, point in zip("AB", points, strict=True):
            dot = Dot(axes.c2p(*point), radius=th.WORKED_DOT_RADIUS, color=th.COLOR_DIST)
            label = th.math(name, th.CAPTION_SIZE, th.COLOR_DIST)
            label.add_background_rectangle(color=th.BACKGROUND, opacity=0.9, buff=0.04)
            outward = normalize(np.append(point - mu, 0.0))  # away from mu, clear of the spoke
            markers.add(VGroup(dot, label.next_to(dot, outward, buff=0.08)))
            spokes.add(Line(center.get_center(), dot.get_center(), color=th.COLOR_TEXT))

        size = th.CAPTION_SIZE
        readout = VGroup(
            VectorizedPoint(),
            th.text("ruler", size, th.COLOR_MUTED),
            th.math(r"\text{distance}^2", size, th.COLOR_DIST),
            th.text("A (long axis)", size, th.COLOR_DIST),
            th.math(fmt(distance), size),
            th.math(fmt(squared[0]), size, th.COLOR_DIST),
            th.text("B (short axis)", size, th.COLOR_DIST),
            th.math(fmt(distance), size),
            th.math(fmt(squared[1]), size, th.COLOR_DIST),
        ).arrange_in_grid(rows=3, cols=3, buff=(0.6, 0.3), col_alignments="lcc")
        readout.next_to(axes, RIGHT, buff=0.8).set_y(axes.get_y() + 0.6)
        contours = VGroup(
            *[
                DashedVMobject(
                    make_cov_ellipse(cfg.MAIN_SIGMA, mu, np.sqrt(d), axes, th.COLOR_DIST)
                )
                for d in squared
            ]
        )
        caption = th.caption(
            "Same ruler distance, different 'Σ distance'."
            " Along the long axis, points are more typical."
        )

        self.play(Create(distance_box), run_time=cfg.T_SHORT)
        self.play(Create(axes), fade_in_cloud(cloud), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(center), Create(ruler), run_time=cfg.T_SHORT)
        self.play(
            *[Create(s) for s in spokes],
            *[FadeIn(m, scale=1.5) for m in markers],
            run_time=cfg.T_MEDIUM,
        )
        self.play(FadeIn(readout), FadeIn(caption), run_time=cfg.T_MEDIUM)
        self.reveal()

        contour_caption = th.caption(
            "Every point on one ellipse has the same distance, so the same density."
        )
        self.play(
            Create(contours),
            FadeOut(ruler),
            swap_caption(caption, contour_caption),
            run_time=cfg.T_SHORT,
        )
        self.reveal()

        final = th.caption(
            "Only two things change: numbers become vectors and matrices,"
            " and distance now accounts for tilt."
        )
        self.play(swap_caption(contour_caption, final), run_time=cfg.T_SHORT)

    @staticmethod
    def _colored_formula(parts: list[str], roles: list[str | None], size: float) -> MathTex:
        """A multi-part MathTex with each piece colored by its role in the density."""
        formula = MathTex(*parts, font_size=size * th.MATH_SCALE, color=th.COLOR_TEXT)
        for piece, role in zip(formula, roles, strict=True):
            if role:
                piece.set_color(ROLE_COLORS[role])
        return formula

    @staticmethod
    def _align_columns(top: MathTex, bottom: MathTex, gap: float = 0.14) -> None:
        """Place matching pieces of two formulas in shared columns, centered in each."""
        widths = [max(a.width, b.width) for a, b in zip(top, bottom, strict=True)]
        x = 0.0
        for a, b, width in zip(top, bottom, widths, strict=True):
            a.set_x(x + width / 2)
            b.set_x(x + width / 2)
            x += width + gap
        bottom.next_to(top, DOWN, buff=0.45, coor_mask=UP)

    @staticmethod
    def _translation_table() -> VGroup:
        """Header plus four rows (center, spread, distance, normalizer), one VGroup per row."""
        size = th.CAPTION_SIZE
        cells = [
            ("Piece", "1D", "nD", "What it means"),
            ("Center", r"\mu", r"\boldsymbol{\mu}", "Where the peak is"),
            ("Spread", r"\sigma^2", r"\Sigma", "How wide, and in nD which way it tilts"),
            (
                "Distance",
                r"\frac{(x-\mu)^2}{\sigma^2}",
                r"(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})",
                "How many sds from the center, squared",
            ),
            (
                "Normalizer",
                r"\frac{1}{\sqrt{2\pi\sigma^2}}",
                r"(2\pi)^{-k/2}|\Sigma|^{-1/2}",
                "Makes the total probability 1",
            ),
        ]
        colors = [th.COLOR_MUTED, th.COLOR_TEXT, th.COLOR_SIGMA, th.COLOR_DIST, th.COLOR_NORM]
        mobjects = []
        for index, (piece, one_d, n_d, meaning) in enumerate(cells):
            color = colors[index]
            if index == 0:
                row = [th.text(t, size, color) for t in (piece, one_d, n_d, meaning)]
            else:
                row = [
                    th.text(piece, size, color),
                    th.math(one_d, th.MIN_SIZE, color),
                    th.math(n_d, th.MIN_SIZE, color),
                    th.text(meaning, th.MIN_SIZE),
                ]
            mobjects.extend(row)
        grid = VGroup(*mobjects).arrange_in_grid(
            rows=len(cells), cols=4, buff=(0.45, 0.16), col_alignments="lccl"
        )
        return VGroup(*[VGroup(*grid[i * 4 : i * 4 + 4]) for i in range(len(cells))])

    # --- 1.5 -----------------------------------------------------------------

    def slide_1_5_the_question(self) -> None:
        """Blue round cloud, '?', orange tilted cloud. Point: state the problem."""
        self.begin_slide(
            notes="""
            So here is the whole problem in one picture. On the left is the
            round cloud a computer gives us, and on the right is the tilted cloud
            we want, drawn to the same scale. The question for the rest of the
            talk is how to turn one into the other.
            """
        )
        title = th.slide_title("The question")
        half_width = mm.shared_half_width(
            [IDENTITY, cfg.MAIN_SIGMA], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        left_axes = make_square_axes((0.0, 0.0), half_width, th.PANEL_MEDIUM)
        right_axes = make_square_axes(cfg.MAIN_MU, half_width, th.PANEL_MEDIUM)
        left_cloud = make_point_cloud(self.z_display, left_axes, th.COLOR_Z)
        right_cloud = make_point_cloud(
            mm.transform(self.z_display, cfg.MAIN_MU, self.main_chol), right_axes, th.COLOR_X
        )
        ellipse = make_cov_ellipse(cfg.MAIN_SIGMA, cfg.MAIN_MU, cfg.QUESTION_ELLIPSE_K, right_axes)
        left_label = th.math(r"\mathcal{N}(\mathbf{0}, I)", th.BODY_SIZE, th.COLOR_Z)
        right_label = th.math(r"\mathcal{N}(\boldsymbol{\mu}, \Sigma)", th.BODY_SIZE, th.COLOR_X)
        left = VGroup(VGroup(left_axes, left_cloud), left_label).arrange(DOWN, buff=0.25)
        right = VGroup(VGroup(right_axes, right_cloud, ellipse), right_label).arrange(
            DOWN, buff=0.25
        )

        arrow = Arrow(LEFT, RIGHT, color=th.COLOR_TEXT, stroke_width=6).scale(1.2)
        question = th.text("?", th.QUESTION_SIZE)
        middle = VGroup(question, arrow).arrange(DOWN, buff=0.2)
        VGroup(left, middle, right).arrange(RIGHT, buff=0.7).move_to(0.2 * DOWN)
        caption = th.caption("How do we turn N(0, I) into N(μ, Σ)?")

        self.play(FadeIn(title), Create(left_axes), Create(right_axes), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(left_cloud), FadeIn(left_label), run_time=cfg.T_MEDIUM)
        self.play(
            fade_in_cloud(right_cloud), Create(ellipse), FadeIn(right_label), run_time=cfg.T_MEDIUM
        )
        self.play(GrowArrow(arrow), Write(question), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    # --- 1.6 -----------------------------------------------------------------

    def slide_1_6_one_d_warmup(self) -> None:
        """N(0, 1) to N(3, 4): scale by sqrt(variance), not the variance. Point: the 1D rule."""
        self.begin_slide(
            notes=f"""
            Let's start with something we already know. We want a normal with
            mean {cfg.ONE_D_MU:g} and variance {cfg.ONE_D_VARIANCE:g}. The obvious
            move is to multiply by {cfg.ONE_D_NAIVE_SCALE:g}, but the curve
            overshoots and the variance jumps to {cfg.ONE_D_NAIVE_SCALE**2:g},
            because stretching by a multiplies the variance by a squared. So we
            stretch by the square root of {cfg.ONE_D_VARIANCE:g}, which is
            {cfg.ONE_D_VARIANCE**0.5:g}, then shift by {cfg.ONE_D_MU:g}. To get
            variance sigma squared, you scale by its square root.
            """
        )
        mu, var, naive = cfg.ONE_D_MU, cfg.ONE_D_VARIANCE, cfg.ONE_D_NAIVE_SCALE
        sd = float(np.sqrt(var))
        title = th.slide_title("Warm-up: we already do this in 1D")
        x_lim = mm.axis_limits_1d([0.0, mu], [naive, sd], cfg.ONE_D_AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        peak = float(mm.normal_pdf_1d(0.0, 0.0, 1.0))
        axes = make_density_axes(x_lim, 1.15 * peak, th.DENSITY_WIDTH, th.DENSITY_HEIGHT)
        axes.move_to([0, -0.95, 0])

        scale, shift, warmth = ValueTracker(1.0), ValueTracker(0.0), ValueTracker(0.0)
        curve = always_redraw(
            lambda: make_bell_curve(
                axes,
                shift.get_value(),
                scale.get_value(),
                interpolate_color(
                    ManimColor(th.COLOR_Z), ManimColor(th.COLOR_X), warmth.get_value()
                ),
            )
        )
        target = DashedVMobject(make_bell_curve(axes, mu, sd, th.COLOR_TEXT), num_dashes=50)
        target.set_stroke(opacity=0.6)
        target_label = th.text("target", th.MIN_SIZE, th.COLOR_MUTED)
        target_label.next_to(axes.c2p(mu + sd, float(mm.normal_pdf_1d(mu + sd, mu, sd))), UR)
        guides = self._live_guides(axes, scale, shift)

        goal = MathTex(
            r"\text{Goal: }\mathcal{N}(",
            fmt(mu),
            r",\ ",
            fmt(var),
            r"\,)",
            font_size=th.BODY_SIZE * th.MATH_SCALE,
        )
        goal_box = SurroundingRectangle(goal[3], color=th.COLOR_SIGMA, buff=0.07)
        goal_note = th.text("variance", th.MIN_SIZE, th.COLOR_SIGMA)
        equation = th.math(r"x = \,?\cdot z + \,?")
        top_left = VGroup(goal, equation).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        top_left.next_to(title, DOWN, buff=0.4).set_x(-6.2, direction=LEFT)
        goal_box.move_to(goal[3])
        # After the closing bracket: right beside the box it ran into the ")".
        goal_note.next_to(goal, RIGHT, buff=0.25)

        readout, variance_value = self._variance_readout(scale)
        readout.next_to(title, DOWN, buff=0.45).to_edge(RIGHT, buff=th.EDGE_BUFF + 0.4)

        self.play(FadeIn(title), Create(axes), run_time=cfg.T_SHORT)
        self.add(curve)
        self.play(Create(target), FadeIn(target_label), FadeIn(curve), run_time=cfg.T_MEDIUM)
        self.add(guides)
        self.play(FadeIn(guides), FadeIn(readout), run_time=cfg.T_SHORT)
        self.play(Write(goal), run_time=cfg.T_MEDIUM)
        self.play(Create(goal_box), FadeIn(goal_note), FadeIn(equation), run_time=cfg.T_SHORT)
        self.reveal()

        # Attempt: multiply by the variance.
        naive_equation = th.math(rf"x = {fmt(naive)}z", color=th.COLOR_BAD)
        naive_equation.move_to(equation, aligned_edge=LEFT)
        cross = make_cross_mark(0.35).next_to(naive_equation, RIGHT, buff=0.35)
        caption = th.caption(f"The variance is {fmt(var)}, so multiply by {fmt(naive)}?")
        self.play(Transform(equation, naive_equation), FadeIn(caption), run_time=cfg.T_SHORT)
        self.play(scale.animate.set_value(naive), run_time=cfg.T_LONG)
        self.wait(cfg.T_SETTLE, frozen_frame=False)
        self.play(
            variance_value.animate.set_color(th.COLOR_BAD), Create(cross), run_time=cfg.T_SHORT
        )
        self.reveal()

        why = th.math(r"\mathrm{Var}(a z) = a^2\, \mathrm{Var}(z)")
        why.next_to(top_left, DOWN, buff=0.35, aligned_edge=LEFT)
        why_caption = th.caption("Stretching by a multiplies the variance by a².")
        self.play(Write(why), swap_caption(caption, why_caption), run_time=cfg.T_MEDIUM)
        self.reveal()

        # The fix: scale by the square root of the variance.
        fixed = th.math(rf"x = {fmt(sd)}z").move_to(equation, aligned_edge=LEFT)
        check = make_check_mark(0.35).move_to(cross)
        fix_caption = th.caption(f"Scale by √{fmt(var)} = {fmt(sd)} instead.")
        self.play(
            Transform(equation, fixed),
            FadeOut(cross),
            swap_caption(why_caption, fix_caption),
            run_time=cfg.T_SHORT,
        )
        self.play(scale.animate.set_value(sd), run_time=cfg.T_LONG)
        self.wait(cfg.T_SETTLE, frozen_frame=False)
        self.play(
            variance_value.animate.set_color(th.COLOR_GOOD), Create(check), run_time=cfg.T_SHORT
        )
        self.reveal()

        shifted = th.math(rf"x = {fmt(mu)} + {fmt(sd)}z", color=th.COLOR_X)
        shifted.move_to(equation, aligned_edge=LEFT)
        shift_caption = th.caption(f"Then shift by {fmt(mu)}: the curve lands on the target.")
        self.play(
            Transform(equation, shifted),
            check.animate.next_to(shifted, RIGHT, buff=0.35),
            swap_caption(fix_caption, shift_caption),
            run_time=cfg.T_SHORT,
        )
        self.play(shift.animate.set_value(mu), warmth.animate.set_value(1.0), run_time=cfg.T_LONG)
        self.wait(cfg.T_SETTLE, frozen_frame=False)
        self.reveal()

        rule = th.math(
            r"x = \mu + \sqrt{\sigma^2}\, z \sim \mathcal{N}(\mu, \sigma^2)", color=th.COLOR_X
        ).move_to(why, aligned_edge=LEFT)
        self.play(FadeOut(shift_caption), FadeTransform(why, rule), run_time=cfg.T_MEDIUM)
        self.reveal()

        # Freeze the live pieces so the overlay and the next fade-out apply to them.
        for mobject in (curve, *guides):
            mobject.clear_updaters()
        takeaway = th.math(
            r"\text{To get variance } \sigma^2, \text{ scale by } \sqrt{\sigma^2}.",
            th.TITLE_SIZE,
        )
        self.play(FadeIn(make_dim_overlay()), run_time=cfg.T_SHORT)
        self.play(Write(takeaway), run_time=cfg.T_MEDIUM)

    @staticmethod
    def _live_guides(axes: Axes, scale: ValueTracker, shift: ValueTracker) -> VGroup:
        """Mean line and one-sd arrow that follow the live curve.

        The lines are rebuilt every frame (cheap); the mu and sigma labels are
        built once and only moved, since rebuilding LaTeX every frame is slow.
        """

        def lines() -> VGroup:
            mean, spread = shift.get_value(), scale.get_value()
            top = float(mm.normal_pdf_1d(mean, mean, spread))
            at_sd = float(mm.normal_pdf_1d(mean + spread, mean, spread))
            return VGroup(
                DashedLine(axes.c2p(mean, 0), axes.c2p(mean, top), color=th.COLOR_TEXT),
                DoubleArrow(
                    axes.c2p(mean, at_sd),
                    axes.c2p(mean + spread, at_sd),
                    buff=0,
                    color=th.COLOR_TEXT,
                    stroke_width=3,
                    tip_length=0.12,
                ),
            )

        guides = always_redraw(lines)
        mu_label = th.math(r"\mu", th.MIN_SIZE)
        sd_label = th.math(r"\sigma", th.MIN_SIZE)
        for label in (mu_label, sd_label):
            label.add_background_rectangle(color=th.BACKGROUND, opacity=0.9, buff=0.03)
        mu_label.add_updater(lambda m: m.next_to(guides[0].get_end(), UR, buff=0.06))
        sd_label.add_updater(lambda m: m.next_to(guides[1], DOWN, buff=0.08))
        return VGroup(guides, mu_label, sd_label)

    @staticmethod
    def _variance_readout(scale: ValueTracker) -> tuple[VGroup, DecimalNumber]:
        """Live 'scale a = ...' and 'variance = a^2 = ...' rows."""
        size = th.BODY_SIZE * th.MATH_SCALE
        a_value = DecimalNumber(scale.get_value(), num_decimal_places=2, font_size=size)
        a_value.add_updater(lambda m: m.set_value(scale.get_value()))
        variance = DecimalNumber(scale.get_value() ** 2, num_decimal_places=2, font_size=size)
        variance.add_updater(lambda m: m.set_value(scale.get_value() ** 2))
        rows = VGroup(
            VGroup(th.math("a ="), a_value).arrange(RIGHT, buff=0.2),
            VGroup(th.math(r"\text{variance} = a^2 ="), variance).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.3, aligned_edge=RIGHT)
        return rows, variance

    # --- 1.7 -----------------------------------------------------------------

    def slide_1_7_nd_catch(self) -> None:
        """The variance becomes a matrix; two naive square roots fail. Point: need a new idea."""
        self.begin_slide(
            notes="""
            Now go up a dimension. The variance is no longer one number, it's a
            whole matrix. If we make the same mistake as before and multiply by
            Σ, we get Σ squared. And taking the square root of each entry fails
            too. So what does the square root of a matrix even mean?
            """
        )
        title = th.slide_title("The nD catch: the variance becomes a matrix")
        var = cfg.ONE_D_VARIANCE
        question = th.math("???", color=th.COLOR_BAD)
        one_d = [
            th.text("1D", th.BODY_SIZE, th.COLOR_MUTED),
            th.math(rf"\text{{variance}} = {fmt(var)}"),
            th.text("(one number)", th.CAPTION_SIZE, th.COLOR_MUTED),
            Arrow(LEFT, RIGHT, color=th.COLOR_MUTED).scale(0.5),
            th.math(rf"\text{{scale by }} \sqrt{{{fmt(var)}}} = {fmt(np.sqrt(var))}"),
            make_check_mark(0.35),
        ]
        n_d = [
            th.text("nD", th.BODY_SIZE, th.COLOR_MUTED),
            VGroup(
                th.math(r"\text{covariance } \Sigma =", color=th.COLOR_SIGMA),
                make_matrix(cfg.MAIN_SIGMA, th.COLOR_SIGMA),
            ).arrange(RIGHT, buff=0.2),
            th.text("(a matrix)", th.CAPTION_SIZE, th.COLOR_MUTED),
            Arrow(LEFT, RIGHT, color=th.COLOR_MUTED).scale(0.5),
            VGroup(th.math(r"\text{scale by } \sqrt{\Sigma} ="), question).arrange(RIGHT, buff=0.2),
            VectorizedPoint(),  # keeps the grid rectangular; the nD row has no mark yet
        ]
        bridge = VGroup(*one_d, *n_d).arrange_in_grid(
            rows=2, cols=6, buff=(0.35, 0.8), col_alignments="llllll"
        )
        one_d, n_d = VGroup(*bridge[:6]), VGroup(*bridge[6:])
        bridge.move_to(0.3 * UP)
        caption = th.caption("In nD, the variance is a whole matrix. What is its square root?")

        self.play(FadeIn(title), run_time=cfg.T_SHORT)
        self.play(FadeIn(one_d, lag_ratio=0.15), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(n_d, lag_ratio=0.15), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)
        for _ in range(2):  # pulse the unknown
            self.play(question.animate(rate_func=there_and_back).scale(1.35), run_time=cfg.T_SHORT)
        self.reveal()

        target = VGroup(
            th.text("Target:", th.CAPTION_SIZE, th.COLOR_SIGMA),
            make_labeled_matrix(r"\Sigma", cfg.MAIN_SIGMA, th.COLOR_SIGMA, th.CAPTION_SIZE),
        ).arrange(RIGHT, buff=0.3)
        target.next_to(title, DOWN, buff=0.15).set_x(-6.4, direction=LEFT)
        attempt_one = self._attempt_row(
            "Attempt 1: multiply by Σ",
            make_labeled_matrix(r"\Sigma", cfg.MAIN_SIGMA, th.COLOR_SIGMA, th.CAPTION_SIZE),
            np.asarray(cfg.MAIN_SIGMA),
            th.math(r"\approx \Sigma^2", th.CAPTION_SIZE, th.COLOR_SIGMA),
        )
        root_tex = [[sqrt_tex(v) for v in row] for row in cfg.MAIN_SIGMA]
        attempt_two = self._attempt_row(
            "Attempt 2: root of each entry",
            VGroup(
                th.math("A =", th.CAPTION_SIZE),
                make_symbolic_matrix(root_tex, th.COLOR_TEXT, th.CAPTION_SIZE),
            ),
            mm.elementwise_sqrt(cfg.MAIN_SIGMA),
            th.math(r"\neq \Sigma", th.CAPTION_SIZE, th.COLOR_BAD),
        )
        rows = VGroup(attempt_one, attempt_two).arrange(DOWN, buff=0.15)
        rows.next_to(target, DOWN, buff=0.1).set_x(0)
        for row in rows:  # line up the plots and arrows of both rows
            row[1:].shift((rows[0][2].get_x() - row[2].get_x()) * RIGHT)

        one_caption = th.caption("Just like 1D: multiplying by the variance squares it.")
        self.play(FadeOut(bridge), swap_caption(caption, one_caption), run_time=cfg.T_SHORT)
        self.play(FadeIn(target), run_time=cfg.T_SHORT)
        self._play_attempt(attempt_one)
        self.reveal()

        two_caption = th.caption(
            "Square-rooting each entry doesn't work."
            " Matrices need a better definition of square root."
        )
        self.play(FadeOut(one_caption), run_time=cfg.T_SHORT)
        self._play_attempt(attempt_two)
        self.play(FadeIn(two_caption), run_time=cfg.T_SHORT)

    def _attempt_row(
        self, label: str, matrix_mob: VGroup, matrix: np.ndarray, verdict: Mobject
    ) -> VGroup:
        """One attempt: the guess, its cloud vs the dashed target, and the measured covariance.

        The cloud uses the displayed samples; the measured covariance uses all
        N_STATS samples, both through the same matrix. It is labeled
        "measured" on purpose: slide 2.1 derives the exact values.
        """
        matrix_mob.arrange(RIGHT, buff=0.2)
        guess = VGroup(th.text(label, th.CAPTION_SIZE), matrix_mob).arrange(
            DOWN, buff=0.2, aligned_edge=LEFT
        )
        produced = mm.covariance_of_linear_map(matrix)
        half_width = mm.shared_half_width(
            [cfg.MAIN_SIGMA, produced], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        axes = make_square_axes((0.0, 0.0), half_width, th.PANEL_ROW)
        cloud = make_point_cloud(
            mm.transform(self.z_display, np.zeros(2), matrix),
            axes,
            th.COLOR_X,
            panel_dot_radius(th.PANEL_ROW),
        )
        target_ellipse = DashedVMobject(
            make_cov_ellipse(cfg.MAIN_SIGMA, (0.0, 0.0), cfg.QUESTION_ELLIPSE_K, axes)
        )
        measured_cov = mm.sample_cov(mm.transform(self.z_stats, np.zeros(2), matrix))
        measured_matrix = VGroup(
            make_symbolic_matrix(
                [[fmt(v, cfg.MEASURED_DECIMALS, fixed=True) for v in row] for row in measured_cov],
                th.COLOR_X,
                th.CAPTION_SIZE,
            ),
            verdict,
        ).arrange(RIGHT, buff=0.2)
        measured = VGroup(
            th.text("Measured ≈", th.CAPTION_SIZE, th.COLOR_MUTED), measured_matrix
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        plot = VGroup(axes, cloud, target_ellipse)
        arrow_in = Arrow(LEFT, RIGHT, color=th.COLOR_MUTED).scale(0.35)
        arrow_out = arrow_in.copy()
        cross = make_cross_mark(0.45)
        return VGroup(guess, arrow_in, plot, arrow_out, measured, cross).arrange(RIGHT, buff=0.3)

    def _play_attempt(self, row: VGroup) -> None:
        guess, arrow_in, plot, arrow_out, measured, cross = row
        axes, cloud, target_ellipse = plot
        self.play(FadeIn(guess), run_time=cfg.T_MEDIUM)
        self.play(GrowArrow(arrow_in), Create(axes), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(cloud), run_time=cfg.T_MEDIUM)
        self.play(Create(target_ellipse), run_time=cfg.T_SHORT)  # on top of the dots
        self.play(GrowArrow(arrow_out), FadeIn(measured), run_time=cfg.T_MEDIUM)
        self.play(Create(cross), run_time=cfg.T_SHORT)
