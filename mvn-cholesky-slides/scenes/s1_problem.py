"""Section 1: the problem.

Computers give us independent noise N(0, I); we want correlated data
N(mu, Sigma). This section shows the gap and why the obvious fixes fail.
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
    make_check_mark,
    boxed,
    DeckSlide,
    Panel,
    axis_labels,
    fade_in_cloud,
    fmt,
    make_bell_curve,
    make_code,
    make_cov_ellipse,
    make_cross_mark,
    make_density_axes,
    make_mean_sd_guides,
    make_labeled_matrix,
    make_matrix,
    make_point_cloud,
    make_square_axes,
    make_symbolic_matrix,
    panel_dot_radius,
    sqrt_tex,
)

IDENTITY = np.eye(2)
SIGMA_SYMBOLS = [[r"\sigma_{xx}", r"\sigma_{xy}"], [r"\sigma_{yx}", r"\sigma_{yy}"]]
ENTRY_INDEX = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (row, col) of each entry, in Matrix order


class S1Problem(DeckSlide):
    """Section 1: what we have, what we want, and why naive fixes fail."""

    section_number = 1
    section_name = "Problem"

    def construct(self) -> None:
        self.slide_1_0_title()
        self.slide_1_0_section()
        self.slide_1_1_hook()
        self.slide_1_2_what_computers_give()
        self.slide_1_3_what_we_want()
        self.slide_1_4_what_each_entry_does()
        self.slide_1_5_the_question()
        self.slide_1_6_one_d_warmup()
        self.slide_1_7_the_trap()
        self.slide_1_8_matrix_square_root()

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

    # --- 1.0 -----------------------------------------------------------------

    def slide_1_0_section(self) -> None:
        """Section slide: number, name and a one-line preview of the section."""
        self.begin_slide(
            notes="""
            Let's start with the problem. We will look at what a computer can
            give us, what we actually want, and why the obvious fixes fail.
            """
        )
        self.show_section_slide("What we have, what we want, and why it is not obvious")

    # --- 1.1 -----------------------------------------------------------------

    def slide_1_1_hook(self) -> None:
        """Independent vs correlated stock returns. Point: real variables move together."""
        self.begin_slide(
            notes="""
            Here are simulated daily returns for two stocks. On the left they
            are independent, so the cloud is round. On the right they tend to
            move together, which tilts the cloud. If we want to simulate a
            portfolio honestly, we need draws like the ones on the right.
            """
        )
        title = th.slide_title("Real variables move together")
        heading = th.text("Stock A vs Stock B returns (simulated)", th.CAPTION_SIZE, th.COLOR_MUTED)
        heading.next_to(title, DOWN, buff=th.TITLE_GAP)

        correlated = mm.correlation_matrix(cfg.HOOK_RHO)
        half_width = mm.shared_half_width(
            [IDENTITY, correlated], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        left = Panel(
            "Independent", IDENTITY, self.z_display, half_width, th.PANEL_MEDIUM, th.COLOR_Z, None
        )
        right = Panel(
            "Correlated", correlated, self.z_display, half_width, th.PANEL_MEDIUM, th.COLOR_X, None
        )
        for panel in (left, right):
            panel.add(axis_labels(panel.axes, "Stock A", "Stock B"))
        VGroup(left, right).arrange(RIGHT, buff=1.8).next_to(heading, DOWN, buff=0.2)
        caption = th.caption("To simulate a portfolio, we need random draws that move together.")

        self.play(FadeIn(title), FadeIn(heading), run_time=cfg.T_SHORT)
        self.play(
            *[FadeIn(VGroup(p.axes, p.label, p[-1])) for p in (left, right)],
            run_time=cfg.T_SHORT,
        )
        self.play(fade_in_cloud(left.cloud), fade_in_cloud(right.cloud))
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    # --- 1.2 -----------------------------------------------------------------

    def slide_1_2_what_computers_give(self) -> None:
        """A round blue N(0, I) cloud. Point: computers give us independent noise."""
        self.begin_slide(
            notes="""
            This is what a computer gives us for free. One line of NumPy draws
            from the 1D standard normal, the familiar bell curve with mean 0 and
            variance 1. Each row here is two independent draws, so the vector z
            is N of 0 and I, and that independence is why the cloud is round.
            """
        )
        title = th.slide_title("What computers give us")
        half_width = mm.shared_half_width([IDENTITY], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        axes = make_square_axes((0.0, 0.0), half_width, th.PANEL_LARGE)
        cloud = make_point_cloud(self.z_display, axes, th.COLOR_Z)

        # The real call, with the deck's own seed and sample count.
        code = make_code(
            f"rng = np.random.default_rng({cfg.RANDOM_SEED})\n"
            f"z = rng.standard_normal(({cfg.N_DISPLAY}, 2))\n"
            "# each row: two independent N(0, 1) draws"
        )
        code.next_to(title, DOWN, buff=0.4).align_to(title, LEFT)
        # Each call samples the 1D standard normal; two independent draws form z.
        one_d = th.math(
            r"\mathcal{N}(0, 1):\quad \varphi(z) = \frac{1}{\sqrt{2\pi}}\, e^{-z^2/2}",
            th.BODY_SIZE,
            th.COLOR_Z,
        )
        vector = th.math(
            r"\mathbf{z} = (z_1, z_2) \sim \mathcal{N}(\mathbf{0}, I)", th.BODY_SIZE, th.COLOR_Z
        )
        distribution = VGroup(one_d, vector).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        distribution.next_to(code, DOWN, buff=0.55).align_to(code, LEFT).shift(0.3 * RIGHT)
        VGroup(axes, cloud).move_to([3.6, -0.35, 0])
        caption = th.caption("Independent. No correlation.")

        self.play(FadeIn(title), FadeIn(code, shift=0.2 * DOWN), run_time=cfg.T_MEDIUM)
        self.play(Create(axes), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(cloud))
        self.play(Write(one_d), run_time=cfg.T_MEDIUM)
        self.play(Write(vector), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    # --- 1.3 -----------------------------------------------------------------

    def slide_1_3_what_we_want(self) -> None:
        """Density, Sigma and the target cloud. Point: Sigma is the blueprint we want."""
        self.begin_slide(
            notes="""
            What we actually want is a multivariate normal with a mean vector mu
            and a covariance matrix Sigma. The diagonal of Sigma holds the
            variances of each variable, and the off-diagonal entries hold how
            much they move together. Data drawn from it looks like this tilted
            orange cloud. How we made it is the rest of the talk.
            """
        )
        title = th.slide_title("What we want")
        density = th.math(
            r"f(\mathbf{x}) = \frac{1}{(2\pi)^{k/2}\, |\Sigma|^{1/2}}"
            r"\exp\!\left(-\frac{1}{2} (\mathbf{x}-\boldsymbol{\mu})^T"
            r" \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)",
            substrings_to_isolate=[r"\Sigma", r"\mathbf{x}"],
        )
        density.set_color_by_tex(r"\Sigma", th.COLOR_SIGMA)
        density.set_color_by_tex(r"\mathbf{x}", th.COLOR_X)
        density.next_to(title, DOWN, buff=0.3).set_x(0)

        self.play(FadeIn(title), run_time=cfg.T_SHORT)
        self.play(Write(density), run_time=cfg.T_LONG)
        self.reveal()

        # Symbols rather than numbers: this slide is about what each entry means.
        sigma_block = VGroup(
            th.math(r"\Sigma =", color=th.COLOR_SIGMA),
            make_symbolic_matrix(SIGMA_SYMBOLS, th.COLOR_SIGMA),
        ).arrange(RIGHT, buff=0.2)
        mu_block = VGroup(
            th.math(r"\boldsymbol{\mu} ="),
            make_symbolic_matrix([[r"\mu_x"], [r"\mu_y"]], th.COLOR_TEXT),
        ).arrange(RIGHT, buff=0.2)
        blocks = VGroup(mu_block, sigma_block).arrange(RIGHT, buff=0.8)
        notes_text = VGroup(
            th.text("Diagonal (boxed): variances", th.CAPTION_SIZE),
            th.text("Off-diagonal (dashed): covariances", th.CAPTION_SIZE),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        left_column = VGroup(blocks, notes_text).arrange(DOWN, buff=0.6)
        left_column.move_to([-3.4, -0.75, 0])
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

        self.play(FadeIn(mu_block), FadeIn(sigma_block), run_time=cfg.T_MEDIUM)
        self.play(Create(diagonal_boxes), FadeIn(notes_text[0]), run_time=cfg.T_MEDIUM)
        self.play(Create(off_diagonal_boxes), FadeIn(notes_text[1]), run_time=cfg.T_MEDIUM)
        self.reveal()

        half_width = mm.shared_half_width(
            [IDENTITY, cfg.MAIN_SIGMA], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        axes = make_square_axes(cfg.MAIN_MU, half_width, th.PANEL_TARGET)
        samples = mm.transform(self.z_display, cfg.MAIN_MU, self.main_chol)
        cloud = make_point_cloud(samples, axes, th.COLOR_X)
        ellipse = make_cov_ellipse(cfg.MAIN_SIGMA, cfg.MAIN_MU, cfg.QUESTION_ELLIPSE_K, axes)
        VGroup(axes, cloud, ellipse).move_to([3.6, -0.85, 0])

        self.play(Create(axes), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(cloud))
        self.play(Create(ellipse), run_time=cfg.T_MEDIUM)

    # --- 1.4 -----------------------------------------------------------------

    def slide_1_4_what_each_entry_does(self) -> None:
        """Change Sigma's entries one at a time. Point: each entry has a visible job."""
        self.begin_slide(
            notes="""
            Here is what each entry actually does, using the same random points
            the whole time. Making sigma xx bigger stretches the cloud along x,
            and making sigma yy smaller squeezes it in y. The covariance sets the
            tilt: negative tilts it down, and zero lines the ellipse up with the axes.
            """
        )
        title = th.slide_title("What each entry does")
        start = np.asarray(cfg.MAIN_SIGMA, dtype=float)
        steps = [(caption, np.asarray(sigma, dtype=float)) for caption, sigma in cfg.SIGMA_WALK]
        # One tracker per distinct entry: sigma_yx always equals sigma_xy.
        var_x, cov_xy, var_y = (
            ValueTracker(start[0, 0]),
            ValueTracker(start[0, 1]),
            ValueTracker(start[1, 1]),
        )

        def current_sigma() -> np.ndarray:
            return np.array(
                [[var_x.get_value(), cov_xy.get_value()], [cov_xy.get_value(), var_y.get_value()]]
            )

        half_width = mm.shared_half_width(
            [start] + [sigma for _, sigma in steps], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        axes = make_square_axes(cfg.MAIN_MU, half_width, th.PANEL_TARGET).move_to([3.7, -0.3, 0])
        # Redrawn every frame from the same z, so the only change is Sigma.
        cloud = always_redraw(
            lambda: make_point_cloud(
                mm.transform(self.z_display, cfg.MAIN_MU, mm.cholesky_factor(current_sigma())),
                axes,
                th.COLOR_X,
            )
        )
        ellipse = always_redraw(
            lambda: make_cov_ellipse(current_sigma(), cfg.MAIN_MU, cfg.QUESTION_ELLIPSE_K, axes)
        )

        symbols = make_symbolic_matrix(SIGMA_SYMBOLS, th.COLOR_SIGMA)
        values = self._live_sigma_matrix(current_sigma)
        sigma_row = VGroup(
            th.math(r"\Sigma =", color=th.COLOR_SIGMA), symbols, th.math("="), values
        ).arrange(RIGHT, buff=0.25)
        sigma_row.move_to([-3.4, 0.2, 0])

        self.play(FadeIn(title), Create(axes), FadeIn(sigma_row), run_time=cfg.T_SHORT)
        self.add(cloud, ellipse)
        self.play(FadeIn(cloud), FadeIn(ellipse), run_time=cfg.T_SHORT)

        previous, boxes, caption = start, VGroup(), VGroup()
        for text, target in steps:
            self.reveal()
            changed = [k for k, (i, j) in enumerate(ENTRY_INDEX) if target[i, j] != previous[i, j]]
            new_boxes = VGroup(
                *[
                    SurroundingRectangle(symbols.get_entries()[k], color=th.COLOR_TEXT)
                    for k in changed
                ],
                *[self._value_box(values.get_entries()[k]) for k in changed],
            )
            new_caption = th.caption(text)
            self.play(
                FadeOut(boxes),
                FadeOut(caption),
                Create(new_boxes),
                FadeIn(new_caption),
                run_time=cfg.T_SHORT,
            )
            self.play(
                var_x.animate.set_value(target[0, 0]),
                cov_xy.animate.set_value(target[0, 1]),
                var_y.animate.set_value(target[1, 1]),
                run_time=cfg.SIGMA_WALK_SECONDS,
            )
            # Not a frozen frame: rerun the updaters so the readout shows the exact target.
            self.wait(cfg.T_SETTLE, frozen_frame=False)
            previous, boxes, caption = target, new_boxes, new_caption

        # Freeze the live pieces so the next slide's fade-out applies to them.
        for mobject in (cloud, ellipse, *values.get_entries()):
            mobject.clear_updaters()

    def _live_sigma_matrix(self, current_sigma) -> DecimalMatrix:
        """Numeric Sigma whose entries follow ``current_sigma()`` every frame."""
        matrix = DecimalMatrix(
            current_sigma(),
            element_to_mobject_config={
                "num_decimal_places": cfg.SIGMA_WALK_DECIMALS,
                "font_size": th.BODY_SIZE * th.MATH_SCALE,
            },
            h_buff=1.5,
            v_buff=0.7,
            element_alignment_corner=ORIGIN,
        ).set_color(th.COLOR_SIGMA)
        for entry, (i, j) in zip(matrix.get_entries(), ENTRY_INDEX, strict=True):
            entry.add_updater(lambda m, i=i, j=j: self._set_centered(m, current_sigma()[i, j]))
        return matrix

    @staticmethod
    def _set_centered(number: DecimalNumber, value: float) -> None:
        """Update a DecimalNumber but keep it centered, so a new minus sign
        does not push it sideways out of its column."""
        center = number.get_center()
        number.set_value(value)
        number.move_to(center)

    def _value_box(self, entry: DecimalNumber) -> SurroundingRectangle:
        """A highlight box sized for the widest readout ("-9.9"), so it fits every value."""
        widest = DecimalNumber(
            -9.9, num_decimal_places=cfg.SIGMA_WALK_DECIMALS, font_size=th.BODY_SIZE * th.MATH_SCALE
        ).move_to(entry)
        return SurroundingRectangle(widest, color=th.COLOR_TEXT)

    # --- 1.5 -----------------------------------------------------------------

    def slide_1_5_the_question(self) -> None:
        """Blue round cloud, '?', orange tilted cloud. Point: state the problem."""
        self.begin_slide(
            notes="""
            So here is the whole problem in one picture. On the left is the
            round cloud we can generate, and on the right is the tilted cloud we
            want. Both plots use the same scale. So the question is: how do we
            change the distribution of the points to get from the left to the right?
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
        caption = th.caption("How do we change the distribution of the points?")

        self.play(FadeIn(title), Create(left_axes), Create(right_axes), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(left_cloud), FadeIn(left_label))
        self.play(fade_in_cloud(right_cloud), Create(ellipse), FadeIn(right_label))
        self.play(GrowArrow(arrow), Write(question), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    # --- 1.6 -----------------------------------------------------------------

    def slide_1_6_one_d_warmup(self) -> None:
        """Stretch then shift a bell curve. Point: in 1D we scale by sigma, then shift by mu."""
        self.begin_slide(
            notes="""
            Let's warm up in one dimension, where we already know the answer.
            Take z from N of 0 and 1, stretch it by the standard deviation, then
            shift it by the mean; watch the mean line and the sigma markers move.
            The result is exactly N of 3 and 4.
            """
        )
        mu, sd = cfg.ONE_D_MU, cfg.ONE_D_SIGMA
        title = th.slide_title("Warm-up: we already do this in 1D")
        axes = self._one_d_axes()
        peak = float(mm.normal_pdf_1d(0.0, 0.0, 1.0))

        curve = make_bell_curve(axes, 0.0, 1.0, th.COLOR_Z)
        curve_label = th.math(r"z \sim \mathcal{N}(0, 1)", th.BODY_SIZE, th.COLOR_Z)
        curve_label.next_to(axes.c2p(-0.5, peak), LEFT, buff=0.3)  # leaves room for the mu label
        guides = make_mean_sd_guides(axes, 0.0, 1.0)

        steps = VGroup(
            th.math(rf"\text{{1. Stretch by }} \sigma = {fmt(sd)}"),
            th.math(rf"\text{{2. Shift by }} \mu = {fmt(mu)}"),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        results = VGroup(
            th.math(rf"{fmt(sd)}z \sim \mathcal{{N}}(0, {fmt(sd**2)})", color=th.COLOR_Z),
            th.math(
                rf"x = {fmt(mu)} + {fmt(sd)}z \sim \mathcal{{N}}({fmt(mu)}, {fmt(sd**2)})",
                color=th.COLOR_X,
            ),
        )
        steps.next_to(title, DOWN, buff=0.4).set_x(-6.4, direction=LEFT)
        for step, result in zip(steps, results, strict=True):
            result.move_to(step, coor_mask=UP).set_x(0.6, direction=LEFT)

        self.play(FadeIn(title), Create(axes), run_time=cfg.T_SHORT)
        self.play(Create(curve), FadeIn(curve_label), run_time=cfg.T_MEDIUM)
        self.play(Create(guides), run_time=cfg.T_MEDIUM)
        self.reveal()

        stretched = make_bell_curve(axes, 0.0, sd, th.COLOR_Z)
        self.play(FadeIn(steps[0]), FadeOut(curve_label), run_time=cfg.T_SHORT)
        self.play(
            Transform(curve, stretched),
            Transform(guides, make_mean_sd_guides(axes, 0.0, sd)),
            FadeIn(results[0]),
            run_time=cfg.T_LONG,
        )
        self.reveal()

        shifted = make_bell_curve(axes, mu, sd, th.COLOR_X)
        self.play(FadeIn(steps[1]), run_time=cfg.T_SHORT)
        self.play(
            Transform(curve, shifted),
            Transform(guides, make_mean_sd_guides(axes, mu, sd)),
            FadeIn(results[1]),
            run_time=cfg.T_LONG,
        )

    def _one_d_axes(self, center_y: float = -0.95) -> Axes:
        """Bell-curve axes shared by slides 1.6 and 1.7, so the trap is drawn to the same scale."""
        x_lim = mm.axis_limits_1d(
            [0.0, cfg.ONE_D_MU], [1.0, cfg.ONE_D_SIGMA], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        peak = float(mm.normal_pdf_1d(0.0, 0.0, 1.0))
        axes = make_density_axes(x_lim, 1.15 * peak, th.DENSITY_WIDTH, th.DENSITY_HEIGHT)
        return axes.move_to([0, center_y, 0])

    # --- 1.7 -----------------------------------------------------------------

    def slide_1_7_the_trap(self) -> None:
        """Scale by the variance instead of sigma. Point: scale by sqrt(variance)."""
        self.begin_slide(
            notes="""
            Here is the trap. If we scale by the variance, 4, instead of the
            standard deviation, 2, the spread doubles and the variance becomes 16,
            four times too big. So the rule is: scale by the square root of the
            variance. Remember this, because in more dimensions we will need the
            square root of a matrix.
            """
        )
        mu, sd = cfg.ONE_D_MU, cfg.ONE_D_SIGMA
        var = sd**2
        title = th.slide_title("The trap: scaling by the variance")
        axes = self._one_d_axes(center_y=-0.55)  # higher, to leave room for the rule box
        right_curve = make_bell_curve(axes, mu, sd, th.COLOR_X)
        right_guides = make_mean_sd_guides(axes, mu, sd)
        trap_curve = DashedVMobject(make_bell_curve(axes, mu, var, th.COLOR_BAD), num_dashes=60)
        trap_guides = make_mean_sd_guides(axes, mu, var, th.COLOR_BAD, show_mean=False)

        rows = VGroup(
            th.math(rf"\text{{Scale by }} \sigma = \sqrt{{{fmt(var)}}} = {fmt(sd)}"),
            th.math(rf"\text{{Scale by }} \sigma^2 = {fmt(var)}", color=th.COLOR_BAD),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        results = VGroup(
            th.math(
                rf"{fmt(mu)} + {fmt(sd)}z \sim \mathcal{{N}}({fmt(mu)}, {fmt(var)})",
                color=th.COLOR_X,
            ),
            th.math(
                rf"{fmt(mu)} + {fmt(var)}z \sim \mathcal{{N}}({fmt(mu)}, {fmt(var**2)})",
                color=th.COLOR_BAD,
            ),
        )
        rows.next_to(title, DOWN, buff=0.4).set_x(-5.8, direction=LEFT)
        for row, result in zip(rows, results, strict=True):
            result.move_to(row, coor_mask=UP).set_x(0.6, direction=LEFT)
        check = make_check_mark(0.35).next_to(rows[0], LEFT, buff=0.2)
        cross = make_cross_mark(0.3).next_to(rows[1], LEFT, buff=0.2)

        rule = VGroup(
            th.text("Scale by the square root of the variance", th.BODY_SIZE),
            th.math(r"x = \mu + \sqrt{\sigma^2}\, z = \mu + \sigma z", th.BODY_SIZE, th.COLOR_X),
        ).arrange(RIGHT, buff=0.6)
        rule_box = boxed(rule, color=th.COLOR_GOOD, buff=0.2).move_to([0, -3.0, 0])

        self.play(FadeIn(title), Create(axes), run_time=cfg.T_SHORT)
        self.play(
            Create(right_curve),
            FadeIn(right_guides),
            FadeIn(rows[0]),
            FadeIn(results[0]),
            Create(check),
            run_time=cfg.T_MEDIUM,
        )
        self.reveal()

        self.play(FadeIn(rows[1]), FadeIn(results[1]), run_time=cfg.T_SHORT)
        self.play(Create(trap_curve), FadeIn(trap_guides), run_time=cfg.T_LONG)
        self.play(Create(cross), run_time=cfg.T_SHORT)
        self.reveal()

        self.play(FadeIn(rule_box, scale=1.05), run_time=cfg.T_MEDIUM)

    # --- 1.8 -----------------------------------------------------------------

    def slide_1_8_matrix_square_root(self) -> None:
        """Two naive matrix 'square roots', measured. Point: element-wise thinking fails."""
        self.begin_slide(
            notes="""
            In more than one dimension, what is the square root of a matrix?
            Let's just try things and measure. Taking the square root of each
            entry gives a cloud that is too thin and the wrong covariance.
            Multiplying by Sigma itself overshoots badly, so we need a better idea.
            """
        )
        title = th.slide_title("What is the square root of a matrix?")
        target = VGroup(
            th.text("Target:", th.CAPTION_SIZE, th.COLOR_SIGMA),
            make_labeled_matrix(r"\Sigma", cfg.MAIN_SIGMA, th.COLOR_SIGMA, th.CAPTION_SIZE),
        ).arrange(RIGHT, buff=0.3)
        target.next_to(title, DOWN, buff=0.15).set_x(-6.4, direction=LEFT)

        root = mm.elementwise_sqrt(cfg.MAIN_SIGMA)
        root_tex = [[sqrt_tex(v) for v in row] for row in cfg.MAIN_SIGMA]
        attempt_one = self._attempt_row(
            "Attempt 1: element-wise root",
            VGroup(
                th.math("A =", th.CAPTION_SIZE),
                make_symbolic_matrix(root_tex, th.COLOR_TEXT, th.CAPTION_SIZE),
            ),
            root,
            "A^2",
        )
        attempt_two = self._attempt_row(
            "Attempt 2: multiply by Σ",
            make_labeled_matrix(r"\Sigma", cfg.MAIN_SIGMA, th.COLOR_SIGMA, th.CAPTION_SIZE),
            np.asarray(cfg.MAIN_SIGMA),
            r"\Sigma^2",
        )
        rows = VGroup(attempt_one, attempt_two).arrange(DOWN, buff=0.15)
        rows.next_to(target, DOWN, buff=0.1).set_x(0)

        self.play(FadeIn(title), FadeIn(target), run_time=cfg.T_SHORT)
        self._play_attempt(attempt_one)
        self.reveal()
        self._play_attempt(attempt_two)
        self.reveal()
        self.play(FadeIn(th.caption("Element-wise thinking fails. We need a better definition.")))

    def _attempt_row(
        self, label: str, matrix_mob: VGroup, matrix: np.ndarray, square_tex: str
    ) -> VGroup:
        """One row of slide 1.8: the guess, its cloud vs the target, the measured covariance.

        The cloud uses the displayed samples, the measured covariance uses all
        N_STATS samples; both go through the same matrix. The measured
        covariance is labeled as the square of the guess (``square_tex``): the
        guesses are symmetric, so A A^T = A^2, which is easier to read here
        than the general A A^T that slide 2.1 derives.
        """
        matrix_mob.arrange(RIGHT, buff=0.2)
        guess = VGroup(th.text(label, th.CAPTION_SIZE), matrix_mob).arrange(
            DOWN, buff=0.2, aligned_edge=LEFT
        )
        produced = mm.covariance_after_matrix(matrix)
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
        measured = VGroup(
            th.math(square_tex + r" \approx"),
            make_matrix(measured_cov, th.COLOR_X, th.CAPTION_SIZE, fixed=True),
        ).arrange(RIGHT, buff=0.2)
        plot = VGroup(axes, cloud, target_ellipse)
        arrow_in = Arrow(LEFT, RIGHT, color=th.COLOR_MUTED).scale(0.35)
        arrow_out = arrow_in.copy()
        cross = make_cross_mark(0.45)
        return VGroup(guess, arrow_in, plot, arrow_out, measured, cross).arrange(RIGHT, buff=0.3)

    def _play_attempt(self, row: VGroup) -> None:
        guess, arrow_in, plot, arrow_out, measured, cross = row
        axes, cloud, target_ellipse = plot
        self.play(FadeIn(guess), run_time=cfg.T_MEDIUM)
        self.play(GrowArrow(arrow_in), Create(axes), Create(target_ellipse), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(cloud))
        self.play(GrowArrow(arrow_out), FadeIn(measured), run_time=cfg.T_MEDIUM)
        self.play(Create(cross), run_time=cfg.T_SHORT)
