"""Section 4: wrap-up.

Recap, limits, the closing slide, and two backup appendix slides: x = mu + G z
also derives the density (A.1), and the eigen view of Sigma (A.2).
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
    boxed,
    fmt,
    fmt_vector,
    make_cov_ellipse,
    fade_in_cloud,
    make_axes,
    make_histogram,
    make_labeled_matrix,
    make_pivot_gauge,
    make_point_cloud,
    make_square_axes,
    panel_dot_radius,
    swap_caption,
)

IDENTITY = np.eye(2)


def dimmed(mobject: Mobject, factor: float) -> Mobject:
    """A copy with every stroke and fill opacity multiplied by ``factor``.

    Unlike set_opacity, this leaves unfilled shapes (the gauge frame, axes)
    unfilled instead of painting them grey.
    """
    copy = mobject.copy()
    for part in copy.family_members_with_points():
        part.set_stroke(opacity=part.get_stroke_opacity() * factor)
        part.set_fill(opacity=part.get_fill_opacity() * factor)
    return copy


class S4WrapUp(DeckSlide):
    """Section 4: recap, limits, thanks, and the appendix."""

    section_number = 4
    section_name = "Wrap-up"

    def construct(self) -> None:
        self.slide_4_0_section()
        self.slide_4_1_recap()
        self.slide_4_2_limits()
        self.slide_4_3_closing()
        self.slide_a_1_derivation()
        self.slide_a_2_eigen_view()

    # --- 4.0 -----------------------------------------------------------------

    def slide_4_0_section(self) -> None:
        """Section slide: number, name and a one-line preview of the section."""
        self.begin_slide(
            notes="""
            To finish, a quick recap, a second look at the smooth curves from
            the start, and the limits of the method.
            """
        )
        self.show_section_slide("Recap, limits and questions")

    # --- 4.1 -----------------------------------------------------------------

    def slide_4_1_recap(self) -> None:
        """Before/after strip with G in the middle. Point: the whole talk in one picture."""
        self.begin_slide(
            notes="""
            To recap: we start from independent noise that any computer can
            produce, factor our chosen Sigma into G times G transpose, and apply
            x equals mu plus G z. The result has exactly the correlation
            structure we asked for.
            """
        )
        title = th.slide_title("Recap")
        # Two lines: at body size the sentence is wider than the frame.
        message = VGroup(
            th.text("Cholesky turns independent noise", th.BODY_SIZE),
            th.text("into the correlation structure you chose.", th.BODY_SIZE),
        ).arrange(DOWN, buff=0.15)
        message.next_to(title, DOWN, buff=0.4).set_x(0)

        half_width = mm.shared_half_width(
            [IDENTITY, cfg.MAIN_SIGMA], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        before_axes = make_square_axes((0.0, 0.0), half_width, th.PANEL_STRIP)
        after_axes = make_square_axes(cfg.MAIN_MU, half_width, th.PANEL_STRIP)
        before = VGroup(
            VGroup(before_axes, make_point_cloud(self.z_display, before_axes, th.COLOR_Z)),
            th.math(r"\mathcal{N}(\mathbf{0}, I)", th.CAPTION_SIZE, th.COLOR_Z),
        ).arrange(DOWN, buff=0.2)
        after_cloud = make_point_cloud(
            mm.transform(self.z_display, cfg.MAIN_MU, self.main_chol), after_axes, th.COLOR_X
        )
        after_ellipse = make_cov_ellipse(
            cfg.MAIN_SIGMA, cfg.MAIN_MU, cfg.QUESTION_ELLIPSE_K, after_axes
        )
        after = VGroup(
            VGroup(after_axes, after_cloud, after_ellipse),
            th.math(r"\mathcal{N}(\boldsymbol{\mu}, \Sigma)", th.CAPTION_SIZE, th.COLOR_X),
        ).arrange(DOWN, buff=0.2)
        arrow = Arrow(LEFT, RIGHT, color=th.COLOR_G, stroke_width=6).scale(1.1)
        arrow_label = th.math(
            r"\mathbf{x} = \boldsymbol{\mu} + G\mathbf{z}", th.CAPTION_SIZE, th.COLOR_G
        )
        middle = VGroup(arrow_label, arrow).arrange(DOWN, buff=0.15)
        strip = VGroup(before, middle, after).arrange(RIGHT, buff=0.6)
        strip.next_to(message, DOWN, buff=0.5).set_x(0)

        self.play(FadeIn(title), Write(message), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(before), run_time=cfg.T_MEDIUM)
        self.play(GrowArrow(arrow), FadeIn(arrow_label), run_time=cfg.T_SHORT)
        self.play(FadeIn(after), run_time=cfg.T_MEDIUM)

    # --- 4.2 -----------------------------------------------------------------

    def slide_4_2_limits(self) -> None:
        """Three limits, each with a callback. Point: know when Cholesky is the wrong tool."""
        self.begin_slide(
            notes="""
            Three limits. First, Cholesky needs every pivot strictly positive;
            at rho equal to 1 the pivot hit zero and it stopped. Second, the
            eigen square root (the eigen view, in the appendix) still works there, because
            the square root of a zero eigenvalue is just zero, which is why
            NumPy's multivariate normal uses SVD by default, even though it is
            slower. Third, this only makes Gaussian data, and real returns have
            fat tails, but the usual fixes, like multivariate t and Gaussian
            copulas, still get their correlation from the same x equals mu plus G z.
            """
        )
        title = th.slide_title("Limits")
        rows = VGroup(
            self._limit_row(
                "Needs strictly positive pivots.",
                [
                    th.text(
                        "At ρ = 1 the pivot is 0 and Cholesky stops.",
                        th.CAPTION_SIZE,
                        th.COLOR_MUTED,
                    )
                ],
                self._gauge_thumbnail(),
            ),
            self._limit_row(
                "Alternatives: eigendecomposition and SVD.",
                [
                    self._eigen_line(),
                    th.text(
                        "NumPy's multivariate_normal defaults to SVD. Slower.",
                        th.CAPTION_SIZE,
                        th.COLOR_MUTED,
                    ),
                ],
                None,
            ),
            self._limit_row(
                "Gaussian only.",
                [
                    th.text(
                        "Real returns have fat tails: the portfolio risk",
                        th.CAPTION_SIZE,
                        th.COLOR_MUTED,
                    ),
                    th.text("can still be understated.", th.CAPTION_SIZE, th.COLOR_MUTED),
                    self._fixes_line(),
                ],
                self._portfolio_thumbnail(),
            ),
        )
        # Lay out the text only; thumbnails are placed afterwards, level with their row.
        text_blocks = VGroup(*[row.text_block for row in rows])
        text_blocks.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        text_blocks.next_to(title, DOWN, buff=0.45).align_to(title, LEFT).shift(0.2 * RIGHT)
        for row in rows:
            if row.thumbnail is not None:
                row.thumbnail.move_to(row.text_block, coor_mask=UP).set_x(th.LIMITS_THUMB_X)

        self.play(FadeIn(title), run_time=cfg.T_SHORT)
        for index, row in enumerate(rows):
            if index > 0:
                self.reveal()
                # Dim the earlier rows so only one limit is in focus at a time.
                self.play(
                    Transform(rows[index - 1], dimmed(rows[index - 1], th.LIMITS_DIM_OPACITY)),
                    run_time=cfg.T_SHORT,
                )
            self.play(FadeIn(row, shift=0.2 * RIGHT), run_time=cfg.T_MEDIUM)

    def _limit_row(self, headline: str, details: list[Mobject], thumbnail) -> VGroup:
        """A teal bullet and headline, muted detail lines under it, an optional thumbnail."""
        bullet = Dot(radius=th.BULLET_RADIUS, color=th.COLOR_SIGMA)
        head = VGroup(bullet, th.text(headline, th.BODY_SIZE)).arrange(RIGHT, buff=0.3)
        detail_block = VGroup(*details).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        detail_block.next_to(head[1], DOWN, buff=0.18, aligned_edge=LEFT)
        row = VGroup(head, detail_block)
        row.text_block = VGroup(head, detail_block)
        row.thumbnail = thumbnail
        if thumbnail is not None:
            row.add(thumbnail)
        return row

    def _gauge_thumbnail(self) -> VGroup:
        """Small copy of the 3.3 pivot gauge, frozen at rho = 1 (pivot 0, amber)."""
        rho = cfg.RHO_ZERO_PIVOT
        pivot = mm.unit_corr_pivot(rho)
        gauge = make_pivot_gauge(pivot, scale=th.LIMITS_THUMB_SCALE)
        # Drop the gauge's own "1" and "0" ticks: scaled down they fall below MIN_SIZE.
        frame, zero_line, bar, _ = gauge
        tags = VGroup(
            th.math(rf"\rho = {fmt(rho)}", th.MIN_SIZE, th.COLOR_WARN),
            th.math(rf"\text{{pivot}} = {fmt(pivot)}", th.MIN_SIZE, th.COLOR_WARN),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        tags.next_to(zero_line, RIGHT, buff=0.2)
        # A zero pivot has no bar to see, so the zero line itself turns amber.
        zero_line.set_stroke(th.COLOR_WARN, width=4)
        return VGroup(frame, zero_line, bar, tags)

    def _fixes_line(self) -> MathTex:
        """'Fixes (multivariate t, copulas) still use x = mu + G z', with G in yellow."""
        line = MathTex(
            r"\text{Fixes (multivariate } t\text{, copulas) still use }"
            r"\mathbf{x} = \boldsymbol{\mu} + ",
            "G",
            r"\mathbf{z}",
            font_size=th.CAPTION_SIZE * th.MATH_SCALE,
            color=th.COLOR_TEXT,
        )
        line[1].set_color(th.COLOR_G)
        return line

    def _eigen_line(self) -> MathTex:
        """'At rho = 1: lambda = 2 and 0, and sqrt(0) = 0 is fine (appendix)'."""
        values, _ = mm.eigen_axes(mm.correlation_matrix(cfg.RHO_ZERO_PIVOT))
        shown = r" \text{ and } ".join(fmt(v) for v in values)
        return th.math(
            rf"\rho = {fmt(cfg.RHO_ZERO_PIVOT)}:\ \lambda = {shown},"
            rf"\ \sqrt{{{fmt(values[-1])}}} = {fmt(np.sqrt(max(values[-1], 0.0)))}"
            r"\text{ is fine (appendix)}",
            th.CAPTION_SIZE,
            th.COLOR_MUTED,
        )

    def _portfolio_thumbnail(self) -> VGroup:
        """Small copy of the 1.1 portfolio histogram (correlated stocks) with its VaR line."""
        returns = mm.portfolio_returns(
            cfg.PORTFOLIO_WEIGHTS, cfg.PORTFOLIO_RHO_CORRELATED, cfg.N_STATS, cfg.RANDOM_SEED
        )
        sd = float(np.std(returns))
        x_lim = mm.axis_limits_1d([0.0], [sd], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        peak = float(mm.normal_pdf_1d(0.0, 0.0, sd))
        axes = Axes(
            x_range=[x_lim[0], x_lim[1], 1],
            y_range=[0, 1.2 * peak, 1.2 * peak],
            x_length=th.LIMITS_THUMB_WIDTH,
            y_length=th.LIMITS_THUMB_HEIGHT,
            tips=False,
            axis_config={"color": th.COLOR_MUTED, "stroke_width": th.AXIS_STROKE},
            y_axis_config={"include_ticks": False},
            x_axis_config={"include_ticks": False},
        )
        bars = make_histogram(returns, axes, cfg.PORTFOLIO_HIST_BINS, th.COLOR_X)
        var = mm.value_at_risk(returns, cfg.VAR_LEVEL)
        var_line = DashedLine(axes.c2p(var, 0), axes.c2p(var, 1.1 * peak), color=th.COLOR_BAD)
        tag = th.text("from the start", th.MIN_SIZE, th.COLOR_MUTED).next_to(axes, DOWN, buff=0.1)
        return VGroup(axes, bars, var_line, tag)

    # --- 4.3 -----------------------------------------------------------------

    def slide_4_3_closing(self) -> None:
        """Thank you and questions."""
        self.begin_slide(
            notes="""
            Thank you for listening. We are happy to take any questions, and we
            can jump back to any slide if it helps.
            """
        )
        thanks = th.text("Thank you", th.DECK_TITLE_SIZE)
        questions = th.text("Questions?", th.TITLE_SIZE, th.COLOR_SIGMA)
        VGroup(thanks, questions).arrange(DOWN, buff=0.5)
        self.play(Write(thanks), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(questions, shift=0.2 * UP), run_time=cfg.T_SHORT)

    # --- A.1 -----------------------------------------------------------------

    def slide_a_1_derivation(self) -> None:
        """Backup: change of variables on x = mu + G z gives the density, checked numerically."""
        self.begin_slide(
            notes="""
            A backup slide, in case anyone asks where the density formula comes
            from. If we run x equals mu plus G z backwards with the change of
            variables formula, the determinant of G gives the square root of
            the determinant of Sigma, and the squared length of z becomes the
            quadratic form. That is exactly the density from slide 1.4, and at
            our worked point both sides give the same numbers.
            """
        )
        title = th.slide_title("Appendix: the derivation is the same transformation")
        lines = VGroup(
            th.math(
                r"f_{\mathbf{x}}(\mathbf{x}) = "
                r"\frac{f_{\mathbf{z}}\big(G^{-1}(\mathbf{x}-\boldsymbol{\mu})\big)}{|\det G|}"
            ),
            th.math(r"|\det G| = |\Sigma|^{1/2}"),
            th.math(
                r"\big\|G^{-1}(\mathbf{x}-\boldsymbol{\mu})\big\|^2"
                r" = (\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})"
            ),
            th.text(
                "Together: exactly the density we started with.", th.CAPTION_SIZE, th.COLOR_MUTED
            ),
        ).arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        lines.next_to(title, DOWN, buff=0.35).set_x(-6.2, direction=LEFT)

        self.play(FadeIn(title), run_time=cfg.T_SHORT)
        for index, line in enumerate(lines):
            if index > 0:
                self.reveal()
            self.play(FadeIn(line, shift=0.2 * DOWN), run_time=cfg.T_MEDIUM)
        self.reveal()

        callback = self._worked_point_check()
        callback.to_edge(RIGHT, buff=th.EDGE_BUFF).align_to(lines, UP)
        self.play(FadeIn(callback, shift=0.2 * UP), run_time=cfg.T_MEDIUM)
        self.play(
            FadeIn(th.caption("One equation derives the distribution and generates the data.")),
            run_time=cfg.T_SHORT,
        )

    def _worked_point_check(self) -> VGroup:
        """Box with the quadratic form, ||z||^2 and the density at the worked point."""
        worked_z = np.asarray(cfg.WORKED_Z)
        worked_x = mm.transform(worked_z, cfg.MAIN_MU, self.main_chol)
        quad = mm.mahalanobis_sq(worked_x, cfg.MAIN_MU, cfg.MAIN_SIGMA)
        z_norm_sq = float(worked_z @ worked_z)
        density = mm.mvn_density(worked_x, cfg.MAIN_MU, cfg.MAIN_SIGMA)
        size = th.CAPTION_SIZE
        content = VGroup(
            th.math(rf"\text{{At }} \mathbf{{x}} = {fmt_vector(worked_x)},", size),
            th.math(rf"\text{{from }} \mathbf{{z}} = {fmt_vector(worked_z)}:", size),
            th.math(
                r"(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})"
                rf" = {fmt(quad)}",
                size,
                th.COLOR_GOOD,
            ),
            th.math(rf"\|\mathbf{{z}}\|^2 = {fmt(z_norm_sq)}", size, th.COLOR_GOOD),
            th.math(
                rf"f(\mathbf{{x}}) \approx {fmt(density, cfg.DENSITY_DECIMALS, fixed=True)}",
                size,
                th.COLOR_GOOD,
            ),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        return boxed(content, buff=0.25)

    # --- A.2 -----------------------------------------------------------------

    def slide_a_2_eigen_view(self) -> None:
        """Backup: Sigma applied to test arrows, then eigen arrows. Point: read the shape."""
        self.begin_slide(
            notes="""
            Cholesky told us how to build this cloud. Eigenvectors tell us how
            to read it. Most arrows turn when I multiply them by Sigma, but two
            directions only stretch: the diagonal by 9 and the other diagonal by
            1. Their square roots, 3 and 1, are exactly how spread out the cloud
            is along each axis.
            """
        )
        sigma = np.asarray(cfg.EIGEN_EXAMPLE_SIGMA, dtype=float)
        title = th.slide_title("Appendix: the eigen view")
        examples = [sigma]
        if cfg.ENABLE_EIGEN_SECOND_EXAMPLE:
            examples.append(np.asarray(cfg.EIGEN_EXAMPLE_SIGMA_2, dtype=float))
        half_width = mm.shared_half_width(examples, cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        axes = make_square_axes((0.0, 0.0), half_width, th.EIGEN_PANEL).move_to([-3.75, -0.3, 0])
        # Morphs between the two examples by interpolating Sigma; every in-between
        # matrix is positive definite, so Cholesky works on each frame.
        blend = ValueTracker(0.0)

        def current_sigma() -> np.ndarray:
            return (1 - blend.get_value()) * examples[0] + blend.get_value() * examples[-1]

        cloud = make_point_cloud(
            mm.transform(self.z_display, np.zeros(2), mm.cholesky_factor(sigma)),
            axes,
            th.COLOR_X,
            panel_dot_radius(th.EIGEN_PANEL),
        )
        sigma_block = make_labeled_matrix(r"\Sigma", sigma, th.COLOR_SIGMA, th.CAPTION_SIZE)
        sigma_block.move_to([0.0, 2.35, 0]).set_x(-0.6, direction=LEFT)
        question = th.caption("Which directions does this cloud stretch along, and by how much?")

        self.play(FadeIn(title), Create(axes), FadeIn(sigma_block), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(cloud), FadeIn(question), run_time=cfg.T_MEDIUM)
        self.reveal()

        tests = self._direction_tests(sigma, sigma_block, question)
        self.reveal()

        self.play(FadeOut(tests), run_time=cfg.T_SHORT)
        ellipse = always_redraw(
            lambda: make_cov_ellipse(current_sigma(), (0.0, 0.0), cfg.EIGEN_ARROW_K, axes)
        )
        arrows = always_redraw(lambda: self._eigen_arrows(current_sigma(), axes))
        rows = self._eigen_rows(sigma).next_to(sigma_block, DOWN, buff=0.6, aligned_edge=LEFT)
        roots = np.sqrt(mm.eigen_axes(sigma)[0])
        caption = th.caption(
            "Eigenvectors set the ellipse's axes. √eigenvalues set the spread: "
            f"{fmt(roots[0])} one way, {fmt(roots[1])} the other."
        )
        self.play(Create(ellipse), run_time=cfg.T_MEDIUM)
        self.play(
            GrowFromPoint(arrows, axes.c2p(0, 0)),
            FadeIn(rows),
            swap_caption(self.eigen_caption, caption),
            run_time=cfg.T_MEDIUM,
        )

        if cfg.ENABLE_EIGEN_SECOND_EXAMPLE:
            self.reveal()
            second = examples[-1]
            live_cloud = always_redraw(
                lambda: make_point_cloud(
                    mm.transform(self.z_display, np.zeros(2), mm.cholesky_factor(current_sigma())),
                    axes,
                    th.COLOR_X,
                    panel_dot_radius(th.EIGEN_PANEL),
                )
            )
            self.remove(cloud)
            self.add(live_cloud, ellipse, arrows)
            new_block = make_labeled_matrix(r"\Sigma", second, th.COLOR_SIGMA, th.CAPTION_SIZE)
            new_block.move_to(sigma_block, aligned_edge=LEFT)
            new_rows = self._eigen_rows(second).move_to(rows, aligned_edge=UL)
            new_values = mm.eigen_axes(second)[0]
            new_caption = th.caption(
                f"Not just a 45° trick: eigenvalues {fmt(new_values[0])} and "
                f"{fmt(new_values[1])}, so spreads {fmt(np.sqrt(new_values[0]))} "
                f"and {fmt(np.sqrt(new_values[1]))}."
            )
            self.play(
                blend.animate.set_value(1.0),
                Transform(sigma_block, new_block),
                Transform(rows, new_rows),
                swap_caption(caption, new_caption),
                run_time=cfg.T_LONG,
            )
            # Not a frozen frame: rerun the updaters so everything shows the exact final Sigma.
            self.wait(cfg.T_SETTLE, frozen_frame=False)
            for live in (live_cloud, ellipse, arrows):
                live.clear_updaters()

    def _direction_tests(self, sigma: np.ndarray, anchor: Mobject, question: Mobject) -> VGroup:
        """Apply Sigma to each test direction on a small plane, one click per direction.

        Returns everything it drew, so the caller can clear it. Leaves the
        current caption in ``self.eigen_caption``.
        """
        images = [sigma @ np.asarray(v) for v in cfg.EIGEN_TEST_DIRECTIONS]
        coords = np.concatenate([np.asarray(cfg.EIGEN_TEST_DIRECTIONS), images])
        low = float(np.floor(min(coords.min(), 0.0))) - 1
        high = float(np.ceil(coords.max())) + 1
        plane = make_axes((low, high), (low, high), th.EIGEN_TEST_PANEL, th.EIGEN_TEST_PANEL)
        plane.move_to([*th.EIGEN_TEST_CENTER, 0])
        plane_label = th.math(r"\text{multiply by }\Sigma", th.MIN_SIZE, th.COLOR_SIGMA)
        plane_label.next_to(plane, UP, buff=0.1)
        readouts = VGroup()
        drawn = VGroup(plane, plane_label, readouts)
        self.play(Create(plane), FadeIn(plane_label), run_time=cfg.T_SHORT)

        captions = [
            "Most directions get rotated.",
            "This one only stretches.",
            "And so does this one.",
        ]
        caption = question
        previous = VGroup()
        for index, (vector, image) in enumerate(
            zip(cfg.EIGEN_TEST_DIRECTIONS, images, strict=True)
        ):
            if index > 0:
                self.reveal()
            vector = np.asarray(vector)
            before = Arrow(plane.c2p(0, 0), plane.c2p(*vector), buff=0, color=th.COLOR_TEXT)
            after = Arrow(plane.c2p(0, 0), plane.c2p(*image), buff=0, color=th.COLOR_SIGMA)
            for arrow in (before, after):
                arrow.set_stroke(width=5)
            readout = th.math(self._readout_tex(vector, image), th.CAPTION_SIZE)
            if readouts:
                readout.next_to(readouts, DOWN, buff=0.3, aligned_edge=LEFT)
            else:
                # Below the test plane's top edge would collide with it, so the
                # readouts start low on the slide, under the Sigma matrix's column.
                readout.move_to([0, th.EIGEN_READOUT_Y, 0]).align_to(anchor, LEFT)
            readouts.add(readout)
            new_caption = th.caption(captions[min(index, len(captions) - 1)])
            self.play(
                previous.animate.set_opacity(0.3),
                GrowArrow(before),
                FadeIn(readout),
                swap_caption(caption, new_caption),
                run_time=cfg.T_SHORT,
            )
            self.play(TransformFromCopy(before, after), run_time=cfg.T_MEDIUM)
            previous = VGroup(before, after)
            drawn.add(previous)
            caption = new_caption
        self.reveal()
        key = th.math(r"\Sigma\mathbf{v} = \lambda\mathbf{v}", th.BODY_SIZE, th.COLOR_SIGMA)
        key_box = SurroundingRectangle(key, color=th.COLOR_SIGMA, buff=0.15)
        VGroup(key, key_box).to_edge(RIGHT, buff=th.EDGE_BUFF).match_y(readouts)
        key_caption = th.caption(
            "Directions Σ only stretches: <b>eigenvectors</b>. "
            "The stretch factors: <b>eigenvalues</b>."
        )
        self.play(
            Write(key),
            Create(key_box),
            swap_caption(caption, key_caption),
            run_time=cfg.T_MEDIUM,
        )
        drawn.add(key, key_box)
        self.eigen_caption = key_caption
        return drawn

    @staticmethod
    def _readout_tex(vector: np.ndarray, image: np.ndarray) -> str:
        """'Sigma (1, 1) = (9, 9) = 9 (1, 1)' if the image is parallel, else 'Sigma v = image'."""
        tex = rf"\Sigma {fmt_vector(vector)} = {fmt_vector(image)}"
        cross = vector[0] * image[1] - vector[1] * image[0]
        if abs(cross) < 1e-9:
            factor = float(image @ vector / (vector @ vector))
            tex += rf" = {fmt(factor)} \cdot {fmt_vector(vector)}"
        return tex

    @staticmethod
    def _eigen_arrows(sigma: np.ndarray, axes: Axes) -> VGroup:
        """Both eigenvectors from the center, scaled by k sqrt(lambda), labeled at the tips."""
        values, vectors = mm.eigen_axes(sigma)
        group = VGroup()
        for index, (value, vector) in enumerate(zip(values, vectors.T, strict=True), start=1):
            tip = cfg.EIGEN_ARROW_K * np.sqrt(value) * vector
            arrow = Arrow(axes.c2p(0, 0), axes.c2p(*tip), buff=0, color=th.COLOR_TEXT)
            arrow.set_stroke(width=6)
            label = th.math(rf"\mathbf{{v}}_{index}", th.CAPTION_SIZE)
            label.add_background_rectangle(color=th.BACKGROUND, opacity=0.85, buff=0.05)
            label.next_to(axes.c2p(*tip), normalize(arrow.get_vector()), buff=0.12)
            group.add(VGroup(arrow, label))
        return group

    @staticmethod
    def _eigen_rows(sigma: np.ndarray) -> VGroup:
        """One row per eigenpair: lambda, its square root, and the unit eigenvector."""
        values, vectors = mm.eigen_axes(sigma)
        rows = VGroup(
            *[
                th.math(
                    rf"\lambda_{i} = {fmt(value)},\ \sqrt{{\lambda_{i}}} = {fmt(np.sqrt(value))},"
                    rf"\ \mathbf{{v}}_{i} \approx {fmt_vector(vector, fixed=True)}",
                    th.CAPTION_SIZE,
                )
                for i, (value, vector) in enumerate(zip(values, vectors.T, strict=True), start=1)
            ]
        )
        return rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
