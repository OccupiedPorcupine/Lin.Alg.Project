"""Section 4: wrap-up.

Recap, applications, limits, and a bonus showing that the same equation
x = mu + L z also derives the density.
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
    make_point_cloud,
    make_square_axes,
)

IDENTITY = np.eye(2)


def bullet_list(*items: str, size: float = th.BODY_SIZE) -> VGroup:
    """Left-aligned items, each with a small teal bullet."""
    rows = VGroup()
    for item in items:
        bullet = Dot(radius=th.BULLET_RADIUS, color=th.COLOR_SIGMA)
        rows.add(VGroup(bullet, th.text(item, size)).arrange(RIGHT, buff=0.3))
    return rows.arrange(DOWN, buff=0.55, aligned_edge=LEFT)


class S4WrapUp(DeckSlide):
    """Section 4: recap, uses, limits, the density derivation, and thanks."""

    section_number = 4
    section_name = "Wrap-up"

    def construct(self) -> None:
        self.slide_4_0_section()
        self.slide_4_1_recap()
        self.slide_4_2_where_used()
        self.slide_4_3_limits()
        self.slide_4_4_bonus_derivation()
        self.slide_4_5_closing()

    # --- 4.0 -----------------------------------------------------------------

    def slide_4_0_section(self) -> None:
        """Section slide: number, name and a one-line preview of the section."""
        self.begin_slide(
            notes="""
            To finish, a quick recap, where this is used, its limits, and one bonus
            connection back to the density formula from the start.
            """
        )
        self.show_section_slide("Recap, uses and limits")

    # --- 4.1 -----------------------------------------------------------------

    def slide_4_1_recap(self) -> None:
        """Before/after strip with L in the middle. Point: the whole talk in one picture."""
        self.begin_slide(
            notes="""
            To recap: we start from independent noise that any computer can
            produce, factor our chosen Sigma into L times L transpose, and apply
            x equals mu plus L z. The result has exactly the correlation
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
        arrow = Arrow(LEFT, RIGHT, color=th.COLOR_L, stroke_width=6).scale(1.1)
        arrow_label = th.math(
            r"\mathbf{x} = \boldsymbol{\mu} + L\mathbf{z}", th.CAPTION_SIZE, th.COLOR_L
        )
        middle = VGroup(arrow_label, arrow).arrange(DOWN, buff=0.15)
        strip = VGroup(before, middle, after).arrange(RIGHT, buff=0.6)
        strip.next_to(message, DOWN, buff=0.5).set_x(0)

        self.play(FadeIn(title), Write(message), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(before), run_time=cfg.T_MEDIUM)
        self.play(GrowArrow(arrow), FadeIn(arrow_label), run_time=cfg.T_SHORT)
        self.play(FadeIn(after), run_time=cfg.T_MEDIUM)

    # --- 4.2 -----------------------------------------------------------------

    def slide_4_2_where_used(self) -> None:
        """Three applications. Point: this is a workhorse, not a toy."""
        self.begin_slide(
            notes="""
            This is not just a classroom trick. Banks use it to simulate
            correlated asset returns for risk, Gaussian processes use it to draw
            random functions, and it is a standard way to generate realistic
            synthetic data with a known correlation structure.
            """
        )
        title = th.slide_title("Where this is used")
        items = bullet_list(
            "Monte Carlo risk and portfolio simulation",
            "Gaussian processes",
            "Synthetic data generation",
        )
        items.next_to(title, DOWN, buff=0.9).align_to(title, LEFT).shift(0.6 * RIGHT)

        self.play(FadeIn(title), run_time=cfg.T_SHORT)
        for index, item in enumerate(items):
            if index > 0:
                self.reveal()
            self.play(FadeIn(item, shift=0.2 * RIGHT), run_time=cfg.T_SHORT)

    # --- 4.3 -----------------------------------------------------------------

    def slide_4_3_limits(self) -> None:
        """Three limitations. Point: know when Cholesky is the wrong tool."""
        self.begin_slide(
            notes="""
            There are limits. Cholesky needs Sigma to be strictly positive
            definite, as we saw with rho equal to 1. The output is always
            Gaussian, so it cannot produce skewed or fat-tailed data. When Sigma
            is close to singular, eigendecomposition or SVD are safer, and that
            is what NumPy's multivariate normal uses by default.
            """
        )
        title = th.slide_title("Limits")
        items = bullet_list(
            "Needs a positive definite Σ.",
            "Only Gaussian data: no skew and no fat tails.",
            "Alternatives: eigendecomposition and SVD.",
        )
        detail = th.text(
            "Slower, but more robust near singular Σ.", th.CAPTION_SIZE, th.COLOR_MUTED
        )
        detail.next_to(items[2][1], DOWN, buff=0.2, aligned_edge=LEFT)
        block = VGroup(items, detail)
        block.next_to(title, DOWN, buff=0.9).align_to(title, LEFT).shift(0.6 * RIGHT)
        caption = th.caption("NumPy's multivariate_normal defaults to SVD.")

        self.play(FadeIn(title), run_time=cfg.T_SHORT)
        self.play(FadeIn(items[0], shift=0.2 * RIGHT), run_time=cfg.T_SHORT)
        self.reveal()
        self.play(FadeIn(items[1], shift=0.2 * RIGHT), run_time=cfg.T_SHORT)
        self.reveal()
        self.play(FadeIn(items[2], shift=0.2 * RIGHT), FadeIn(detail), run_time=cfg.T_SHORT)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    # --- 4.4 -----------------------------------------------------------------

    def slide_4_4_bonus_derivation(self) -> None:
        """Change of variables on x = mu + L z gives the MVN density, checked numerically."""
        self.begin_slide(
            notes="""
            A bonus for the curious. If we run x equals mu plus L z backwards
            with the change of variables formula, the determinant of L gives the
            square root of the determinant of Sigma, and the squared length of z
            becomes the quadratic form. That is exactly the density from the
            start, and at our worked point both sides give the same numbers.
            """
        )
        title = th.slide_title("Bonus: the derivation is the same transformation")
        lines = VGroup(
            th.math(
                r"f_{\mathbf{x}}(\mathbf{x}) = "
                r"\frac{f_{\mathbf{z}}\big(L^{-1}(\mathbf{x}-\boldsymbol{\mu})\big)}{|\det L|}"
            ),
            th.math(r"|\det L| = |\Sigma|^{1/2}"),
            th.math(
                r"\big\|L^{-1}(\mathbf{x}-\boldsymbol{\mu})\big\|^2"
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

    # --- 4.5 -----------------------------------------------------------------

    def slide_4_5_closing(self) -> None:
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
