"""Section 3: the demo (brief part b).

Check that the sampler works, then vary Sigma and watch the cloud change:
side-by-side structures, a live correlation sweep, and the eigen view.
"""

import sys
import textwrap
from pathlib import Path

# manim-slides loads this file by path, so the project root is not importable
# by default. Put it first on sys.path so config, theme, ... resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from manim import *
from manim_slides import ThreeDSlide

import config as cfg
import mvn_math as mm
import theme as th
from components import (
    DeckSlide,
    Panel,
    fade_in_cloud,
    fmt,
    fmt_vector,
    make_cov_ellipse,
    make_labeled_matrix,
    make_point_cloud,
    make_square_axes,
    make_symbolic_matrix,
    panel_dot_radius,
)

PANEL_CAPTION_CHARS = 16  # wrap per-panel captions on slide 3.2 at this many characters


class S3Demo(DeckSlide):
    """Section 3: verification, covariance structures, rho sweep, eigen view."""

    section_number = 3
    section_name = "Demo"

    def construct(self) -> None:
        self.slide_3_0_section()
        self.slide_3_1_verification()
        self.slide_3_2_structures_side_by_side()
        self.slide_3_3_correlation_sweep()
        self.slide_3_4_eigen_view()

    # --- 3.0 -----------------------------------------------------------------

    def slide_3_0_section(self) -> None:
        """Section slide: number, name and a one-line preview of the section."""
        self.begin_slide(
            notes="""
            Time for the demo, which is part b of the brief. First we check that
            the sampler works, then we change Sigma and watch how the shape and
            spread of the data respond.
            """
        )
        self.show_section_slide("Brief part (b): change Σ, watch the data")

    # --- 3.1 -----------------------------------------------------------------

    def slide_3_1_verification(self) -> None:
        """Target vs sample statistics with density contours. Point: the sampler works."""
        self.begin_slide(
            notes=f"""
            Before playing with it, let's check that it works. We drew
            {cfg.N_STATS:,} samples and measured their mean and covariance; the
            dots you see are the first {cfg.N_DISPLAY} of them. The measured
            values match the targets to about two decimal places, and the cloud
            sits inside the 1 and 2 sigma contours of the true density.
            """
        )
        title = th.slide_title("Does the sampler work?")
        samples = mm.transform(self.z_stats, cfg.MAIN_MU, self.main_chol)
        half_width = mm.shared_half_width([cfg.MAIN_SIGMA], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        axes = make_square_axes(cfg.MAIN_MU, half_width, th.PANEL_LARGE).move_to([-3.6, -0.1, 0])
        cloud = make_point_cloud(samples[: cfg.N_DISPLAY], axes, th.COLOR_X)
        contours = VGroup(
            *[make_cov_ellipse(cfg.MAIN_SIGMA, cfg.MAIN_MU, k, axes) for k in cfg.CONTOUR_KS]
        )
        contour_note = th.math(
            r",\ ".join(rf"{fmt(k)}\sigma" for k in cfg.CONTOUR_KS) + r"\text{ contours}",
            th.MIN_SIZE,
            th.COLOR_SIGMA,
        ).next_to(axes, DOWN, buff=0.12)

        size = th.CAPTION_SIZE
        mu_column = [[m] for m in cfg.MAIN_MU]
        mu_hat_column = [[m] for m in mm.sample_mean(samples)]
        table = VGroup(
            th.text("Target", size, th.COLOR_MUTED),
            th.text(f"From {cfg.N_STATS:,} samples", size, th.COLOR_MUTED),
            make_labeled_matrix(r"\Sigma", cfg.MAIN_SIGMA, th.COLOR_SIGMA, size),
            make_labeled_matrix(
                r"\hat{\Sigma}", mm.sample_cov(samples), th.COLOR_X, size, fixed=True
            ),
            make_labeled_matrix(r"\boldsymbol{\mu}", mu_column, th.COLOR_TEXT, size),
            make_labeled_matrix(
                r"\hat{\boldsymbol{\mu}}", mu_hat_column, th.COLOR_X, size, fixed=True
            ),
        ).arrange_in_grid(rows=3, cols=2, buff=(0.7, 0.5), col_alignments="ll")
        table.move_to([3.4, -0.2, 0])

        self.play(FadeIn(title), Create(axes), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(cloud))
        self.play(Create(contours), FadeIn(contour_note), run_time=cfg.T_MEDIUM)
        self.reveal()
        self.play(FadeIn(table[:2]), FadeIn(table[2]), FadeIn(table[4]), run_time=cfg.T_SHORT)
        self.play(FadeIn(table[3], shift=0.2 * LEFT), FadeIn(table[5], shift=0.2 * LEFT))
        self.play(FadeIn(th.caption("The sampler reproduces Σ.")), run_time=cfg.T_SHORT)

    # --- 3.2 -----------------------------------------------------------------

    def slide_3_2_structures_side_by_side(self) -> None:
        """Four covariance structures from the same z. Point: Sigma alone sets the shape."""
        self.begin_slide(
            notes="""
            Now part b of the brief: different covariance structures. Every
            panel uses exactly the same z samples, so the only thing changing is
            Sigma. Equal variances give a circle, unequal variances stretch it
            along an axis, and a positive or negative correlation tilts it up or down.
            """
        )
        title = th.slide_title("Covariance structures side by side")
        sigmas = [sigma for _, sigma in cfg.DEMO_STRUCTURES]
        half_width = mm.shared_half_width(sigmas, cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        panels = VGroup()
        for (name, sigma), caption in zip(cfg.DEMO_STRUCTURES, cfg.DEMO_CAPTIONS, strict=True):
            panel = Panel(
                name.replace("rho", "ρ"),
                sigma,
                self.z_display,
                half_width,
                th.PANEL_SMALL,
                matrix_side=RIGHT,
            )
            wrapped = textwrap.fill(caption, PANEL_CAPTION_CHARS, break_on_hyphens=False)
            panel.caption = th.text(wrapped, th.MIN_SIZE, th.COLOR_X)
            notes = VGroup(panel.matrix, panel.caption).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            notes.next_to(panel.axes, RIGHT, buff=0.3)
            panel.label.next_to(panel.axes, UP, buff=0.08).align_to(panel.axes, LEFT)
            panel.add(panel.caption)
            panels.add(panel)
        panels.arrange_in_grid(rows=2, cols=2, buff=(0.9, 0.2), cell_alignment=LEFT)
        panels.next_to(title, DOWN, buff=0.15).set_x(0)
        same_z = th.caption("Same z samples in every panel. Only Σ changes.")

        self.play(FadeIn(title), FadeIn(same_z), run_time=cfg.T_SHORT)
        for index, panel in enumerate(panels):
            if index > 0:
                self.reveal()
            self.play(
                FadeIn(panel.label), Create(panel.axes), FadeIn(panel.matrix), run_time=cfg.T_SHORT
            )
            self.play(fade_in_cloud(panel.cloud), run_time=cfg.T_MEDIUM)
            self.play(FadeIn(panel.caption), run_time=cfg.T_SHORT)

    # --- 3.3 -----------------------------------------------------------------

    def slide_3_3_correlation_sweep(self) -> None:
        """Live rho sweep from fixed z, ending at rho = 1. Point: correlation tilts and squeezes."""
        self.begin_slide(
            notes="""
            Here we sweep the correlation live. The samples z never change; only
            rho does, through the mixing recipe from earlier. As rho approaches 1
            the fresh-noise term shrinks and the cloud collapses onto a line. At
            exactly 1 there is no fresh noise left, Sigma is singular, and
            Cholesky fails.
            """
        )
        title = th.slide_title("Correlation sweep")
        start, *targets = cfg.RHO_SWEEP_PATH
        rho = ValueTracker(start)
        half_width = mm.shared_half_width(
            [mm.correlation_matrix(0.0)], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        axes = make_square_axes((0.0, 0.0), half_width, th.PANEL_SWEEP).move_to([-3.3, -0.35, 0])
        dot_radius = panel_dot_radius(th.PANEL_SWEEP)
        # Redraw from the same z every frame, so the cloud morphs instead of re-randomizing.
        cloud = always_redraw(
            lambda: make_point_cloud(
                mm.transform(self.z_display, np.zeros(2), mm.mixing_factor(rho.get_value())),
                axes,
                th.COLOR_X,
                dot_radius,
            )
        )
        readouts = self._sweep_readouts(rho).move_to([3.4, 1.0, 0])

        self.play(FadeIn(title), Create(axes), FadeIn(readouts), run_time=cfg.T_SHORT)
        self.add(cloud)
        self.play(FadeIn(cloud), run_time=cfg.T_SHORT)
        for target in targets:
            self.reveal()
            self.play(
                rho.animate.set_value(target),
                run_time=cfg.RHO_SWEEP_SECONDS_PER_LEG,
                rate_func=smooth,
            )
            # Not a frozen frame: run the updaters once more so the readouts and
            # cloud show the exact target value, not the last in-between frame.
            self.wait(cfg.T_SETTLE, frozen_frame=False)
        self.reveal()

        self.play(rho.animate.set_value(cfg.RHO_FAILURE), run_time=cfg.T_MEDIUM)
        self.wait(cfg.T_SETTLE, frozen_frame=False)
        sqrt_value = readouts[1][1]
        singular = mm.correlation_matrix(cfg.RHO_FAILURE)
        failure = VGroup(
            make_labeled_matrix(r"\Sigma", singular, th.COLOR_SIGMA, th.CAPTION_SIZE),
            th.math(
                rf"\det(\Sigma) = {fmt(np.linalg.det(singular))}", th.CAPTION_SIZE, th.COLOR_BAD
            ),
        ).arrange(RIGHT, buff=0.5)
        failure.next_to(readouts, DOWN, buff=0.7)
        self.play(
            sqrt_value.animate.set_color(th.COLOR_BAD), FadeIn(failure), run_time=cfg.T_MEDIUM
        )
        self.play(
            FadeIn(th.caption("Cholesky fails: Σ is not positive definite.", th.COLOR_BAD)),
            run_time=cfg.T_SHORT,
        )
        # Freeze the cloud: an always_redraw mobject would otherwise ignore the
        # fade-out at the start of the next slide (it rebuilds itself every frame).
        cloud.clear_updaters()

    def _sweep_readouts(self, rho: ValueTracker) -> VGroup:
        """Live 'rho = ...' and 'sqrt(1 - rho^2) = ...' rows, plus the L they fill in."""
        size = th.BODY_SIZE * th.MATH_SCALE
        rho_value = DecimalNumber(
            rho.get_value(), num_decimal_places=cfg.SWEEP_DECIMALS, font_size=size
        )
        rho_value.add_updater(lambda m: m.set_value(rho.get_value()))
        sqrt_value = DecimalNumber(1.0, num_decimal_places=cfg.SWEEP_DECIMALS, font_size=size)
        sqrt_value.set_color(th.COLOR_L)
        sqrt_value.add_updater(lambda m: m.set_value(mm.mixing_factor(rho.get_value())[1, 1]))

        rho_row = VGroup(th.math(r"\rho ="), rho_value).arrange(RIGHT, buff=0.2)
        sqrt_row = VGroup(th.math(r"\sqrt{1-\rho^2} =", color=th.COLOR_L), sqrt_value)
        sqrt_row.arrange(RIGHT, buff=0.2)
        l_block = VGroup(
            th.math("L =", color=th.COLOR_L),
            make_symbolic_matrix(
                [["1", "0"], [r"\rho", r"\sqrt{1-\rho^2}"]], th.COLOR_L, th.CAPTION_SIZE
            ),
        ).arrange(RIGHT, buff=0.2)
        return VGroup(rho_row, sqrt_row, l_block).arrange(DOWN, buff=0.4, aligned_edge=LEFT)

    # --- 3.4 -----------------------------------------------------------------

    def slide_3_4_eigen_view(self) -> None:
        """Eigenvectors of Sigma on the cloud. Point: the eigen-decomposition gives the ellipse."""
        self.begin_slide(
            notes="""
            One more way to read Sigma. Its eigenvectors point along the axes
            of the ellipse, and the eigenvalues say how much variance lies along
            each one. The arrows here are two standard deviations long, so they
            reach exactly to the 2 sigma contour.
            """
        )
        title = th.slide_title("The eigen view")
        values, vectors = mm.eigen_axes(cfg.MAIN_SIGMA)
        samples = mm.transform(self.z_display, cfg.MAIN_MU, self.main_chol)
        half_width = mm.shared_half_width([cfg.MAIN_SIGMA], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        axes = make_square_axes(cfg.MAIN_MU, half_width, th.PANEL_LARGE).move_to([-3.4, -0.35, 0])
        cloud = make_point_cloud(samples, axes, th.COLOR_X).set_opacity(0.45)
        ellipse = make_cov_ellipse(cfg.MAIN_SIGMA, cfg.MAIN_MU, cfg.EIGEN_ARROW_K, axes)

        arrows, tip_labels, rows = VGroup(), VGroup(), VGroup()
        center = np.asarray(cfg.MAIN_MU)
        for index, (value, vector) in enumerate(zip(values, vectors.T), start=1):
            tip = center + cfg.EIGEN_ARROW_K * np.sqrt(value) * vector
            arrow = Arrow(axes.c2p(*center), axes.c2p(*tip), buff=0, color=th.COLOR_TEXT)
            arrow.set_stroke(width=6)
            label = th.math(rf"\lambda_{index}", th.CAPTION_SIZE)
            label.add_background_rectangle(color=th.BACKGROUND, opacity=0.85, buff=0.05)
            label.next_to(axes.c2p(*tip), normalize(arrow.get_vector()), buff=0.12)
            arrows.add(arrow)
            tip_labels.add(label)
            rows.add(
                th.math(
                    rf"\lambda_{index} \approx {fmt(value)},\quad "
                    rf"\mathbf{{v}}_{index} \approx {fmt_vector(vector, fixed=True)}",
                    th.CAPTION_SIZE,
                )
            )
        sigma_block = make_labeled_matrix(r"\Sigma", cfg.MAIN_SIGMA, th.COLOR_SIGMA)
        side = VGroup(sigma_block, rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT))
        side.arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to([3.5, 0.0, 0])

        self.play(FadeIn(title), Create(axes), run_time=cfg.T_SHORT)
        self.play(fade_in_cloud(cloud), Create(ellipse), FadeIn(sigma_block))
        for arrow, label, row in zip(arrows, tip_labels, rows):
            self.reveal()
            self.play(GrowArrow(arrow), FadeIn(label), FadeIn(row), run_time=cfg.T_MEDIUM)
        self.play(
            FadeIn(
                th.caption(
                    "Eigenvectors set the ellipse's axes. Eigenvalues set the spread along them."
                )
            ),
            run_time=cfg.T_SHORT,
        )


class S3bThreeD(ThreeDSlide):
    """Optional slide 3.5: a correlated 3D cloud. Rendered only when ENABLE_3D_BONUS is True."""

    def construct(self) -> None:
        self.slide_3_5_three_d_bonus()

    def slide_3_5_three_d_bonus(self) -> None:
        """A 3D N(mu, Sigma) cloud with a slow camera orbit. Point: same recipe in 3D."""
        notes = """
            The same recipe works in any dimension. Here is a three dimensional
            Sigma, factored with Cholesky and applied to 3D standard normals.
            As the camera moves, you can see the cloud is a tilted ellipsoid.
            """
        self.next_slide(notes=notes)
        title = th.slide_title("Bonus: the same recipe in 3D")
        footer = th.footer(f"{S3Demo.section_number}  {S3Demo.section_name}")
        self.add_fixed_in_frame_mobjects(title, footer)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-50 * DEGREES)

        z = mm.standard_normal_samples(cfg.N_DISPLAY_3D, 3, cfg.RANDOM_SEED)
        samples = mm.transform(z, cfg.THREE_D_MU, mm.cholesky_factor(cfg.THREE_D_SIGMA))
        half_width = mm.shared_half_width([cfg.THREE_D_SIGMA], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        axis_range = [-half_width, half_width, 1]
        axes = ThreeDAxes(
            x_range=axis_range,
            y_range=axis_range,
            z_range=axis_range,
            x_length=5.5,
            y_length=5.5,
            z_length=5.5,
            axis_config={"color": th.COLOR_MUTED},
        )
        dots = VGroup(
            *[
                Dot3D(axes.c2p(*point), radius=th.DOT3D_RADIUS, color=th.COLOR_X, resolution=(4, 4))
                for point in samples
            ]
        )
        self.play(Create(axes), FadeIn(title), run_time=cfg.T_SHORT)
        self.play(FadeIn(dots, lag_ratio=0.002), run_time=cfg.T_MEDIUM)
        self.next_slide(notes=notes)
        self.move_camera(
            theta=-50 * DEGREES + cfg.THREE_D_ROTATION_DEGREES * DEGREES,
            run_time=cfg.THREE_D_ROTATION_SECONDS,
            rate_func=linear,
        )
