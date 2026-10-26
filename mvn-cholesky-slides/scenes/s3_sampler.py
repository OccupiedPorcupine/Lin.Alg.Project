"""Section 3: putting the sampler to work (brief part b).

Check that x = mu + G z really produces Sigma, then turn the dial on Sigma:
side-by-side structures, the eigen view of the shape, and a correlation
sweep that runs the Cholesky pipeline until it breaks.
"""

import sys
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
    HeroFrame,
    boxed,
    fade_in_cloud,
    fmt,
    fmt_vector,
    make_axes,
    make_cov_ellipse,
    make_dim_overlay,
    make_labeled_matrix,
    make_pivot_gauge,
    make_point_cloud,
    make_square_axes,
    make_symbolic_matrix,
    panel_dot_radius,
    pivot_color,
    through_data,
)


def make_deformable_grid(axes: Axes, half_width: float) -> NumberPlane:
    """A faint unit grid over z in [-half_width, half_width]^2, centered on the axes' origin.

    Applying G to it (through ``through_data``) shows how G moves the plane.
    """
    unit = axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0]
    grid = NumberPlane(
        x_range=[-half_width, half_width, 1],
        y_range=[-half_width, half_width, 1],
        x_length=2 * half_width * unit,
        y_length=2 * half_width * unit,
        background_line_style={"stroke_color": th.COLOR_TEXT, "stroke_opacity": 0.3},
        axis_config={"stroke_color": th.COLOR_TEXT, "stroke_opacity": 0.3},
    )
    return grid.shift(axes.c2p(0, 0) - grid.c2p(0, 0))


class LiveEntry(DecimalNumber):
    """A number that sits on a placeholder entry and follows ``value_fn`` every frame.

    The placeholder (an entry of a Matrix, sized for the widest value) is
    hidden; the number stays centered on it, so a new minus sign does not
    push it sideways out of its column.
    """

    def __init__(self, placeholder: Mobject, value_fn, color_fn=None, size=th.CAPTION_SIZE):
        super().__init__(
            value_fn(), num_decimal_places=cfg.SWEEP_DECIMALS, font_size=size * th.MATH_SCALE
        )
        placeholder.set_opacity(0)
        self.move_to(placeholder)

        def follow(number: DecimalNumber) -> None:
            number.set_value(value_fn())
            number.move_to(placeholder)
            if color_fn is not None:
                number.set_color(color_fn())

        follow(self)
        self.add_updater(follow)


def widest_number() -> str:
    """Placeholder text as wide as the widest live value, e.g. '-0.000'."""
    return "-0." + "0" * cfg.SWEEP_DECIMALS


class S3Sampler(DeckSlide):
    """Section 3: verification, covariance structures, the eigen view, the rho sweep."""

    section_number = 3
    section_name = "Putting the sampler to work"

    def construct(self) -> None:
        self.slide_3_0_section()
        self.slide_3_1_does_it_work()
        self.slide_3_2_turning_the_dial()
        self.slide_3_3_eigen_view()
        self.slide_3_4_until_it_breaks()

    # --- 3.0 -----------------------------------------------------------------

    def slide_3_0_section(self) -> None:
        """Section slide over the last frame of 2.5, which stays on screen into 3.1."""
        self.begin_slide(
            notes="""
            The sampler is built, so let's put it to work. This section is part
            b of the brief: we check that it really produces Sigma, then turn
            the dial on Sigma and watch the data respond.
            """
        )
        # Exactly the frame 2.5 ended on, added without animation, so the
        # switch between the two section videos is invisible.
        self.hero = HeroFrame(self.z_display, self.main_chol)
        self.add(self.hero)
        self.overlay = make_dim_overlay()
        self.play(FadeIn(self.overlay), run_time=cfg.T_SHORT)
        self.show_section_slide("Brief part (b): change Σ, watch the data")
        self.section_card = VGroup(*self.mobjects[-3:])

    # --- 3.1 -----------------------------------------------------------------

    def slide_3_1_does_it_work(self) -> None:
        """Target vs measured Sigma and mu, contours, the 3x3 check. Point: the sampler works."""
        notes = f"""
            We built a sampler, but does it really produce the Sigma we asked
            for? Measured from {cfg.N_STATS:,} samples, the mean and covariance
            match the targets to about two decimal places, and the dots fill the
            1 and 2 sigma contours just as the density predicts. The 3 by 3 G we
            computed by hand in 2.4 passes the same test.
            """
        # Not begin_slide: that would fade out the orange cloud, which carries over from 2.5.
        self.next_slide(notes=notes)
        self.current_notes = notes
        hero = self.hero
        title = th.slide_title("Does it work?")
        bridge = th.caption("We built a sampler. Does it really produce Σ?")

        self.play(FadeOut(self.overlay), FadeOut(self.section_card), run_time=cfg.T_SHORT)
        plot = VGroup(hero.plane, hero.cloud)
        self.play(
            FadeOut(VGroup(hero.grid, hero.marker, hero.label, hero.guides, hero.code)),
            plot.animate.scale(th.VERIFY_PLANE_HEIGHT / hero.plane.height).move_to(
                [*th.VERIFY_PLANE_CENTER, 0]
            ),
            FadeIn(title),
            run_time=cfg.T_MEDIUM,
        )
        self.play(FadeIn(bridge), run_time=cfg.T_SHORT)
        self.reveal()

        samples = mm.transform(self.z_stats, cfg.MAIN_MU, self.main_chol)
        table = self._target_vs_measured(samples).move_to([3.2, 0.2, 0])
        self.play(FadeIn(table[:2]), FadeIn(table[2]), FadeIn(table[4]), run_time=cfg.T_SHORT)
        self.play(FadeIn(table[3], shift=0.2 * LEFT), FadeIn(table[5], shift=0.2 * LEFT))
        self.play(
            *self._flash_matches(table[2], table[3]), *self._flash_matches(table[4], table[5])
        )
        self.reveal()

        contours = self._contours(hero.plane)
        self.play(
            *[Create(ellipse) for ellipse, _ in contours],
            *[FadeIn(label) for _, label in contours],
            run_time=cfg.T_MEDIUM,
        )
        self.reveal()

        check = self._three_by_three_check().move_to([2.75, 0.15, 0])
        self.play(FadeOut(table), run_time=cfg.T_SHORT)
        self.play(FadeIn(check[0]), FadeIn(check[1]), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(check[2]), FadeIn(check[3], shift=0.2 * LEFT), run_time=cfg.T_MEDIUM)
        self.play(*self._flash_matches(check[2], check[3]))
        self.play(
            FadeOut(bridge),
            FadeIn(th.caption("The G we computed by hand generates exactly the Σ we asked for.")),
            run_time=cfg.T_SHORT,
        )

    def _target_vs_measured(self, samples: np.ndarray) -> VGroup:
        """2x3 grid: headings, Sigma | Sigma-hat, mu | mu-hat (measured to 2 places)."""
        size = th.CAPTION_SIZE
        mu_column = [[m] for m in cfg.MAIN_MU]
        mu_hat_column = [[m] for m in mm.sample_mean(samples)]
        return VGroup(
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
        ).arrange_in_grid(rows=3, cols=2, buff=(0.8, 0.5), col_alignments="ll")

    @staticmethod
    def _flash_matches(target: VGroup, measured: VGroup) -> list[Animation]:
        """Flash each target entry and its measured partner green, pair by pair."""
        pairs = zip(target[1].get_entries(), measured[1].get_entries(), strict=True)
        return [
            LaggedStart(
                *[
                    AnimationGroup(
                        Indicate(a, color=th.COLOR_GOOD, scale_factor=1.15),
                        Indicate(b, color=th.COLOR_GOOD, scale_factor=1.15),
                    )
                    for a, b in pairs
                ],
                lag_ratio=0.3,
                run_time=cfg.T_MEDIUM,
            )
        ]

    def _contours(self, plane: NumberPlane) -> VGroup:
        """The k-sigma ellipses of the true density, each labeled at its long end."""
        values, vectors = mm.eigen_axes(cfg.MAIN_SIGMA)
        contours = VGroup()
        for k in cfg.CONTOUR_KS:
            ellipse = make_cov_ellipse(cfg.MAIN_SIGMA, cfg.MAIN_MU, k, plane)
            tip = np.asarray(cfg.MAIN_MU) + k * np.sqrt(values[0]) * vectors[:, 0]
            label = th.math(rf"{fmt(k)}\sigma", th.MIN_SIZE, th.COLOR_SIGMA)
            label.add_background_rectangle(color=th.BACKGROUND, opacity=0.85, buff=0.04)
            label.next_to(plane.c2p(*tip), UR, buff=0.06)
            contours.add(VGroup(ellipse, label))
        return contours

    def _three_by_three_check(self) -> VGroup:
        """The hand-computed 3x3 G from 2.4, and the Sigma-hat it generates."""
        size = th.CAPTION_SIZE
        dim = len(cfg.EXAMPLE_3X3_SIGMA)
        chol = mm.cholesky_factor(cfg.EXAMPLE_3X3_SIGMA)
        z = mm.standard_normal_samples(cfg.N_STATS, dim, cfg.RANDOM_SEED)
        measured = mm.sample_cov(mm.transform(z, np.zeros(dim), chol))
        heading = th.text("The 3×3 example from 2.4:", size, th.COLOR_MUTED)
        g_block = make_labeled_matrix("G", chol, th.COLOR_G, size)
        target = make_labeled_matrix(r"\Sigma", cfg.EXAMPLE_3X3_SIGMA, th.COLOR_SIGMA, size)
        sigma_hat = make_labeled_matrix(r"\hat{\Sigma}", measured, th.COLOR_X, size, fixed=True)
        VGroup(heading, g_block).arrange(DOWN, buff=0.3)
        VGroup(target, sigma_hat).arrange(RIGHT, buff=0.6).next_to(g_block, DOWN, buff=0.45)
        return VGroup(heading, g_block, target, sigma_hat)

    # --- 3.2 -----------------------------------------------------------------

    def slide_3_2_turning_the_dial(self) -> None:
        """Four Sigmas, same z, G applied to each. Point: diagonal spreads, off-diagonal tilts."""
        self.begin_slide(
            notes="""
            Now we trust the machine, so let's feed it different Sigmas. Every
            panel starts from exactly the same z, so the only thing changing is
            G. The identity does nothing, a diagonal G stretches each axis by a
            square root, and an off-diagonal entry shears the grid and tilts the
            cloud one way or the other.
            """
        )
        title = th.slide_title("Turning the dial")
        same_z = th.text("Same z in every panel. Only Σ changes.", th.CAPTION_SIZE, th.COLOR_MUTED)
        same_z.to_edge(RIGHT, buff=th.EDGE_BUFF).match_y(title)
        sigmas = [sigma for _, sigma in cfg.DEMO_STRUCTURES]
        half_width = mm.shared_half_width(sigmas, cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        panels = VGroup(
            *[self._demo_panel(name, sigma, half_width) for name, sigma in cfg.DEMO_STRUCTURES]
        )
        panels.arrange(RIGHT, buff=0.35, aligned_edge=UP)
        panels.next_to(title, DOWN, buff=0.3).set_x(0)
        caption = th.caption(
            "Now we trust the machine. Feed it different Σ and watch what G does to the same cloud."
        )

        self.play(FadeIn(title), FadeIn(same_z), run_time=cfg.T_SHORT)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)
        for index, (panel, (_, sigma)) in enumerate(zip(panels, cfg.DEMO_STRUCTURES, strict=True)):
            self.reveal()
            label, axes, grid, cloud, sigma_block, g_block = panel
            chol = mm.cholesky_factor(sigma)
            moved = make_point_cloud(
                mm.transform(self.z_display, np.zeros(2), chol),
                axes,
                th.COLOR_X,
                panel_dot_radius(th.DEMO_PANEL),
            )
            new_caption = th.caption(self._demo_caption(index, sigma))
            self.play(
                FadeIn(label),
                Create(axes),
                FadeIn(sigma_block),
                FadeIn(grid),
                fade_in_cloud(cloud, cfg.T_SHORT),
                run_time=cfg.T_SHORT,
            )
            self.play(
                FadeIn(g_block, shift=0.2 * DOWN),
                ApplyPointwiseFunction(through_data(axes, lambda data, g=chol: g @ data), grid),
                Transform(cloud, moved),
                FadeOut(caption),
                FadeIn(new_caption),
                run_time=cfg.T_MEDIUM,
            )
            caption = new_caption
        self.reveal()
        takeaway = th.caption(
            "<b>G's diagonal sets the spread. G's off-diagonal sets the tilt.</b>", th.COLOR_G
        )
        self.play(FadeOut(caption), FadeIn(takeaway), run_time=cfg.T_SHORT)

    def _demo_panel(self, name: str, sigma: list[list[float]], half_width: float) -> VGroup:
        """Label, axes, faint z grid, blue z cloud, Sigma and G (G applied when revealed)."""
        size = th.MIN_SIZE
        label = th.text(name.replace("rho", "ρ").replace("= -", "= −"), th.CAPTION_SIZE)
        axes = make_square_axes((0.0, 0.0), half_width, th.DEMO_PANEL)
        grid = make_deformable_grid(axes, cfg.DEMO_GRID_HALF_WIDTH)
        cloud = make_point_cloud(self.z_display, axes, th.COLOR_Z, panel_dot_radius(th.DEMO_PANEL))
        sigma_block = make_labeled_matrix(r"\Sigma", sigma, th.COLOR_SIGMA, size)
        g_block = make_labeled_matrix("G", mm.cholesky_factor(sigma), th.COLOR_G, size)
        # A single row of four panels: label, plot, then Sigma over G underneath.
        matrices = VGroup(sigma_block, g_block).arrange(DOWN, buff=0.2)
        matrices.next_to(axes, DOWN, buff=0.25)
        label.next_to(axes, UP, buff=0.15)
        return VGroup(label, axes, grid, cloud, sigma_block, g_block)

    @staticmethod
    def _demo_caption(index: int, sigma: list[list[float]]) -> str:
        """The panel's caption from config, with its square-rooted variances filled in."""
        # The caption names the operation ("√4"), not its value, to echo the 1D rule.
        roots = [f"√{fmt(sigma[i][i])}" for i in (0, 1)]
        return cfg.DEMO_CAPTIONS[index].format(root_x=roots[0], root_y=roots[1])

    # --- 3.3 -----------------------------------------------------------------

    def slide_3_3_eigen_view(self) -> None:
        """Sigma applied to test arrows, then eigen arrows on the ellipse. Point: read the shape."""
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
        title = th.slide_title("The eigen view")
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
        self.play(fade_in_cloud(cloud), FadeIn(question))
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
            FadeOut(self.eigen_caption),
            FadeIn(caption),
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
                FadeOut(caption),
                FadeIn(new_caption),
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
                FadeOut(caption),
                FadeIn(new_caption),
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
            FadeOut(caption),
            FadeIn(key_caption),
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

    # --- 3.4 -----------------------------------------------------------------

    def slide_3_4_until_it_breaks(self) -> None:
        """Live rho sweep with the Cholesky pipeline beside it. Point: the pivot decides."""
        self.begin_slide(
            notes="""
            Let's run the same machine from 2.4 while turning rho up.
            Elimination leaves a second pivot of 1 minus rho squared, and its
            square root is the only entry in G that gives the cloud any
            thickness. At exactly rho equals 1 the pivot hits zero, x2 is just a
            copy of x1, and NumPy refuses to factor it. Push past 1 and the
            pivot goes negative: there's no square root at all, because no real
            data could have that covariance.
            """
        )
        title = th.slide_title("Turning the dial until it breaks")
        start, *targets = cfg.RHO_SWEEP_PATH
        rho = ValueTracker(start)
        half_width = mm.shared_half_width(
            [mm.correlation_matrix(0.0)], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
        axes = make_square_axes((0.0, 0.0), half_width, th.PANEL_SWEEP).move_to([-3.75, -0.3, 0])
        # Redrawn from the same z every frame, so the cloud morphs instead of re-randomizing.
        cloud = always_redraw(lambda: self._sweep_cloud(rho.get_value(), axes))
        pipeline = self._pipeline(rho).move_to([1.95, -0.1, 0])
        gauge_center = np.array([*th.SWEEP_GAUGE_CENTER, 0])
        gauge = always_redraw(
            lambda: make_pivot_gauge(mm.unit_corr_pivot(rho.get_value()), gauge_center)
        )
        gauge_label = th.math(r"\text{pivot } 1-\rho^2", th.MIN_SIZE, th.COLOR_MUTED)
        gauge_label.next_to(gauge[0], UP, buff=0.15).set_x(gauge_center[0])

        self.play(FadeIn(title), Create(axes), run_time=cfg.T_SHORT)
        self.add(cloud)
        self.play(FadeIn(cloud), FadeIn(pipeline), FadeIn(gauge), FadeIn(gauge_label))
        caption = VGroup()
        for target in targets:
            self.reveal()
            self.play(
                FadeOut(caption),
                rho.animate.set_value(target),
                run_time=cfg.RHO_SWEEP_SECONDS_PER_LEG,
                rate_func=smooth,
            )
            # Not a frozen frame: run the updaters once more so the readouts and
            # cloud show the exact target value, not the last in-between frame.
            self.wait(cfg.T_SETTLE, frozen_frame=False)
            caption = VGroup()
        last = targets[-1]
        caption = th.caption(
            f"Pivot ≈ {fmt(mm.unit_corr_pivot(last), cfg.SWEEP_DECIMALS)}: "
            "a hairline cloud, but Cholesky still works.",
            th.COLOR_GOOD,
        )
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)
        self.reveal()

        self.play(
            FadeOut(caption),
            rho.animate.set_value(cfg.RHO_ZERO_PIVOT),
            run_time=cfg.T_MEDIUM,
        )
        self.wait(cfg.T_SETTLE, frozen_frame=False)
        copy_label = th.math(r"x_2 = x_1", th.CAPTION_SIZE, th.COLOR_WARN)
        copy_label.add_background_rectangle(color=th.BACKGROUND, opacity=0.9, buff=0.05)
        copy_label.next_to(axes.c2p(half_width * 0.45, half_width * 0.6), LEFT, buff=0.1)
        caption = th.caption(
            "Pivot = 0. NumPy's Cholesky refuses: it needs every pivot strictly positive.",
            th.COLOR_WARN,
        )
        self.play(FadeIn(copy_label), FadeIn(caption), run_time=cfg.T_SHORT)

        if cfg.ENABLE_NEGATIVE_PIVOT_REVEAL:
            self.reveal()
            # The cloud fades as rho passes 1 (see _sweep_cloud): no data exists out there.
            self.play(
                FadeOut(caption),
                FadeOut(copy_label),
                rho.animate.set_value(cfg.RHO_NEGATIVE_PIVOT),
                run_time=cfg.T_LONG,
            )
            self.wait(cfg.T_SETTLE, frozen_frame=False)
            g_entry = pipeline.g_entry
            pivot = mm.unit_corr_pivot(cfg.RHO_NEGATIVE_PIVOT)
            root = th.math(rf"\sqrt{{{fmt(pivot)}}}", th.CAPTION_SIZE, th.COLOR_BAD)
            root.move_to(g_entry)
            g_entry.clear_updaters()
            caption = th.caption("No data in the world has this covariance.", th.COLOR_BAD)
            self.play(
                FadeOut(g_entry),
                FadeIn(root),
                pipeline.g_box.animate.set_color(th.COLOR_BAD),
                FadeIn(caption),
                run_time=cfg.T_MEDIUM,
            )
            pipeline.add(root)
        self.reveal()

        # Three lines in a box over the plot, which is empty by now: as one
        # caption line the summary would have to shrink below the minimum size.
        summary = VGroup(
            th.text("Positive pivots: Cholesky works.", th.BODY_SIZE, th.COLOR_GOOD),
            th.text("Zero pivot: the data collapses.", th.BODY_SIZE, th.COLOR_WARN),
            th.text("Negative pivot: the Σ is impossible.", th.BODY_SIZE, th.COLOR_BAD),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        summary_box = boxed(summary).move_to(axes)
        self.play(FadeOut(caption), FadeIn(summary_box, scale=1.05), run_time=cfg.T_MEDIUM)
        # Freeze the live pieces: an always_redraw mobject rebuilds itself every
        # frame and would ignore the fade-out at the start of the next slide.
        for live in (cloud, gauge, *pipeline.live_entries):
            live.clear_updaters()

    def _sweep_cloud(self, rho: float, axes: Axes) -> VGroup:
        """The cloud for [[1, rho], [rho, 1]] from the fixed z.

        Past |rho| = 1 no real G exists; the degenerate rho = 1 line is drawn
        fading out as rho heads to RHO_NEGATIVE_PIVOT, so the cloud leaves
        smoothly instead of vanishing in one frame. np.linalg.cholesky is never
        called here: the factor is the closed form from mvn_math.
        """
        factor = mm.unit_corr_factor(rho)
        fade = 1.0
        if factor is None:
            factor = mm.unit_corr_factor(float(np.sign(rho)))
            fade = max(0.0, 1.0 - (abs(rho) - 1.0) / (cfg.RHO_NEGATIVE_PIVOT - 1.0))
        cloud = make_point_cloud(
            mm.transform(self.z_display, np.zeros(2), factor),
            axes,
            th.COLOR_X,
            panel_dot_radius(th.PANEL_SWEEP),
        )
        return cloud.set_fill(opacity=th.DOT_OPACITY * fade)

    def _pipeline(self, rho: ValueTracker) -> VGroup:
        """The 2.4 route for [[1, rho], [rho, 1]], recomputed every frame.

        Rows: Sigma, the elimination step, D, and G with G22 boxed. Live values
        sit on placeholder entries sized for '-0.000'. The returned group has
        ``live_entries``, ``g_entry`` and ``g_box`` attributes for later steps.
        """
        size = th.CAPTION_SIZE
        wide = widest_number()

        def value() -> float:
            return rho.get_value()

        def pivot() -> float:
            return mm.unit_corr_pivot(rho.get_value())

        def pivot_shade() -> str:
            return pivot_color(pivot())

        sigma_matrix = make_symbolic_matrix([["1", wide], [wide, "1"]], th.COLOR_SIGMA, size)
        sigma_row = VGroup(th.math(r"\Sigma =", size, th.COLOR_SIGMA), sigma_matrix)
        elim_vector = make_symbolic_matrix([["0", wide]], th.COLOR_TEXT, size)
        elim_row = VGroup(th.math(r"R_2 \leftarrow R_2 - \rho R_1:", size), elim_vector)
        slot = th.math(wide, size)
        diag_row = VGroup(th.math(r"D = \mathrm{diag}(1,\,", size), slot, th.math(")", size))
        g_matrix = make_symbolic_matrix([["1", "0"], [wide, wide]], th.COLOR_G, size)
        g_row = VGroup(th.math("G =", size, th.COLOR_G), g_matrix)
        for row in (sigma_row, elim_row, diag_row, g_row):
            row.arrange(RIGHT, buff=0.2)

        sigma_entries = sigma_matrix.get_entries()
        g_entries = g_matrix.get_entries()
        live = [
            LiveEntry(sigma_entries[1], value, size=size).set_color(th.COLOR_SIGMA),
            LiveEntry(sigma_entries[2], value, size=size).set_color(th.COLOR_SIGMA),
            LiveEntry(elim_vector.get_entries()[1], pivot, pivot_shade, size),
            LiveEntry(slot, pivot, pivot_shade, size),
            LiveEntry(g_entries[2], value, size=size).set_color(th.COLOR_G),
            LiveEntry(g_entries[3], lambda: np.sqrt(max(0.0, pivot())), size=size).set_color(
                th.COLOR_G
            ),
        ]
        g_box = SurroundingRectangle(g_entries[3], color=th.COLOR_G, buff=0.08)
        g_note = th.text(
            "the only entry giving\nthe cloud its thickness", th.MIN_SIZE, th.COLOR_MUTED
        )
        rows = VGroup(sigma_row, elim_row, diag_row, VGroup(g_row, g_box))
        rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        g_box.move_to(g_entries[3])
        g_note.next_to(g_row, DOWN, buff=0.15, aligned_edge=LEFT)
        group = VGroup(rows, g_note, *live)
        group.live_entries = live
        group.g_entry = live[-1]
        group.g_box = g_box
        return group


class S3bThreeD(ThreeDSlide):
    """Optional slide 3.5: the 3x3 example from 2.4 in 3D. Rendered only when ENABLE_3D_BONUS."""

    def construct(self) -> None:
        self.slide_3_5_three_d_bonus()

    def slide_3_5_three_d_bonus(self) -> None:
        """A 3D N(0, Sigma) cloud with a slow camera orbit. Point: same recipe in 3D."""
        notes = """
            The same recipe works in any dimension. Here is the 3 by 3 Sigma
            from our hand calculation, factored with Cholesky and applied to 3D
            standard normals. As the camera moves, you can see the cloud is a
            tilted ellipsoid.
            """
        self.next_slide(notes=notes)
        title = th.slide_title("Bonus: the same recipe in 3D")
        self.add_fixed_in_frame_mobjects(title)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-50 * DEGREES)

        dim = len(cfg.EXAMPLE_3X3_SIGMA)
        z = mm.standard_normal_samples(cfg.N_DISPLAY_3D, dim, cfg.RANDOM_SEED)
        samples = mm.transform(z, np.zeros(dim), mm.cholesky_factor(cfg.EXAMPLE_3X3_SIGMA))
        half_width = mm.shared_half_width(
            [cfg.EXAMPLE_3X3_SIGMA], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP
        )
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
