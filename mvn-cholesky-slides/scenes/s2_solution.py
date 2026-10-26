"""Section 2: the solution.

A matrix A turns N(0, I) into N(0, A A^T). So we need a "square root" of
Sigma, and the Cholesky factor L is the standard choice.
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
    fade_in_cloud,
    fmt,
    fmt_vector,
    make_check_mark,
    make_code,
    make_cov_ellipse,
    make_labeled_matrix,
    make_matrix,
    make_plane,
    make_point_cloud,
    make_square_axes,
    make_symbolic_matrix,
)

IDENTITY = np.eye(2)

NUMPY_SNIPPET = """\
L = np.linalg.cholesky(Sigma)
Z = rng.standard_normal((n, 2))
X = mu + Z @ L.T"""


def colored_math(*parts: str, size: float = th.BODY_SIZE) -> MathTex:
    """MathTex built from parts, with the deck's symbol colors applied.

    Each part is its own submobject, so TransformMatchingTex can match them
    and each symbol can be colored on its own.
    """
    formula = MathTex(*parts, font_size=size * th.MATH_SCALE, color=th.COLOR_TEXT)
    for part in formula:
        tex = part.tex_string
        if r"\Sigma" in tex:
            part.set_color(th.COLOR_SIGMA)
        elif r"\mathbf{x}" in tex or tex.startswith("x_"):
            part.set_color(th.COLOR_X)
        elif tex.startswith("z_2"):
            part.set_color(th.COLOR_Z2)
        elif r"\mathbf{z}" in tex or tex.startswith("z_1"):
            part.set_color(th.COLOR_Z)
        elif tex.startswith("L"):
            part.set_color(th.COLOR_L)
    return formula


class S2Solution(DeckSlide):
    """Section 2: covariance of A z, square roots, Cholesky, and the algorithm."""

    section_number = 2
    section_name = "Solution"

    def construct(self) -> None:
        self.slide_2_0_section()
        self.slide_2_1_covariance_of_az()
        self.slide_2_2_square_root_of_sigma()
        self.slide_2_3_cholesky()
        self.slide_2_4_computing_it()
        self.slide_2_5_mixing_recipe()
        self.slide_2_6_algorithm_and_hero()

    # --- 2.0 -----------------------------------------------------------------

    def slide_2_0_section(self) -> None:
        """Section slide: number, name and a one-line preview of the section."""
        self.begin_slide(
            notes="""
            Now the solution. We will find the right idea of a square root for a
            matrix, compute it with Cholesky, and turn it into a four line algorithm.
            """
        )
        self.show_section_slide("A square root for matrices")

    # --- 2.1 -----------------------------------------------------------------

    def slide_2_1_covariance_of_az(self) -> None:
        """Derive Cov(A z) = A A^T. Point: a matrix acts on covariance as A (.) A^T."""
        self.begin_slide(
            notes="""
            Here is the one fact that explains everything. Remember z is the
            vector of independent standard normals the computer gives us, so its
            covariance is the identity. If x equals A times z, its covariance is
            A times I times A transpose, which is just A A transpose, and that
            explains the strange numbers from the square root attempts.
            """
        )
        title = th.slide_title("What a matrix does to covariance")
        # Say what z is up front: the "= A I A^T" step relies on E[z z^T] = I.
        premise = VGroup(
            th.math(
                r"\mathbf{x} = A\mathbf{z}, \quad \boldsymbol{\mu} = \mathbf{0}",
                th.CAPTION_SIZE,
                th.COLOR_MUTED,
            ),
            colored_math(
                r"\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)",
                r"\text{: independent standard normals, so }",
                r"E[\mathbf{z}\mathbf{z}^T] = I",
                size=th.CAPTION_SIZE,
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        premise[1][1].set_color(th.COLOR_MUTED)
        lines = [
            colored_math(
                r"\mathrm{Cov}(\mathbf{x})", "=", "E[", r"\mathbf{x}", r"\mathbf{x}^T", "]"
            ),
            colored_math("=", "E[", "A", r"\mathbf{z}", r"\mathbf{z}^T", "A^T", "]"),
            colored_math("=", "A", "E[", r"\mathbf{z}", r"\mathbf{z}^T", "]", "A^T"),
            colored_math("=", "A", "I", "A^T"),
            colored_math("=", "A", "A^T"),
        ]
        premise.next_to(title, DOWN, buff=0.4).set_x(-6.2, direction=LEFT)
        lines[0].next_to(premise, DOWN, buff=0.3, aligned_edge=LEFT)
        equals_x = lines[0][1].get_x()
        for previous, line in zip(lines, lines[1:]):
            line.next_to(previous, DOWN, buff=0.22)
            line.shift((equals_x - line[0].get_x()) * RIGHT)  # line up the "=" signs
        result_box = SurroundingRectangle(lines[-1], color=th.COLOR_TEXT, buff=0.15)

        self.play(FadeIn(title), FadeIn(premise[0]), run_time=cfg.T_SHORT)
        self.play(FadeIn(premise[1]), run_time=cfg.T_SHORT)
        self.play(Write(lines[0]), run_time=cfg.T_MEDIUM)
        for previous, line in zip(lines, lines[1:]):
            self.reveal()
            self.play(TransformMatchingTex(previous.copy(), line), run_time=cfg.T_MEDIUM)
        self.play(Create(result_box), run_time=cfg.T_SHORT)
        self.reveal()

        caption = th.caption(
            f'<span foreground="{th.COLOR_X}"><b>x</b></span> is still normal: '
            "a linear transform of a Gaussian stays Gaussian."
        )
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)
        self.reveal()

        root_cov = mm.elementwise_sqrt_covariance(cfg.MAIN_SIGMA)
        sigma_sq = mm.covariance_after_matrix(cfg.MAIN_SIGMA)
        callback = VGroup(
            th.text("Back to the failed square roots:", th.CAPTION_SIZE, th.COLOR_MUTED),
            VGroup(
                th.math(r"A A^T ="), make_matrix(root_cov, th.COLOR_TEXT, th.CAPTION_SIZE)
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                th.math(r"\Sigma\,\Sigma^T =", color=th.COLOR_SIGMA),
                make_matrix(sigma_sq, th.COLOR_TEXT, th.CAPTION_SIZE),
            ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        callback_box = boxed(callback).move_to([3.9, -0.45, 0])
        self.play(FadeIn(callback_box, shift=0.2 * LEFT), run_time=cfg.T_MEDIUM)

    # --- 2.2 -----------------------------------------------------------------

    def slide_2_2_square_root_of_sigma(self) -> None:
        """Define a matrix square root; show rotations. Point: many roots exist, pick one."""
        self.begin_slide(
            notes="""
            So the right notion of a square root is any matrix A with A A
            transpose equal to Sigma, just like sigma times sigma is sigma
            squared in 1D. But it is not unique. Rotating the round cloud first
            changes nothing, so A times any rotation Q works too.
            """
        )
        title = th.slide_title('Defining a "square root" of Σ')
        one_d = th.math(r"\sigma \cdot \sigma = \sigma^2")
        n_d = colored_math("A", "A^T", "=", r"\Sigma")
        swap = th.math(r"\longleftrightarrow", color=th.COLOR_MUTED)
        analogy = VGroup(one_d, swap, n_d).arrange(RIGHT, buff=0.6)
        analogy.next_to(title, DOWN, buff=0.5).set_x(-6.2, direction=LEFT)

        definition = VGroup(
            th.text("Any A with", th.BODY_SIZE),
            colored_math("A", "A^T", "=", r"\Sigma"),
            th.text("works.", th.BODY_SIZE),
        ).arrange(RIGHT, buff=0.2)
        rotation = colored_math("(AQ)(AQ)^T", "=", r"A\,QQ^T A^T", "=", "A", "A^T")
        rotation_note = th.text("Q any rotation, so QQᵀ = I", th.CAPTION_SIZE, th.COLOR_MUTED)
        left_column = VGroup(definition, rotation, rotation_note).arrange(
            DOWN, buff=0.45, aligned_edge=LEFT
        )
        left_column.next_to(analogy, DOWN, buff=0.7, aligned_edge=LEFT)

        half_width = mm.shared_half_width([IDENTITY], cfg.AXIS_N_SD, cfg.AXIS_ROUND_STEP)
        axes = make_square_axes((0.0, 0.0), half_width, th.PANEL_ROTATION).move_to([4.2, -0.6, 0])
        cloud = make_point_cloud(self.z_display, axes, th.COLOR_Z)
        marker = Dot(axes.c2p(*cfg.WORKED_Z), radius=th.WORKED_DOT_RADIUS, color=th.COLOR_TEXT)
        unit_circle = DashedVMobject(
            make_cov_ellipse(IDENTITY, (0.0, 0.0), cfg.QUESTION_ELLIPSE_K, axes, th.COLOR_Z)
        )
        rotate_label = th.text("rotate by Q", th.CAPTION_SIZE, th.COLOR_MUTED)
        rotate_label.next_to(axes, UP, buff=0.15)

        self.play(FadeIn(title), run_time=cfg.T_SHORT)
        self.play(Write(one_d), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(swap), Write(n_d), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(definition), run_time=cfg.T_MEDIUM)
        self.reveal()

        self.play(Write(rotation), FadeIn(rotation_note), run_time=cfg.T_MEDIUM)
        self.play(Create(axes), fade_in_cloud(cloud), Create(unit_circle))
        self.play(FadeIn(marker, scale=2), FadeIn(rotate_label), run_time=cfg.T_SHORT)
        self.play(
            Rotate(
                VGroup(cloud, marker),
                angle=np.deg2rad(cfg.ROTATION_DEMO_DEGREES),
                about_point=axes.c2p(0, 0),
            ),
            run_time=cfg.T_LONG,
        )
        self.play(
            FadeIn(th.caption("Many square roots exist. We need a rule to pick one.")),
            run_time=cfg.T_SHORT,
        )

    # --- 2.3 -----------------------------------------------------------------

    def slide_2_3_cholesky(self) -> None:
        """Sigma = L L^T with L lower triangular. Point: Cholesky is the rule we pick."""
        self.begin_slide(
            notes="""
            The rule we pick is the Cholesky decomposition: write Sigma as L L
            transpose, where L is lower triangular with a positive diagonal. It
            is unique, it exists exactly when Sigma is symmetric positive
            definite, and it is fast and stable, which is why every library uses it.
            """
        )
        title = th.slide_title("Cholesky decomposition")
        equation = colored_math(r"\Sigma", "=", "L", "L^T", size=th.TITLE_SIZE)
        entries = [
            ["L_{11}", "0", "0"],
            ["L_{21}", "L_{22}", "0"],
            ["L_{31}", "L_{32}", "L_{33}"],
        ]
        lower = make_symbolic_matrix(entries, th.COLOR_L)
        for entry, tex in zip(lower.get_entries(), sum(entries, [])):
            if tex == "0":
                entry.set_color(th.COLOR_MUTED)  # grey out the zeros above the diagonal
        l_block = VGroup(th.math("L =", color=th.COLOR_L), lower).arrange(RIGHT, buff=0.2)
        shape_note = th.text("lower triangular, positive diagonal", th.CAPTION_SIZE, th.COLOR_MUTED)
        shape_note.next_to(l_block, DOWN, buff=0.25)
        top_row = VGroup(equation, VGroup(l_block, shape_note)).arrange(RIGHT, buff=1.6)
        top_row.next_to(title, DOWN, buff=0.45).set_x(0)

        facts = VGroup(
            th.text("1.  Unique.", th.BODY_SIZE),
            th.text("2.  Exists exactly when Σ is symmetric positive definite.", th.BODY_SIZE),
            th.text("3.  Fast (about n³/3 operations) and numerically stable.", th.BODY_SIZE),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        facts.next_to(top_row, DOWN, buff=0.6).set_x(0)

        self.play(FadeIn(title), Write(equation), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(l_block), FadeIn(shape_note), run_time=cfg.T_MEDIUM)
        for fact in facts:
            self.reveal()
            self.play(FadeIn(fact, shift=0.2 * RIGHT), run_time=cfg.T_SHORT)

    # --- 2.4 -----------------------------------------------------------------

    def slide_2_4_computing_it(self) -> None:
        """2x2 Cholesky by hand, verified, then the general formulas."""
        self.begin_slide(
            notes="""
            Let's compute L for our example by hand. The first entry is the
            square root of 4, the entry below it is 2 divided by 2, and the last
            entry is whatever variance is left after removing that. Multiplying
            back gives Sigma, and the general formulas are the same pattern
            applied one column at a time.
            """
        )
        steps = mm.cholesky_2x2_steps(cfg.MAIN_SIGMA)
        chol = steps.factor
        title = th.slide_title("Computing it")

        sigma_block = make_labeled_matrix(r"\Sigma", cfg.MAIN_SIGMA, th.COLOR_SIGMA)
        unknown = make_symbolic_matrix([["?", "0"], ["?", "?"]], th.COLOR_L)
        unknown.get_entries()[1].set_color(th.COLOR_MUTED)
        l_block = VGroup(th.math("L =", color=th.COLOR_L), unknown).arrange(RIGHT, buff=0.2)
        matrices = VGroup(sigma_block, l_block).arrange(DOWN, buff=0.7, aligned_edge=LEFT)
        matrices.next_to(title, DOWN, buff=0.5).set_x(-6.0, direction=LEFT)

        s11, s21, s22 = (fmt(v) for v in (steps.sigma11, steps.sigma21, steps.sigma22))
        l11, l21, l22 = (fmt(v) for v in (steps.l11, steps.l21, steps.l22))
        step_lines = VGroup(
            th.math(rf"L_{{11}} = \sqrt{{\Sigma_{{11}}}} = \sqrt{{{s11}}} = {l11}"),
            th.math(rf"L_{{21}} = \Sigma_{{21}} / L_{{11}} = {s21} / {l11} = {l21}"),
            th.math(
                rf"L_{{22}} = \sqrt{{\Sigma_{{22}} - L_{{21}}^2}}"
                rf" = \sqrt{{{s22} - {l21}^2}} = {l22}"
            ),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        step_lines.next_to(matrices, RIGHT, buff=1.0).align_to(matrices, UP)

        self.play(FadeIn(title), FadeIn(matrices), run_time=cfg.T_MEDIUM)
        sigma_entries = sigma_block[1].get_entries()
        l_entries = unknown.get_entries()
        # (Sigma entry used, L entry filled) for L11, L21, L22, in row-major entry order.
        for line, sigma_index, l_index, value in zip(
            step_lines, (0, 2, 3), (0, 2, 3), (l11, l21, l22)
        ):
            self.reveal()
            sigma_box = SurroundingRectangle(sigma_entries[sigma_index], color=th.COLOR_SIGMA)
            l_box = SurroundingRectangle(l_entries[l_index], color=th.COLOR_L)
            filled = th.math(value, color=th.COLOR_L).move_to(l_entries[l_index])
            self.play(Create(sigma_box), Create(l_box), Write(line), run_time=cfg.T_MEDIUM)
            self.play(Transform(l_entries[l_index], filled), run_time=cfg.T_SHORT)
            self.play(FadeOut(sigma_box), FadeOut(l_box), run_time=cfg.T_SHORT)
        self.reveal()

        verify = VGroup(
            th.math("L L^T =", color=th.COLOR_L),
            make_matrix(chol, th.COLOR_L),
            make_matrix(chol.T, th.COLOR_L),
            th.math("="),
            make_matrix(mm.covariance_after_matrix(chol), th.COLOR_SIGMA),
            th.math(r"= \Sigma", color=th.COLOR_SIGMA),
        ).arrange(RIGHT, buff=0.2)
        verify.next_to(step_lines, DOWN, buff=0.6, aligned_edge=LEFT)
        check = make_check_mark(0.5).next_to(verify, RIGHT, buff=0.3)
        self.play(FadeIn(verify), run_time=cfg.T_MEDIUM)
        self.play(Create(check), run_time=cfg.T_SHORT)
        self.reveal()

        general = VGroup(
            th.math(r"L_{jj} = \sqrt{\Sigma_{jj} - \sum_{k<j} L_{jk}^2}"),
            th.math(
                r"L_{ij} = \frac{1}{L_{jj}}\Big(\Sigma_{ij} - \sum_{k<j} L_{ik} L_{jk}\Big)"
                r"\quad \text{for } i > j"
            ),
        ).arrange(DOWN, buff=0.6)
        general.move_to(0.2 * DOWN)
        leftovers = VGroup(matrices, step_lines, verify, check)
        self.play(FadeOut(leftovers), run_time=cfg.T_SHORT)
        self.play(Write(general[0]), run_time=cfg.T_MEDIUM)
        self.play(Write(general[1]), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(th.caption("Same pattern, one column at a time")), run_time=cfg.T_SHORT)

    # --- 2.5 -----------------------------------------------------------------

    def slide_2_5_mixing_recipe(self) -> None:
        """L for unit variances as a mixing recipe. Point: shared randomness is correlation."""
        self.begin_slide(
            notes="""
            Here is the intuition for what L does. With unit variances, x1 is
            just z1, and x2 takes a slice of z1 plus some fresh noise z2. The
            slice of z1 is exactly the part x2 shares with x1, and that sharing
            is what we call correlation.
            """
        )
        rho = cfg.MIXING_DEMO_RHO
        shared, fresh = mm.variance_shares(rho)
        title = th.slide_title("Intuition: L is a mixing recipe")

        sigma_rho = VGroup(
            th.math(r"\Sigma(\rho) =", color=th.COLOR_SIGMA),
            make_symbolic_matrix([["1", r"\rho"], [r"\rho", "1"]], th.COLOR_SIGMA),
        ).arrange(RIGHT, buff=0.2)
        l_rho = VGroup(
            th.math("L =", color=th.COLOR_L),
            make_symbolic_matrix([["1", "0"], [r"\rho", r"\sqrt{1-\rho^2}"]], th.COLOR_L),
        ).arrange(RIGHT, buff=0.2)
        arrow = Arrow(LEFT, RIGHT, color=th.COLOR_MUTED).scale(0.6)
        top_row = VGroup(sigma_rho, arrow, l_rho).arrange(RIGHT, buff=0.4)
        top_row.next_to(title, DOWN, buff=0.4).set_x(0)

        x1_line = colored_math("x_1", "=", "z_1")
        x2_line = colored_math("x_2", "=", r"\rho", "z_1", "+", r"\sqrt{1-\rho^2}", "z_2")
        recipe = VGroup(x1_line, x2_line).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        x2_line.shift((x1_line[1].get_x() - x2_line[1].get_x()) * RIGHT)  # align "="
        recipe.next_to(top_row, DOWN, buff=0.45).set_x(0)

        bar = self._variance_bar(shared, fresh, rho).next_to(recipe, DOWN, buff=0.45).set_x(0)

        self.play(FadeIn(title), FadeIn(sigma_rho), run_time=cfg.T_SHORT)
        self.play(GrowArrow(arrow), FadeIn(l_rho), run_time=cfg.T_MEDIUM)
        self.reveal()
        self.play(Write(x1_line), run_time=cfg.T_MEDIUM)
        self.play(Write(x2_line), run_time=cfg.T_MEDIUM)
        self.reveal()
        self.play(FadeIn(bar[0]), run_time=cfg.T_SHORT)
        self.play(GrowFromEdge(bar[1], LEFT), FadeIn(bar[3]), run_time=cfg.T_MEDIUM)
        self.play(GrowFromEdge(bar[2], LEFT), FadeIn(bar[4]), run_time=cfg.T_MEDIUM)
        self.play(
            FadeIn(
                th.caption(
                    "x<sub>2</sub> borrows a fraction of x<sub>1</sub>'s randomness. "
                    "The shared part is the correlation."
                )
            ),
            run_time=cfg.T_SHORT,
        )

    def _variance_bar(self, shared: float, fresh: float, rho: float) -> VGroup:
        """Stacked bar of Var(x2) = shared + fresh, with a label under each part."""
        heading = th.math(
            rf"\mathrm{{Var}}(x_2) = 1 \quad \text{{for }} \rho = {fmt(rho)}", th.CAPTION_SIZE
        )
        shared_part = Rectangle(width=th.BAR_WIDTH * shared, height=th.BAR_HEIGHT)
        shared_part.set_fill(th.COLOR_Z1, 0.85).set_stroke(width=0)
        fresh_part = Rectangle(width=th.BAR_WIDTH * fresh, height=th.BAR_HEIGHT)
        fresh_part.set_fill(th.COLOR_Z2, 0.85).set_stroke(width=0)
        fresh_part.next_to(shared_part, RIGHT, buff=0)
        parts = VGroup(shared_part, fresh_part)
        heading.next_to(parts, UP, buff=0.2)
        shared_label = th.math(
            rf"\rho^2 = {fmt(shared)} \text{{ shared with }} x_1", th.CAPTION_SIZE, th.COLOR_Z1
        ).next_to(parts, DOWN, buff=0.2, aligned_edge=LEFT)
        fresh_label = th.math(
            rf"1-\rho^2 = {fmt(fresh)} \text{{ fresh noise}}", th.CAPTION_SIZE, th.COLOR_Z2
        ).next_to(parts, DOWN, buff=0.2, aligned_edge=RIGHT)
        return VGroup(heading, shared_part, fresh_part, shared_label, fresh_label)

    # --- 2.6 -----------------------------------------------------------------

    def slide_2_6_algorithm_and_hero(self) -> None:
        """Algorithm box, then L applied to the plane and the cloud, then the code."""
        self.begin_slide(
            notes="""
            Putting it together, the algorithm is four lines: choose mu and
            Sigma, factor Sigma, draw independent normals, and compute mu plus L
            z. Watch the grid: L shears the whole plane, the circle becomes an
            ellipse, and then everything shifts by mu. Our tracked point lands
            exactly where the hand calculation says, and in NumPy it is three lines.
            """
        )
        title = th.slide_title("The algorithm")
        algorithm = VGroup(
            colored_math(r"1.\ \text{Choose } \boldsymbol{\mu} \text{ and }", r"\Sigma"),
            colored_math(r"2.\ ", "L", r"= \operatorname{chol}(", r"\Sigma", ")"),
            colored_math(r"3.\ ", r"\mathbf{z}", r"\sim \mathcal{N}(\mathbf{0}, I)"),
            colored_math(r"4.\ ", r"\mathbf{x}", r"= \boldsymbol{\mu} +", "L", r"\mathbf{z}"),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        algorithm_box = boxed(algorithm, buff=0.45).move_to(0.2 * DOWN)

        self.play(FadeIn(title), FadeIn(algorithm_box[0]), run_time=cfg.T_SHORT)
        for line in algorithm:
            self.play(FadeIn(line, shift=0.2 * RIGHT), run_time=cfg.T_SHORT)
        self.reveal()

        self.play(FadeOut(algorithm_box), run_time=cfg.T_SHORT)
        self._hero_animation()
        self.reveal()
        self._code_reveal()

    def _hero_animation(self) -> None:
        """Apply L to a grid and the cloud, shift by mu, and track the worked point.

        Coordinates: everything is drawn on one NumberPlane (`plane`) that never
        moves. The data range is too big for a unit-scale plane, so instead of
        ApplyMatrix (which acts on screen coordinates) we map points through
        data coordinates: screen -> plane.p2c -> apply L or shift -> plane.c2p.
        The grid uses that map pointwise; the dots get target positions computed
        in numpy. Both are linear interpolations, so they stay in sync every frame.
        """
        # Cover all three clouds: z at 0, L z at 0 (wider), and mu + L z at mu.
        x_lim, y_lim = mm.axis_limits(
            [np.zeros(2), np.zeros(2), cfg.MAIN_MU],
            [IDENTITY, cfg.MAIN_SIGMA, cfg.MAIN_SIGMA],
            cfg.AXIS_N_SD,
            cfg.AXIS_ROUND_STEP,
        )
        plane = make_plane(x_lim, y_lim, th.HERO_MAX_WIDTH, th.HERO_MAX_HEIGHT).move_to(
            [-3.2, -0.45, 0]
        )
        grid_extent = cfg.HERO_GRID_HALF_WIDTH
        unit = plane.get_x_unit_size()
        grid = NumberPlane(
            x_range=[-grid_extent, grid_extent, 1],
            y_range=[-grid_extent, grid_extent, 1],
            x_length=2 * grid_extent * unit,
            y_length=2 * grid_extent * unit,
            background_line_style={
                "stroke_color": th.COLOR_TEXT,
                "stroke_opacity": 0.6,
                "stroke_width": 2,
            },
            axis_config={"stroke_color": th.COLOR_TEXT, "stroke_opacity": 0.6},
        )
        grid.shift(plane.c2p(0, 0) - grid.c2p(0, 0))

        def through_data(fn):
            return lambda point: plane.c2p(*fn(np.array(plane.p2c(point))))

        apply_chol = through_data(lambda data: self.main_chol @ data)
        apply_shift = through_data(lambda data: data + np.asarray(cfg.MAIN_MU))

        z_cloud = make_point_cloud(self.z_display, plane, th.COLOR_Z)
        sheared = make_point_cloud(
            mm.transform(self.z_display, np.zeros(2), self.main_chol), plane, th.COLOR_Z
        )
        final = make_point_cloud(
            mm.transform(self.z_display, cfg.MAIN_MU, self.main_chol), plane, th.COLOR_X
        )

        worked_z = np.asarray(cfg.WORKED_Z)
        worked_lz = self.main_chol @ worked_z
        worked_x = mm.transform(worked_z, cfg.MAIN_MU, self.main_chol)
        dot = Dot(plane.c2p(*worked_z), radius=th.WORKED_DOT_RADIUS, color=th.COLOR_TEXT)
        ring = Circle(radius=th.WORKED_RING_RADIUS, color=th.COLOR_TEXT, stroke_width=3).move_to(
            dot
        )
        ring.add_updater(lambda m: m.move_to(dot))
        label = self._point_label(r"\mathbf{z} = " + fmt_vector(worked_z), th.COLOR_Z, dot)
        label.add_updater(lambda m: m.next_to(dot, UR, buff=0.1))

        side = self._hero_side_panel()
        step_one = th.text("Step 1: apply L", th.CAPTION_SIZE, th.COLOR_L)
        step_two = th.text("Step 2: shift by μ", th.CAPTION_SIZE)
        for step in (step_one, step_two):
            step.next_to(side, DOWN, buff=0.6)

        self.play(Create(plane), FadeIn(side), run_time=cfg.T_MEDIUM)
        self.play(Create(grid), fade_in_cloud(z_cloud))
        self.play(FadeIn(dot, scale=2), Create(ring), FadeIn(label), run_time=cfg.T_SHORT)
        self.reveal()

        self.play(FadeIn(step_one), run_time=cfg.T_SHORT)
        self.play(
            ApplyPointwiseFunction(apply_chol, grid),
            Transform(z_cloud, sheared),
            dot.animate.move_to(plane.c2p(*worked_lz)),
            run_time=cfg.T_LONG,
        )
        self.play(
            Transform(
                label,
                self._point_label("L" + r"\mathbf{z} = " + fmt_vector(worked_lz), th.COLOR_L, dot),
            ),
            run_time=cfg.T_SHORT,
        )
        self.reveal()

        self.play(FadeOut(step_one), FadeIn(step_two), run_time=cfg.T_SHORT)
        self.play(
            ApplyPointwiseFunction(apply_shift, grid),
            Transform(z_cloud, final),
            dot.animate.move_to(plane.c2p(*worked_x)),
            run_time=cfg.T_LONG,
        )
        self.play(
            Transform(
                label, self._point_label(r"\mathbf{x} = " + fmt_vector(worked_x), th.COLOR_X, dot)
            ),
            run_time=cfg.T_SHORT,
        )
        # Guard against a coordinate mix-up: the dot must sit on the grid point
        # mu + L z computed in numpy (x = [2, 0.5] for the default config).
        assert np.allclose(plane.p2c(dot.get_center()), worked_x, atol=1e-6)

        ellipse = make_cov_ellipse(cfg.MAIN_SIGMA, cfg.MAIN_MU, cfg.QUESTION_ELLIPSE_K, plane)
        guides = self._coordinate_guides(plane, worked_x)
        self.play(Create(ellipse), Create(guides), run_time=cfg.T_MEDIUM)
        self.hero_side = VGroup(side, step_two)

    def _point_label(self, tex: str, color: str, dot: Dot) -> VGroup:
        """A math label on a dark backing so it stays readable over the dots."""
        label = th.math(tex, th.CAPTION_SIZE, color)
        label.add_background_rectangle(color=th.BACKGROUND, opacity=0.95, buff=0.06)
        return label.next_to(dot, UR, buff=0.1)

    def _hero_side_panel(self) -> VGroup:
        """x = mu + L z with the actual L and mu, to the right of the plane."""
        formula = colored_math(r"\mathbf{x}", "=", r"\boldsymbol{\mu}", "+", "L", r"\mathbf{z}")
        l_block = make_labeled_matrix("L", self.main_chol, th.COLOR_L)
        mu_block = make_labeled_matrix(
            r"\boldsymbol{\mu}", [[m] for m in cfg.MAIN_MU], th.COLOR_TEXT
        )
        side = VGroup(formula, VGroup(l_block, mu_block).arrange(RIGHT, buff=0.5))
        return side.arrange(DOWN, buff=0.5).move_to([3.6, 1.1, 0])

    def _coordinate_guides(self, plane: NumberPlane, point: np.ndarray) -> VGroup:
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

    def _code_reveal(self) -> None:
        """Swap the side panel for the three-line NumPy version."""
        code = make_code(NUMPY_SNIPPET)
        heading = th.text("In NumPy", th.CAPTION_SIZE, th.COLOR_MUTED)
        block = VGroup(heading, code).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        block.to_edge(RIGHT, buff=th.EDGE_BUFF).set_y(0.3)
        self.play(FadeOut(self.hero_side), run_time=cfg.T_SHORT)
        self.play(FadeIn(block, shift=0.2 * UP), run_time=cfg.T_MEDIUM)
