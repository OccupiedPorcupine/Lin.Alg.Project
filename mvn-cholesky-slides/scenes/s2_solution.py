"""Section 2: the solution.

A matrix A turns N(0, I) into N(0, A A^T). So we need a "square root" of
Sigma: a matrix G with G G^T = Sigma. Forcing G to be lower triangular gives
the Cholesky factor, which we compute by hand two ways and then use in the
four-line sampling algorithm.
"""

import sys
from fractions import Fraction
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
    HeroFrame,
    boxed,
    fade_in_cloud,
    fmt,
    fmt_vector,
    make_check_mark,
    make_cross_mark,
    make_dim_overlay,
    make_hero_grid,
    make_labeled_matrix,
    make_point_cloud,
    make_point_label,
    make_symbolic_matrix,
    make_worked_marker,
    sqrt_tex,
    through_data,
    swap_caption,
)

IDENTITY = np.eye(2)
GENERAL_A = [["a", "b"], ["c", "d"]]  # the general 2x2 matrix of slides 2.1 and 2.2
# The one exact symbolic value in the deck: the off-diagonal of A A^T for the
# element-wise root of MAIN_SIGMA. tests/test_mvn_math.py checks it numerically.
SQRT_ATTEMPT_COV_TEX = r"2\sqrt{2} + \sqrt{10}"


def colored_math(*parts: str, size: float = th.BODY_SIZE) -> MathTex:
    """MathTex built from parts, with the deck's symbol colors applied.

    Each part is its own submobject, so TransformMatchingTex can match them
    and each symbol can be colored on its own: Sigma teal, G yellow, x orange,
    z blue. Everything else (A, L, D, numbers) stays white.
    """
    formula = MathTex(*parts, font_size=size * th.MATH_SCALE, color=th.COLOR_TEXT)
    for part in formula:
        tex = part.tex_string.strip()
        if r"\Sigma" in tex:
            part.set_color(th.COLOR_SIGMA)
        elif tex.startswith("G"):
            part.set_color(th.COLOR_G)
        elif r"\mathbf{x}" in tex or tex.startswith("x_"):
            part.set_color(th.COLOR_X)
        elif r"\mathbf{z}" in tex or tex.startswith("z_"):
            part.set_color(th.COLOR_Z)
    return formula


def num_tex(value: float) -> str:
    """LaTeX for a matrix entry: simple fractions as fractions (1/2), else ``fmt``."""
    frac = Fraction(value).limit_denominator(12)
    if frac.denominator != 1 and abs(float(frac) - value) < 1e-9:
        sign = "-" if frac < 0 else ""
        return rf"{sign}\tfrac{{{abs(frac.numerator)}}}{{{frac.denominator}}}"
    return fmt(value)


def paren(value: float) -> str:
    """A number for the middle of a formula: negatives get parentheses."""
    return f"({num_tex(value)})" if value < 0 else num_tex(value)


def tex_matrix(values, color: str = th.COLOR_TEXT, size: float = th.BODY_SIZE) -> Matrix:
    """A numeric matrix with fraction-style entries; zeros above the diagonal stay muted."""
    rows = np.asarray(values, dtype=float)
    matrix = make_symbolic_matrix([[num_tex(v) for v in row] for row in rows], color, size)
    for (i, j), entry in zip(np.ndindex(rows.shape), matrix.get_entries(), strict=True):
        if rows[i, j] == 0 and i != j:
            entry.set_color(th.COLOR_MUTED)
    return matrix


def labeled(label_tex: str, mobject: Mobject, color: str = th.COLOR_TEXT) -> VGroup:
    """'label = mobject', for example 'L =' next to a matrix."""
    label = th.math(label_tex + " =", color=color)
    return VGroup(label, mobject).arrange(RIGHT, buff=0.2)


def entry_at(matrix: Matrix, row: int, col: int) -> Mobject:
    """Entry (row, col) of a square Matrix mobject, 0-based."""
    size = len(matrix.get_rows())
    return matrix.get_entries()[row * size + col]


def replace_entry(matrix: Matrix, row: int, col: int, tex: str, color: str) -> MathTex:
    """A new entry mobject sitting exactly where entry (row, col) is now."""
    old = entry_at(matrix, row, col)
    return MathTex(tex, font_size=old.font_size, color=color).move_to(old)


class S2Solution(DeckSlide):
    """Section 2: covariance of A z, square roots, Cholesky by hand, the algorithm."""

    section_number = 2
    section_name = "Solution"

    def construct(self) -> None:
        self.legend = None  # the z / A / x legend kept in the corner on 2.1
        self.slide_2_0_section()
        self.slide_2_1_key_fact()
        self.slide_2_2_cholesky_plan()
        self.slide_2_3a_lu()
        self.slide_2_3b_ldv()
        self.slide_2_3c_ldlt()
        self.slide_2_3d_split_d()
        self.slide_2_4_algorithm_and_hero()

    def begin_slide(self, notes: str) -> None:
        """Like DeckSlide.begin_slide, but the legend (if shown) stays on screen."""
        self.next_slide(notes=notes)
        self.current_notes = notes
        leftovers = [m for m in self.mobjects if m is not self.legend]
        if leftovers:
            self.play(FadeOut(*leftovers), run_time=cfg.T_SHORT)

    def clear_except_legend(self, *keep: Mobject) -> None:
        """Fade out everything but the legend (and ``keep``), between reveals of one slide."""
        leftovers = [m for m in self.mobjects if m is not self.legend and m not in keep]
        if leftovers:
            self.play(FadeOut(*leftovers), run_time=cfg.T_SHORT)

    # --- 2.0 -----------------------------------------------------------------

    def slide_2_0_section(self) -> None:
        """Section slide: number, name and a one-line preview of the section."""
        self.begin_slide(
            notes="""
            Now the solution. We will work out what a matrix does to the spread
            of the data, use that to define the square root of a matrix, compute
            it by hand, and turn it into a four line algorithm.
            """
        )
        self.show_section_slide("A square root for matrices")

    # --- 2.1 -----------------------------------------------------------------

    def slide_2_1_key_fact(self) -> None:
        """Derive Cov(A z) = A A^T. Point: a matrix acts on the spread as A (.) A^T."""
        self.begin_slide(
            notes="""
            Let's name our characters. z is the random point we start with, A is
            the matrix we multiply by, and x is the point we get out. Each new
            variable is a mix of z₁ and z₂, independent spreads add, and
            covariance only comes from shared ingredients. Put those in a matrix
            and it's exactly A times A transpose, the nD version of 'scaling by a
            multiplies variance by a²'. So the real question is: what A gives
            A Aᵀ = Σ?
            """
        )
        title = th.slide_title("What a matrix does to the spread")
        self.legend = self._legend()
        mu_note = th.caption(
            "We ignore μ here. Shifting moves the cloud but doesn't change its spread, "
            "just like in 1D."
        )
        self.play(FadeIn(title), FadeIn(self.legend), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(mu_note), run_time=cfg.T_SHORT)
        self.reveal()

        question, mixes, rule = self._spread_question(title)
        self._play_question_and_mixing(question, mixes, rule, mu_note)
        self.reveal()

        self.clear_except_legend(title)
        self._play_variances_and_covariance(title)
        self.reveal()

        self.clear_except_legend(title)
        self._play_cov_matrix(title)
        self.reveal()

        # The two worked attempts need the full width, so the legend steps aside.
        self.clear_except_legend(title)
        self.play(FadeOut(self.legend), run_time=cfg.T_SHORT)
        self._play_failed_attempts(title)
        self.reveal()

        self.clear_except_legend(title)
        self.play(FadeIn(self.legend), run_time=cfg.T_SHORT)
        self._play_real_question(title)

    def _legend(self) -> VGroup:
        """Who is who, kept in the top-right corner for the rest of the section."""
        size = th.MIN_SIZE
        rows = VGroup(
            colored_math(r"\mathbf{z} = (z_1, z_2)", r"\text{: the start}", size=size),
            VGroup(
                th.math("A =", size),
                make_symbolic_matrix(GENERAL_A, th.COLOR_TEXT, size),
                th.math(r"\text{: the 2 in 1D}", size),
            ).arrange(RIGHT, buff=0.12),
            colored_math(r"\mathbf{x} = A\mathbf{z}", r"\text{: what we get}", size=size),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        rows[0][0].set_color(th.COLOR_Z)
        rows[2][0].set_color(th.COLOR_X)
        legend = boxed(rows, buff=0.15)
        return legend.to_corner(UR, buff=0.2)

    def _spread_question(self, title: Mobject) -> tuple[VGroup, VGroup, VGroup]:
        """The 1D fact, the nD question, what A z does, and the one rule we need."""
        question = VGroup(
            colored_math(
                r"\text{1D:}\quad", "x", r"= a\,", "z", r"\;\Rightarrow\; \mathrm{Var}(x) = a^2"
            ),
            colored_math(
                r"\text{nD:}\quad",
                r"\mathbf{x}",
                "= A",
                r"\mathbf{z}",
                r"\;\Rightarrow\; \text{spread of }",
                r"\mathbf{x}",
                "= ?",
            ),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        question[0][1].set_color(th.COLOR_X)  # plain x and z in the 1D line
        question[0][3].set_color(th.COLOR_Z)
        check = make_check_mark(0.35).next_to(question[0], RIGHT, buff=0.3)
        question.add(check)
        mixes = VGroup(
            colored_math("x_1", "=", "a", "z_1", "+", "b", "z_2"),
            colored_math("x_2", "=", "c", "z_1", "+", "d", "z_2"),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        rule = colored_math(
            r"\mathrm{Var}(a",
            "z_1",
            "+ b",
            "z_2",
            ") = a^2 + b^2",
            r"\quad\text{(independent, variance 1)}",
        )
        rule[-1].set_color(th.COLOR_MUTED).scale(th.MIN_SIZE / th.BODY_SIZE, about_edge=LEFT)
        question.next_to(title, DOWN, buff=0.45).align_to(title, LEFT)
        mixes.next_to(question, DOWN, buff=0.55).align_to(title, LEFT).shift(0.3 * RIGHT)
        rule.next_to(mixes, DOWN, buff=0.55).align_to(title, LEFT)
        return question, mixes, rule

    def _play_question_and_mixing(
        self, question: VGroup, mixes: VGroup, rule: VGroup, mu_note: Mobject
    ) -> None:
        self.play(FadeOut(mu_note), FadeIn(question[0]), Create(question[2]), run_time=cfg.T_SHORT)
        self.play(FadeIn(question[1]), run_time=cfg.T_MEDIUM)
        self.reveal()
        mix_caption = th.caption("Each new variable is a mix of the two independent ones.")
        self.play(Write(mixes), FadeIn(mix_caption), run_time=cfg.T_MEDIUM)
        self.reveal()
        self.play(
            Write(rule),
            swap_caption(mix_caption, th.caption("Independent spreads add up.")),
            run_time=cfg.T_MEDIUM,
        )

    def _play_variances_and_covariance(self, title: Mobject) -> None:
        """Var(x1), Var(x2), then Cov(x1, x2) through one expansion."""
        variances = VGroup(
            colored_math(r"\mathrm{Var}(", "x_1", ") = a^2 + b^2"),
            colored_math(r"\mathrm{Var}(", "x_2", ") = c^2 + d^2"),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        variances.next_to(title, DOWN, buff=0.45).align_to(title, LEFT)
        expand = colored_math(
            r"\mathrm{Cov}(",
            "x_1",
            ",",
            "x_2",
            r") = E\big[(a",
            "z_1",
            "+ b",
            "z_2",
            ")(c",
            "z_1",
            "+ d",
            "z_2",
            r")\big]",
        )
        terms = MathTex(
            "=",
            r"ac\cdot",
            r"E[z_1^2]",
            "+",
            r"(ad + bc)\cdot",
            r"E[z_1 z_2]",
            "+",
            r"bd\cdot",
            r"E[z_2^2]",
            font_size=th.BODY_SIZE * th.MATH_SCALE,
        )
        expand.next_to(variances, DOWN, buff=0.5).align_to(title, LEFT)
        terms.next_to(expand, DOWN, buff=0.35).align_to(title, LEFT).shift(0.6 * RIGHT)
        self.play(FadeIn(variances, shift=0.2 * DOWN), run_time=cfg.T_MEDIUM)
        self.play(Write(expand), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(terms), run_time=cfg.T_MEDIUM)

        ones = [MathTex("1", font_size=terms.font_size).move_to(terms[i]) for i in (2, 8)]
        zero = MathTex("0", font_size=terms.font_size, color=th.COLOR_BAD).move_to(terms[5])
        independent = th.text("independent", th.MIN_SIZE, th.COLOR_BAD)
        independent.next_to(terms[5], DOWN, buff=0.3)
        self.play(Transform(terms[2], ones[0]), Transform(terms[8], ones[1]), run_time=cfg.T_MEDIUM)
        self.play(Transform(terms[5], zero), FadeIn(independent), run_time=cfg.T_MEDIUM)
        self.play(FadeOut(terms[3:6]), FadeOut(independent), run_time=cfg.T_SHORT)
        result = colored_math(r"\mathrm{Cov}(", "x_1", ",", "x_2", ") = ac + bd")
        result.next_to(variances, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(
            FadeOut(expand),
            FadeOut(terms[:3]),
            FadeOut(terms[6:]),
            FadeIn(result, shift=0.2 * UP),
            FadeIn(th.caption("Only the shared ingredients create covariance.")),
            run_time=cfg.T_MEDIUM,
        )

    def _play_cov_matrix(self, title: Mobject) -> None:
        """Collect the three numbers into Cov(x), then show A A^T is the same matrix."""
        cov_entries = [["a^2 + b^2", "ac + bd"], ["ac + bd", "c^2 + d^2"]]
        cov = VGroup(
            colored_math(r"\mathrm{Cov}(", r"\mathbf{x}", ") ="),
            make_symbolic_matrix(cov_entries, th.COLOR_X),
        ).arrange(RIGHT, buff=0.2)
        cov.next_to(title, DOWN, buff=0.45).align_to(title, LEFT)

        a_mat = make_symbolic_matrix(GENERAL_A, th.COLOR_TEXT)
        at_mat = make_symbolic_matrix([["a", "c"], ["b", "d"]], th.COLOR_TEXT)
        product = make_symbolic_matrix(cov_entries, th.COLOR_X)
        aat = VGroup(th.math("A A^T ="), a_mat, at_mat, th.math("="), product)
        aat.arrange(RIGHT, buff=0.2).next_to(cov, DOWN, buff=0.5).align_to(title, LEFT)

        key = th.math(r"\mathrm{Cov}(A\mathbf{z}) = A A^T", th.TITLE_SIZE)
        key_box = boxed(key, color=th.COLOR_TEXT, buff=0.25)
        key_box.next_to(aat, DOWN, buff=0.45).set_x(0)
        self.play(FadeIn(cov), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(aat[:4]), run_time=cfg.T_MEDIUM)
        # Row i of A times column j of A^T gives entry (i, j); box all four
        # matching pieces in the same color so the match is visible.
        for index, entry in enumerate(product.get_entries()):
            i, j = divmod(index, 2)
            boxes = VGroup(
                SurroundingRectangle(a_mat.get_rows()[i], color=th.COLOR_X, buff=0.08),
                SurroundingRectangle(at_mat.get_columns()[j], color=th.COLOR_X, buff=0.08),
                SurroundingRectangle(cov[1].get_entries()[index], color=th.COLOR_X, buff=0.08),
            )
            self.play(Create(boxes), FadeIn(entry), run_time=cfg.T_SHORT / 2)
            self.play(FadeOut(boxes), run_time=cfg.T_SHORT / 2)
        self.play(FadeIn(product.get_brackets()), run_time=cfg.T_SHORT)
        self.play(FadeIn(key_box, scale=1.05), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(th.caption("The nD version of Var(a z) = a².")), run_time=cfg.T_SHORT)

    def _play_failed_attempts(self, title: Mobject) -> None:
        """Exact A A^T for the two failed guesses from slide 1.7."""
        sigma = np.asarray(cfg.MAIN_SIGMA, dtype=float)
        attempt_one = self._attempt_block(
            "Attempt 1:  A = Σ",
            [[num_tex(v) for v in row] for row in sigma],
            mm.covariance_of_linear_map(sigma),
        )
        root = mm.elementwise_sqrt(sigma)
        root_tex = [[sqrt_tex(v) for v in row] for row in sigma]
        root_cov = mm.covariance_of_linear_map(root)
        attempt_two = self._attempt_block("Attempt 2:  element-wise root", root_tex, root_cov)
        blocks = VGroup(attempt_one, attempt_two).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        blocks.next_to(title, DOWN, buff=0.35).align_to(title, LEFT)

        leak = th.text(
            f"The off-diagonal {chr(0x221A)}2 leaks into the diagonal: "
            f"{fmt(sigma[0, 0])} + {fmt(root[0, 1] ** 2)} = {fmt(root_cov[0, 0])}, "
            f"not {fmt(sigma[0, 0])}.",
            th.MIN_SIZE,
            th.COLOR_BAD,
        )
        leak.next_to(blocks, DOWN, buff=0.25).align_to(title, LEFT)
        for (label, cross), (a_block, lines, result) in blocks:
            self.play(FadeIn(label), FadeIn(a_block), run_time=cfg.T_SHORT)
            self.play(
                LaggedStart(*[Write(line) for line in lines], lag_ratio=0.5), run_time=cfg.T_SHORT
            )
            self.play(FadeIn(result), Create(cross), run_time=cfg.T_SHORT)
        self.play(FadeIn(leak), run_time=cfg.T_SHORT)
        self.play(
            FadeIn(th.caption("The measurements from earlier were right. Now we know why.")),
            run_time=cfg.T_SHORT,
        )

    def _attempt_block(self, label: str, a_tex: list[list[str]], cov: np.ndarray) -> VGroup:
        """One failed guess: a label with a cross, over a row with its A, the
        three entries of A A^T worked out, and the resulting matrix."""
        size = th.CAPTION_SIZE
        a, b, c, d = (entry for row in a_tex for entry in row)

        def squared(tex: str) -> str:
            return f"({tex})^2" if "sqrt" in tex else f"{tex}^2"

        def times(left: str, right: str) -> str:
            return rf"{left}\cdot {right}"

        exact = abs(cov[0, 1] - round(cov[0, 1])) < 1e-9
        off = fmt(cov[0, 1]) if exact else SQRT_ATTEMPT_COV_TEX
        lines = VGroup(
            colored_math(
                r"\mathrm{Var}(",
                "x_1",
                rf") = {squared(a)} + {squared(b)} = {fmt(cov[0, 0])}",
                size=size,
            ),
            colored_math(
                r"\mathrm{Var}(",
                "x_2",
                rf") = {squared(c)} + {squared(d)} = {fmt(cov[1, 1])}",
                size=size,
            ),
            colored_math(r"\mathrm{Cov} = ", rf"{times(a, c)} + {times(b, d)} = {off}", size=size),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        a_block = VGroup(th.math("A =", size), make_symbolic_matrix(a_tex, th.COLOR_TEXT, size))
        a_block.arrange(RIGHT, buff=0.15)
        result_entries = [
            [fmt(cov[0, 0]), off],
            [off, fmt(cov[1, 1])],
        ]
        result = make_symbolic_matrix(result_entries, th.COLOR_TEXT, size)
        row = VGroup(a_block, lines, result).arrange(RIGHT, buff=0.4)
        # The cross sits by the label: the row already fills the frame width.
        header = VGroup(th.text(label, size, th.COLOR_MUTED), make_cross_mark(0.3))
        header.arrange(RIGHT, buff=0.3)
        return VGroup(header, row).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

    def _play_real_question(self, title: Mobject) -> None:
        """We need A A^T = Sigma: three equations in a, b, c, d."""
        sigma = np.asarray(cfg.MAIN_SIGMA, dtype=float)
        need = colored_math(r"\text{We need } A A^T =", r"\Sigma", ":")
        need.next_to(title, DOWN, buff=0.6).align_to(title, LEFT)
        equations = self._three_equations(sigma)
        equations.next_to(need, DOWN, buff=0.5).align_to(need, LEFT).shift(0.8 * RIGHT)
        self.play(FadeIn(need), run_time=cfg.T_SHORT)
        for equation in equations:
            self.play(Write(equation), run_time=cfg.T_SHORT)
        self.play(
            FadeIn(th.caption("Find a, b, c, d. That's what 'square root of a matrix' means.")),
            run_time=cfg.T_SHORT,
        )

    @staticmethod
    def _three_equations(sigma: np.ndarray, size: float = th.BODY_SIZE) -> VGroup:
        """a^2 + b^2 = s11, c^2 + d^2 = s22, ac + bd = s21, with the '=' signs lined up."""
        equations = VGroup(
            colored_math("a^2", "+", "b^2", "=", fmt(sigma[0, 0]), size=size),
            colored_math("c^2", "+", "d^2", "=", fmt(sigma[1, 1]), size=size),
            colored_math("ac", "+", "bd", "=", fmt(sigma[1, 0]), size=size),
        ).arrange(DOWN, buff=0.3)
        for equation in equations[1:]:
            equation.shift((equations[0][3].get_x() - equation[3].get_x()) * RIGHT)
        return equations

    # --- 2.2 -----------------------------------------------------------------

    def slide_2_2_cholesky_plan(self) -> None:
        """The plan: lower-triangular A (called G), then the four steps in symbols."""
        self.legend = None  # the A legend is done; begin_slide fades it with the rest
        self.begin_slide(
            notes="""
            A A transpose equals Σ gives three equations, but there are four
            unknowns, so we get one free choice. Cholesky's choice is to make A
            lower triangular, and we call that matrix G. Here is the plan for
            finding it: eliminate to get L U, pull the pivots out into D, use
            symmetry to get L D L transpose, then split D in half. Next we do each
            step by hand on a 3 by 3. If anyone asks: with a positive diagonal G is
            unique, and it costs about n cubed over 3 operations.
            """
        )
        title = th.slide_title("The plan: Cholesky decomposition")
        intro = VGroup(
            th.text("A Aᵀ = Σ: 3 equations, 4 unknowns. One free choice.", th.CAPTION_SIZE),
            th.text(
                "Cholesky's choice: b = 0. The lower-triangular A is called G.", th.CAPTION_SIZE
            ),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        a_matrix = make_symbolic_matrix(GENERAL_A, th.COLOR_TEXT)
        a_block = labeled("A", a_matrix)
        top = VGroup(intro, a_block).arrange(RIGHT, buff=0.8)
        top.next_to(title, DOWN, buff=0.35).align_to(title, LEFT)

        key = colored_math(r"\Sigma", "=", "G", "G^T", size=th.TITLE_SIZE)
        key_box = boxed(key, color=th.COLOR_G, buff=0.2).next_to(top, DOWN, buff=0.25).set_x(0)

        steps = VGroup(
            colored_math(r"1.\ \ ", r"\Sigma", r"= L\,U"),
            colored_math(r"2.\ \ ", r"\Sigma", r"= L\,D\,V"),
            colored_math(r"3.\ \ ", r"\Sigma", r"= L\,D\,L^T"),
            colored_math(r"4.\ \ ", r"\Sigma", r"= (L\sqrt{D})(L\sqrt{D})^T =", "G", "G^T"),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        steps.next_to(key_box, DOWN, buff=0.25).align_to(title, LEFT).shift(0.4 * RIGHT)
        reasons = VGroup(
            th.text("eliminate: L stores the multipliers", th.CAPTION_SIZE, th.COLOR_MUTED),
            th.text("pull the pivots out of U into D", th.CAPTION_SIZE, th.COLOR_MUTED),
            th.text("symmetry gives V = Lᵀ", th.CAPTION_SIZE, th.COLOR_MUTED),
            th.text("split D in half: G = L√D", th.CAPTION_SIZE, th.COLOR_MUTED),
        )
        for step, reason in zip(steps, reasons, strict=True):
            reason.move_to(step, coor_mask=UP).set_x(th.PLAN_REASON_X, direction=LEFT)
        caption = th.caption("Needs every pivot > 0. Next: each step by hand on a 3×3.")

        self.play(FadeIn(title), FadeIn(intro), FadeIn(a_block), run_time=cfg.T_MEDIUM)
        zero = replace_entry(a_matrix, 0, 1, "0", th.COLOR_MUTED)
        g_label = th.math("G =", color=th.COLOR_G).move_to(a_block[0], aligned_edge=RIGHT)
        self.play(
            Transform(entry_at(a_matrix, 0, 1), zero),
            Transform(a_block[0], g_label),
            a_matrix.get_brackets().animate.set_color(th.COLOR_G),
            run_time=cfg.T_MEDIUM,
        )
        self.play(FadeIn(key_box, scale=1.05), run_time=cfg.T_SHORT)
        for step, reason in zip(steps, reasons, strict=True):
            self.reveal()
            self.play(FadeIn(step, shift=0.2 * RIGHT), FadeIn(reason), run_time=cfg.T_SHORT)
        self.play(FadeIn(caption), run_time=cfg.T_SHORT)

    # --- 2.3a ----------------------------------------------------------------

    def slide_2_3a_lu(self) -> None:
        """Gaussian elimination on the 3x3 Sigma. Point: Sigma = L U, L holds the multipliers."""
        self.begin_slide(
            notes="""
            We already know how to factor a matrix: Gaussian elimination. Each
            step subtracts a multiple of one row from another. We record those
            multipliers in L, and what's left is U.
            """
        )
        route = mm.ldl_steps(cfg.EXAMPLE_3X3_SIGMA)
        title = th.slide_title("By hand, step 1: LU (ordinary elimination)")
        work = tex_matrix(cfg.EXAMPLE_3X3_SIGMA, th.COLOR_SIGMA)
        work_label = th.math(r"\Sigma =", color=th.COLOR_SIGMA)
        work_block = VGroup(work_label, work).arrange(RIGHT, buff=0.2)
        size = len(cfg.EXAMPLE_3X3_SIGMA)
        unknown = [
            ["1" if i == j else ("?" if i > j else "0") for j in range(size)] for i in range(size)
        ]
        lower = make_symbolic_matrix(unknown, th.COLOR_TEXT)
        for entry, (i, j) in zip(lower.get_entries(), np.ndindex(size, size), strict=True):
            if i != j:
                entry.set_color(th.COLOR_MUTED)
        lower_block = labeled("L", lower)
        top = VGroup(work_block, lower_block).arrange(RIGHT, buff=2.0)
        top.next_to(title, DOWN, buff=0.4).set_x(0)

        lines = VGroup(*[self._elimination_line(step) for step in route.steps])
        lines.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        lines.next_to(top, DOWN, buff=0.45).set_x(0)

        self.play(FadeIn(title), FadeIn(work_block), FadeIn(lower_block), run_time=cfg.T_MEDIUM)
        pivot_col = -1
        for step, line in zip(route.steps, lines, strict=True):
            self.reveal()
            if step.source_row != pivot_col:
                pivot_col = step.source_row
                pivot_box = self._swap_pivot_box(work, pivot_col)
            self.play(Write(line), run_time=cfg.T_MEDIUM)
            new_row = [
                replace_entry(work, step.target_row, j, num_tex(v), self._row_color(v, j, step))
                for j, v in enumerate(step.result_row)
            ]
            multiplier = replace_entry(
                lower, step.target_row, step.source_row, num_tex(step.multiplier), th.COLOR_TEXT
            )
            self.play(
                *[
                    Transform(entry_at(work, step.target_row, j), new)
                    for j, new in enumerate(new_row)
                ],
                TransformFromCopy(line[1], multiplier),
                FadeOut(entry_at(lower, step.target_row, step.source_row)),
                run_time=cfg.T_MEDIUM,
            )
        self.reveal()
        self.play(FadeOut(pivot_box), run_time=cfg.T_SHORT)
        u_label = th.math("U =").move_to(work_label, aligned_edge=RIGHT)
        result = th.math(r"\Sigma = L\,U", th.TITLE_SIZE).next_to(lines, DOWN, buff=0.45)
        result[0][0].set_color(th.COLOR_SIGMA)
        self.play(
            Transform(work_label, u_label),
            work.get_brackets().animate.set_color(th.COLOR_TEXT),
            *[entry_at(work, 0, j).animate.set_color(th.COLOR_TEXT) for j in range(size)],
            run_time=cfg.T_SHORT,
        )
        self.play(Write(result), run_time=cfg.T_SHORT)
        self.play(
            FadeIn(th.caption("L records the multipliers. U is what's left after elimination.")),
            run_time=cfg.T_SHORT,
        )

    @staticmethod
    def _row_color(value: float, col: int, step: mm.EliminationStep) -> str:
        """Eliminated entries go grey; the rest of the new row turns white (no longer Sigma)."""
        return th.COLOR_MUTED if value == 0 and col < step.target_row else th.COLOR_TEXT

    def _swap_pivot_box(self, work: Matrix, col: int) -> SurroundingRectangle:
        """Box the pivot of column ``col`` (and drop the previous pivot box)."""
        box = SurroundingRectangle(entry_at(work, col, col), color=th.COLOR_GOOD, buff=0.1)
        previous = getattr(self, "_pivot_box", None)
        animations = [Create(box)] + ([FadeOut(previous)] if previous else [])
        self.play(*animations, run_time=cfg.T_SHORT)
        self._pivot_box = box
        return box

    @staticmethod
    def _elimination_line(step: mm.EliminationStep) -> MathTex:
        """'R2 <- R2 - 1/2 R1  =>  [0, 4, 2]'; part 1 is the multiplier, to fly into L."""
        target, source = step.target_row + 1, step.source_row + 1
        row = r",\ ".join(num_tex(v) for v in step.result_row)
        return MathTex(
            rf"R_{target} \leftarrow R_{target} -",
            num_tex(step.multiplier),
            rf"R_{source} \;\Rightarrow\; [\,{row}\,]",
            font_size=th.BODY_SIZE * th.MATH_SCALE,
        )

    # --- 2.3b ----------------------------------------------------------------

    def slide_2_3b_ldv(self) -> None:
        """Pull the pivots out of U. Point: Sigma = L D V."""
        self.begin_slide(
            notes="""
            U's diagonal holds the pivots. If we pull them out into a diagonal
            matrix D, we get L D V.
            """
        )
        route = mm.ldl_steps(cfg.EXAMPLE_3X3_SIGMA)
        title = th.slide_title("Step 2: LDV (pull the pivots out)")
        upper = tex_matrix(route.upper)
        diag = tex_matrix(route.diag)
        v_mat = tex_matrix(route.v)
        row = VGroup(th.math("U ="), upper, th.math("="), diag, v_mat)
        row.arrange(RIGHT, buff=0.3).next_to(title, DOWN, buff=0.8).set_x(0)
        names = VGroup(
            th.math("D", th.CAPTION_SIZE, th.COLOR_MUTED).next_to(diag, DOWN, buff=0.2),
            th.math("V", th.CAPTION_SIZE, th.COLOR_MUTED).next_to(v_mat, DOWN, buff=0.2),
        )
        size = len(route.pivots)
        boxes = VGroup(
            *[
                SurroundingRectangle(entry_at(upper, k, k), color=th.COLOR_GOOD, buff=0.06)
                for k in range(size)
            ]
        )
        pivots_label = th.text("pivots", th.CAPTION_SIZE, th.COLOR_GOOD)
        pivots_label.next_to(upper, DOWN, buff=0.2)

        self.play(FadeIn(title), FadeIn(row[:2]), run_time=cfg.T_MEDIUM)
        self.play(Create(boxes), FadeIn(pivots_label), run_time=cfg.T_MEDIUM)
        self.reveal()

        divide = th.text("Divide each row of U by its pivot.", th.CAPTION_SIZE, th.COLOR_MUTED)
        divide.next_to(row, DOWN, buff=1.0)
        self.play(FadeIn(row[2]), FadeIn(divide), run_time=cfg.T_SHORT)
        self.play(
            *[TransformFromCopy(entry_at(upper, k, k), entry_at(diag, k, k)) for k in range(size)],
            FadeIn(diag.get_brackets()),
            FadeIn(VGroup(*[entry_at(diag, i, j) for i, j in np.ndindex(size, size) if i != j])),
            run_time=cfg.T_MEDIUM,
        )
        self.play(TransformFromCopy(upper, v_mat), FadeIn(names), run_time=cfg.T_MEDIUM)
        result = th.math(r"\Sigma = L\,D\,V", th.TITLE_SIZE).next_to(divide, DOWN, buff=0.5)
        result[0][0].set_color(th.COLOR_SIGMA)
        self.play(Write(result), run_time=cfg.T_SHORT)
        self.play(
            FadeIn(th.caption("Same factorization, with the pivots separated out.")),
            run_time=cfg.T_SHORT,
        )

    # --- 2.3c ----------------------------------------------------------------

    def slide_2_3c_ldlt(self) -> None:
        """Transpose L and it lands on V. Point: symmetry gives Sigma = L D L^T."""
        self.begin_slide(
            notes="""
            Because Σ is symmetric, V turns out to be exactly L transposed. So
            Σ = L D Lᵀ.
            """
        )
        route = mm.ldl_steps(cfg.EXAMPLE_3X3_SIGMA)
        title = th.slide_title("Step 3: LDLᵀ (symmetry gives V for free)")
        lower = tex_matrix(route.lower)
        v_mat = tex_matrix(route.v)
        blocks = VGroup(labeled("L", lower), labeled("V", v_mat)).arrange(RIGHT, buff=2.2)
        blocks.next_to(title, DOWN, buff=0.45).set_x(0)

        self.play(FadeIn(title), FadeIn(blocks), run_time=cfg.T_MEDIUM)
        self.reveal()

        flipped = lower.copy()
        size = len(route.pivots)
        moves = [
            entry.animate(path_arc=-PI / 3).move_to(entry_at(v_mat, j, i))
            for entry, (i, j) in zip(flipped.get_entries(), np.ndindex(size, size), strict=True)
        ]
        self.play(
            *moves,
            # L and V have the same shape, so the brackets just slide across.
            flipped.get_brackets().animate.shift(v_mat.get_center() - lower.get_center()),
            run_time=cfg.T_LONG,
        )
        match = SurroundingRectangle(v_mat, color=th.COLOR_GOOD, buff=0.15)
        lands = th.math(r"V = L^T", color=th.COLOR_GOOD).next_to(match, DOWN, buff=0.2)
        self.play(Create(match), FadeIn(lands), FadeOut(flipped), run_time=cfg.T_MEDIUM)
        self.reveal()

        reason = VGroup(
            th.math(r"\Sigma = \Sigma^T = (L D V)^T = V^T D\, L^T"),
            th.text("The factorization is unique, so V = Lᵀ.", th.CAPTION_SIZE, th.COLOR_MUTED),
        ).arrange(DOWN, buff=0.25)
        reason[0][0][0].set_color(th.COLOR_SIGMA)
        reason[0][0][2].set_color(th.COLOR_SIGMA)
        reason.next_to(lands, DOWN, buff=0.4).set_x(0)
        result = th.math(r"\Sigma = L\,D\,L^T", th.TITLE_SIZE).next_to(reason, DOWN, buff=0.3)
        result[0][0].set_color(th.COLOR_SIGMA)
        self.play(FadeIn(reason), run_time=cfg.T_MEDIUM)
        self.play(Write(result), run_time=cfg.T_SHORT)
        self.play(
            FadeIn(th.caption("Symmetric matrix, symmetric factorization.")), run_time=cfg.T_SHORT
        )

    # --- 2.3d ----------------------------------------------------------------

    def slide_2_3d_split_d(self) -> None:
        """Split D into two square roots. Point: Cholesky is LDL^T with rooted pivots."""
        self.begin_slide(
            notes="""
            Now the trick: split D into two square roots and give one to each
            side. L √D is our G, and Σ = G Gᵀ. That's Cholesky. It needs every
            pivot to be strictly positive, which is exactly the positive definite
            condition.
            """
        )
        route = mm.ldl_steps(cfg.EXAMPLE_3X3_SIGMA)
        title = th.slide_title("Step 4: split D in half, giving G Gᵀ")
        roots = ",\\ ".join(fmt(np.sqrt(p)) for p in route.pivots)
        size = th.CAPTION_SIZE
        split = th.math(
            rf"D = \sqrt{{D}}\,\sqrt{{D}}, \qquad \sqrt{{D}} = \operatorname{{diag}}({roots})",
            size,
        )
        regroup = colored_math(
            r"\Sigma",
            r"= L\sqrt{D}\,\sqrt{D}\,L^T = (L\sqrt{D})(L\sqrt{D})^T =",
            "G",
            "G^T",
            size=size,
        )
        compute = VGroup(
            colored_math("G", r"= L\sqrt{D} =", size=size),
            tex_matrix(route.lower, size=size),
            tex_matrix(np.sqrt(route.diag), size=size),
            th.math("=", size),
            tex_matrix(route.chol, th.COLOR_G, size),
        ).arrange(RIGHT, buff=0.2)
        verify = VGroup(
            colored_math("G", "G^T", "=", size=size),
            tex_matrix(mm.covariance_of_linear_map(route.chol), th.COLOR_SIGMA, size),
            colored_math("=", r"\Sigma", size=size),
            make_check_mark(0.4),
        ).arrange(RIGHT, buff=0.2)
        lines = VGroup(split, regroup, compute, verify).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        lines.next_to(title, DOWN, buff=0.35).align_to(title, LEFT)

        self.play(FadeIn(title), Write(split), run_time=cfg.T_MEDIUM)
        self.reveal()
        self.play(Write(regroup), run_time=cfg.T_MEDIUM)
        self.reveal()
        self.play(FadeIn(compute[:3]), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(compute[3:]), run_time=cfg.T_MEDIUM)
        scale_note = th.caption("Scale each column of L by the square root of its pivot.")
        self.play(FadeIn(scale_note), run_time=cfg.T_SHORT)
        self.reveal()
        self.play(FadeIn(verify[:3]), run_time=cfg.T_MEDIUM)
        self.play(Create(verify[3]), run_time=cfg.T_SHORT)
        self.play(
            swap_caption(
                scale_note, th.caption("Cholesky is just LDLᵀ with the pivots square-rooted.")
            ),
            run_time=cfg.T_SHORT,
        )
        self.reveal()
        self._play_pivot_payoff()

    def _play_pivot_payoff(self) -> None:
        """Dim the slide; the positive-pivot rule front and center, plus the plant for 3.3."""
        payoff = VGroup(
            th.text("Cholesky needs every pivot to be strictly positive.", th.BODY_SIZE),
            th.text("That's exactly what 'positive definite' means.", th.BODY_SIZE, th.COLOR_GOOD),
        ).arrange(DOWN, buff=0.2)
        payoff_box = boxed(payoff, color=th.COLOR_GOOD, buff=0.3).move_to(0.6 * UP)
        plant = VGroup(
            VGroup(
                th.text("For", th.CAPTION_SIZE),
                make_symbolic_matrix(
                    [["1", r"\rho"], [r"\rho", "1"]], th.COLOR_SIGMA, size=th.CAPTION_SIZE
                ),
                th.math(r"\text{, the pivots are } 1 \text{ and } 1 - \rho^2.", th.CAPTION_SIZE),
            ).arrange(RIGHT, buff=0.15),
            th.text(
                "Watch that second one: it comes back when we push ρ to 1.",
                th.CAPTION_SIZE,
                th.COLOR_WARN,
            ),
        ).arrange(DOWN, buff=0.25)
        plant.next_to(payoff_box, DOWN, buff=0.6)
        self.play(FadeIn(make_dim_overlay()), run_time=cfg.T_SHORT)
        self.play(FadeIn(payoff_box, scale=1.05), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(plant, shift=0.2 * UP), run_time=cfg.T_MEDIUM)

    # --- 2.4 -----------------------------------------------------------------

    def slide_2_4_algorithm_and_hero(self) -> None:
        """Algorithm box, then G applied to the plane and the cloud, then the code.

        The slide ends on exactly HeroFrame (components.py); Section 3 opens on it.
        """
        self.begin_slide(
            notes="""
            Back to our 2 by 2. Its Σ is the top-left corner of the 3 by 3, so its
            G is the top-left corner of the G we just found: 2, 0, 1, 2. The whole algorithm is
            four lines: choose μ and Σ, factor Σ, draw independent normals, and
            compute μ plus G z. Watch the grid: G shears the whole plane, the
            circle becomes an ellipse, and then everything shifts by μ. Our
            tracked point lands exactly at 2, 0.5, and in NumPy it is three lines.
            """
        )
        title = th.slide_title("The algorithm")
        bridge = self._corner_bridge()
        algorithm = VGroup(
            colored_math(r"1.\ \text{Choose } \boldsymbol{\mu} \text{ and }", r"\Sigma"),
            colored_math(r"2.\ ", "G", r"= \operatorname{chol}(", r"\Sigma", ")"),
            colored_math(r"3.\ ", r"\mathbf{z}", r"\sim \mathcal{N}(\mathbf{0}, I)"),
            colored_math(r"4.\ ", r"\mathbf{x}", r"= \boldsymbol{\mu} +", "G", r"\mathbf{z}"),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        algorithm_box = boxed(algorithm, buff=0.45)
        VGroup(bridge, algorithm_box).arrange(RIGHT, buff=0.9).next_to(title, DOWN, buff=0.7).set_x(
            0
        )
        corner_box = SurroundingRectangle(
            VGroup(*bridge.corner_entries), color=th.COLOR_G, buff=0.1
        )

        big, arrow, small = bridge[0]
        self.play(FadeIn(title), FadeIn(big), run_time=cfg.T_MEDIUM)
        self.play(Create(corner_box), FadeIn(bridge[1]), run_time=cfg.T_SHORT)
        self.play(GrowArrow(arrow), FadeIn(small, shift=0.2 * RIGHT), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(algorithm_box[0]), run_time=cfg.T_SHORT)
        for line in algorithm:
            self.play(FadeIn(line, shift=0.2 * RIGHT), run_time=cfg.T_SHORT)
        self.reveal()

        self.play(
            FadeOut(title),
            FadeOut(bridge),
            FadeOut(corner_box),
            FadeOut(algorithm_box),
            run_time=cfg.T_SHORT,
        )
        hero = HeroFrame(self.z_display, self.main_chol)
        self._hero_animation(hero)
        self.reveal()
        self.play(FadeOut(self.hero_side), run_time=cfg.T_SHORT)
        self.play(FadeIn(hero.code, shift=0.2 * UP), run_time=cfg.T_MEDIUM)
        # Swap in the reference frame (same objects, same draw order) so the
        # last frame here and the first frame of Section 3 are identical.
        self.remove(*self.mobjects)
        self.add(hero)
        self.wait(cfg.T_SETTLE)

    def _corner_bridge(self) -> VGroup:
        """The 3x3 G with its top-left 2x2 boxed, an arrow, and the 2x2 G.

        MAIN_SIGMA is the top-left block of EXAMPLE_3X3_SIGMA, and Cholesky of
        a top-left block is the top-left block of the Cholesky factor, so the
        2x2 G comes straight out of the 3x3 work. The caller boxes ``corner_entries``
        after layout, so the box sits on their final positions.
        """
        chol_3x3 = mm.ldl_steps(cfg.EXAMPLE_3X3_SIGMA).chol
        assert np.allclose(chol_3x3[:2, :2], self.main_chol), (
            "MAIN_SIGMA must be the top-left 2x2 block of EXAMPLE_3X3_SIGMA for this slide"
        )
        size = th.CAPTION_SIZE
        big = VGroup(
            th.math(r"G_{3\times 3} =", size, th.COLOR_G), tex_matrix(chol_3x3, th.COLOR_G, size)
        ).arrange(RIGHT, buff=0.15)
        arrow = Arrow(LEFT, RIGHT, color=th.COLOR_MUTED).scale(0.45)
        small = make_labeled_matrix("G", self.main_chol, th.COLOR_G)
        row = VGroup(big, arrow, small).arrange(RIGHT, buff=0.25)
        note = th.text("our 2×2 is the top-left corner", th.CAPTION_SIZE, th.COLOR_MUTED)
        bridge = VGroup(row, note).arrange(DOWN, buff=0.35)
        bridge.corner_entries = [entry_at(big[1], i, j) for i in range(2) for j in range(2)]
        return bridge

    def _hero_animation(self, hero: HeroFrame) -> None:
        """Apply G to a grid and the cloud, shift by mu, and track the worked point.

        Coordinates: everything is drawn on one NumberPlane that never moves.
        It is not unit-scale, so instead of ApplyMatrix (which acts on screen
        coordinates) points go through data coordinates (components.through_data).
        The grid uses that map pointwise; the dots get target positions computed
        in numpy. Both are linear interpolations, so they stay in sync every frame.
        """
        plane = hero.plane
        mu = np.asarray(cfg.MAIN_MU, dtype=float)
        grid = make_hero_grid(plane)
        apply_chol = through_data(plane, lambda data: self.main_chol @ data)
        apply_shift = through_data(plane, lambda data: data + mu)
        cloud = make_point_cloud(self.z_display, plane, th.COLOR_Z)
        sheared = make_point_cloud(
            mm.transform(self.z_display, np.zeros(2), self.main_chol), plane, th.COLOR_Z
        )

        worked_z = np.asarray(cfg.WORKED_Z)
        worked_gz = self.main_chol @ worked_z
        marker = make_worked_marker(plane.c2p(*worked_z))
        label = make_point_label(r"\mathbf{z} = " + fmt_vector(worked_z), th.COLOR_Z, marker[0])
        label.add_updater(lambda m: m.next_to(marker[0], UR, buff=0.1))

        side = self._hero_side_panel()
        step_one = th.text("Step 1: apply G", th.CAPTION_SIZE, th.COLOR_G)
        step_two = th.text("Step 2: shift by μ", th.CAPTION_SIZE)
        for step in (step_one, step_two):
            step.next_to(side, DOWN, buff=0.6)

        self.play(Create(plane), FadeIn(side), run_time=cfg.T_MEDIUM)
        self.play(Create(grid), fade_in_cloud(cloud), run_time=cfg.T_MEDIUM)
        self.play(FadeIn(marker, scale=2), FadeIn(label), run_time=cfg.T_SHORT)
        self.reveal()

        self.play(FadeIn(step_one), run_time=cfg.T_SHORT)
        self.play(
            ApplyPointwiseFunction(apply_chol, grid),
            Transform(cloud, sheared),
            marker.animate.move_to(plane.c2p(*worked_gz)),
            run_time=cfg.T_LONG,
        )
        label.clear_updaters()
        gz_label = make_point_label(
            "G" + r"\mathbf{z} = " + fmt_vector(worked_gz), th.COLOR_G, marker[0]
        )
        self.play(Transform(label, gz_label), run_time=cfg.T_SHORT)
        label.add_updater(lambda m: m.next_to(marker[0], UR, buff=0.1))
        self.reveal()

        self.play(FadeOut(step_one), FadeIn(step_two), run_time=cfg.T_SHORT)
        self.play(
            ApplyPointwiseFunction(apply_shift, grid),
            Transform(cloud, hero.cloud),  # blue fades to orange on the way
            marker.animate.move_to(hero.marker.get_center()),
            run_time=cfg.T_LONG,
        )
        label.clear_updaters()
        # Guard against a coordinate mix-up: the point must sit on the grid point
        # mu + G z computed in numpy (x = [2, 0.5] for the default config).
        assert np.allclose(plane.p2c(marker[0].get_center()), hero.worked_x, atol=1e-6)
        self.play(ReplacementTransform(label, hero.label), run_time=cfg.T_SHORT)
        self.play(Create(hero.guides), run_time=cfg.T_MEDIUM)
        self.hero_side = VGroup(side, step_two)

    def _hero_side_panel(self) -> VGroup:
        """x = mu + G z with the actual G and mu, to the right of the plane."""
        formula = colored_math(r"\mathbf{x}", "=", r"\boldsymbol{\mu}", "+", "G", r"\mathbf{z}")
        g_block = make_labeled_matrix("G", self.main_chol, th.COLOR_G)
        mu_block = make_labeled_matrix(
            r"\boldsymbol{\mu}", [[m] for m in cfg.MAIN_MU], th.COLOR_TEXT
        )
        side = VGroup(formula, VGroup(g_block, mu_block).arrange(RIGHT, buff=0.5))
        return side.arrange(DOWN, buff=0.5).move_to([*th.HERO_SIDE_CENTER, 0])
