# Slide flow

The deck as it is now, slide by slide. Each **click** is one press of the
right arrow in the presenter. Numbers on screen come from `config.py` through
`mvn_math.py`. 108 clicks in total, about 13 minutes of animation; the talk
itself runs longer with speaking time.

## How to use this file

Edit this file directly to describe the flow you want, then ask Claude to
implement it. Any of these work:

- **Reorder:** move a slide's whole `###` block up or down.
- **Remove:** delete the block, or add `> REMOVE` under its heading.
- **Change a slide:** edit its clicks or point, or add a line starting
  `> CHANGE:` describing what you want.
- **Add a slide:** add a new `###` block with `> NEW` under the heading and
  describe its point and clicks.
- **Move between sections:** move the block under another `##` section.

Rough wording is fine. The method names in backticks tell Claude which code
each slide is.

**Notation:** G is always the Cholesky factor (yellow). L only appears on 2.4
as the unit lower-triangular factor from elimination. A is a general matrix
on 2.1 to 2.3. Colors: z blue, x orange, Σ teal, G yellow, distance pink,
normalizer grey, failure red, zero pivot amber, confirmation green.

**If the talk runs long, cut in this order:** A.1 (already after the closing
slide), 1.4, 2.4e, 3.3 (then drop "(eigen view)" on 4.3), 3.5 (off by default).

---

## Section 1: Problem (`scenes/s1_problem.py`, 31 clicks)

### 1.0 Title slide
`slide_1_0_title`
- **Clicks:**
  1. "Simulating the Multivariate Normal / with a Given Covariance Matrix", subtitle "Cholesky decomposition and covariance structures", presenter name.

### 1.0 Section slide: Problem
`slide_1_0_section`
- **Clicks:**
  1. Big "1", "Problem", subtitle "What we have, what we want, and why it is not obvious".

### 1.1 Hook: where correlated randomness matters
`slide_1_1_hook`
- **Point:** all three uses need random numbers that move together. No math yet.
- **Clicks:** the title changes to each use case.
  1. **Monte Carlo portfolio risk.** "Stock A vs Stock B returns (simulated)": blue round cloud (ρ = 0) beside orange tilted cloud (ρ = 0.8).
  2. The clouds morph into overlaid histograms of the 50/50 portfolio return, with legend "ρ = 0: sd 0.71, 5% VaR −1.16" and "ρ = 0.8: sd 0.95, 5% VaR −1.56", and red dashed "5% worst case" lines. Caption: "Ignore correlation and you underestimate risk."
  3. **Gaussian processes.** 60 blue jagged dots morph into a smooth orange curve, three times (earlier curves dim). Caption: "A random smooth curve is one sample from a 60-dimensional normal."
  4. **Synthetic data generation.** "Original data (illustrative)" heatmap, then "20,000 simulated rows", then the "Synthetic copy" heatmap with measured correlations; 4 synthetic exam-score rows tick in. Caption: "Fake data with real structure: safe to share, useful for testing models."
  5. Dim overlay: "All three need random numbers / that move together."

### 1.2 What computers give us: N(0, I)
`slide_1_2_what_computers_give`
- **Point:** computers give us independent noise; each row is one point.
- **Clicks:**
  1. NumPy code with the printed `Z[:3]` rows (generated from the real samples). Each row is highlighted and flies to its blue dot, then the rest of the cloud fills in.
  2. "each value ~ N(0, 1): φ(z) = e^(−z²/2)/√(2π)", the readout "correlation(z₁, z₂) ≈ 0.00" (from 20,000 samples), then the two-line caption "Each draw is independent of the others. / The two variables are not correlated."

### 1.3 What we want: N(μ, Σ)
`slide_1_3_what_we_want`
- **Point:** μ says where the cloud sits; Σ says how wide it is and which way it tilts.
- **Clicks:**
  1. The blue cloud shrinks into the top-left corner (labeled N(0, I)); the orange N(μ, Σ) cloud appears.
  2. μ = [1; 2] with a dot at the cloud's center. Σ = [[4, 2], [2, 5]]: diagonal boxed ("variances (spread)"), off-diagonal dashed ("covariance (moving together)"). Caption: "μ says where the cloud sits. Σ says how wide it is and which way it tilts."

### 1.4 Reading the formula: same idea as 1D
`slide_1_4_reading_the_formula`
- **Point:** the nD density is the 1D one with numbers turned into vectors and matrices; the only new idea is the distance term.
- **Clicks:** terms colored by role (center white, spread teal, distance pink, normalizer grey).
  1. 1D formula with a small bell curve.
  2. It rearranges (TransformMatchingTex) into the matrix-like form. Caption: "Same formula, just rearranged."
  3. The nD formula appears under it, term by term in shared columns.
  4. Translation table: header and Center row.
  5. Spread row.
  6. Distance row.
  7. Normalizer row.
  8. Only the nD formula stays, distance term boxed in pink. Zoomed orange cloud, dashed ruler circle of radius 2, points A (long axis) and B (short axis): ruler 2 and 2, distance² 0.61 and 1.64. Caption: "Same ruler distance, different 'Σ distance'. Along the long axis, points are more typical."
  9. Pink contour ellipses through A and B. Caption: "Every point on one ellipse has the same distance, so the same density."
  10. Caption: "Only two things change: numbers become vectors and matrices, and distance now accounts for tilt."

### 1.5 The question
`slide_1_5_the_question`
- **Clicks:**
  1. N(0, I) cloud, a "?" over an arrow, N(μ, Σ) cloud with its 2σ ellipse. Caption: "How do we turn N(0, I) into N(μ, Σ)?"

### 1.6 Warm-up: from N(0, 1) to N(3, 4)
`slide_1_6_one_d_warmup`
- **Point:** to get variance σ², scale by √(σ²).
- **Clicks:** the mean line, the σ arrow and live readouts "a = …" and "variance = a² = …" stay on screen throughout.
  1. Blue N(0, 1) curve, dashed target N(3, 4), "Goal: N(3, 4)" with the 4 boxed in teal ("variance"), and the slot x = ? · z + ?.
  2. x = 4z (red): the curve overshoots, the readout climbs to 16 in red, ✗.
  3. Var(az) = a² Var(z). Caption: "Stretching by a multiplies the variance by a²."
  4. x = 2z: a animates down to 2 = √4, the readout shows 4 in green, ✓.
  5. x = 3 + 2z: the curve slides onto the dashed target and turns orange.
  6. The general rule x = μ + √(σ²) z ~ N(μ, σ²).
  7. Dim overlay: "To get variance σ², scale by √(σ²)."

### 1.7 The nD catch: the variance becomes a matrix
`slide_1_7_nd_catch`
- **Point:** element-wise thinking fails; we need a real definition of a matrix square root.
- **Clicks:**
  1. Bridge: "1D variance = 4 (one number) → scale by √4 = 2 ✓" and "nD covariance Σ (a matrix) → scale by √Σ = ???" (pulsing). Caption: "In nD, the variance is a whole matrix. What is its square root?"
  2. Attempt 1, multiply by Σ: cloud vs dashed target ellipse, "Measured ≈ [[20.0, 18.3], [18.3, 29.7]] ≈ Σ²", ✗. Caption: "Just like 1D: multiplying by the variance squares it."
  3. Attempt 2, root of each entry A = [[2, √2], [√2, √5]]: "Measured ≈ [[6.0, 6.1], [6.1, 7.1]] ≠ Σ", ✗. Caption: "Square-rooting each entry doesn't work. Matrices need a better definition of square root."

---

## Section 2: Solution (`scenes/s2_solution.py`, 39 clicks)

### 2.0 Section slide: Solution
`slide_2_0_section`
- **Clicks:**
  1. Big "2", "Solution", subtitle "A square root for matrices".

### 2.1 Key fact: what a matrix does to the spread
`slide_2_1_key_fact`
- **Point:** Cov(A z) = A Aᵀ, the nD version of Var(a z) = a².
- **Clicks:** a legend box (z, A, x) sits in the top-right corner through 2.3.
  1. Legend: z = (z₁, z₂) the start; A = [[a, b], [c, d]] the "2" in 1D; x = A z what we get. Caption: "We ignore μ here. Shifting moves the cloud but doesn't change its spread, just like in 1D."
  2. "1D: x = a z ⇒ Var(x) = a² ✓" and "nD: x = A z ⇒ spread of x = ?"
  3. x₁ = a z₁ + b z₂, x₂ = c z₁ + d z₂. Caption: "Each new variable is a mix of the two independent ones."
  4. Var(a z₁ + b z₂) = a² + b². Caption: "Independent spreads add up."
  5. Var(x₁), Var(x₂), then the expansion of Cov(x₁, x₂): E[z₁²] and E[z₂²] become 1, E[z₁z₂] becomes a red 0 ("independent"), the middle term fades, leaving ac + bd. Caption: "Only the shared ingredients create covariance."
  6. Cov(x) as a matrix, then A Aᵀ multiplied out entry by entry with matching boxes. Boxed: "Cov(A z) = A Aᵀ". Caption: "The nD version of Var(a z) = a²."
  7. The legend steps aside. Attempt 1 (A = Σ): 20, 29, 18 → [[20, 18], [18, 29]] ✗. Attempt 2 (element-wise root): 6, 7, 2√2 + √10 ✗, with "The off-diagonal √2 leaks into the diagonal: 4 + 2 = 6, not 4." Caption: "The measurements from earlier were right. Now we know why."
  8. The legend returns. "We need A Aᵀ = Σ:" a² + b² = 4, c² + d² = 5, ac + bd = 2. Caption: "Find a, b, c, d. That's what 'square root of a matrix' means."

### 2.2 Many square roots
`slide_2_2_many_square_roots`
- **Clicks:**
  1. σ · σ = σ² ⟷ A Aᵀ = Σ.
  2. (AQ)(AQ)ᵀ = A QQᵀ Aᵀ = A Aᵀ ("Q any rotation"). The blue cloud rotates 90° and stays round. Caption: "Many square roots exist. We need a rule to pick one."

### 2.3 The rule: force a triangle (Cholesky)
`slide_2_3_force_a_triangle`
- **Clicks:**
  1. b is greyed to 0; solve a² = 4 → a = 2, then c = 1, then d = 2, each filling A in yellow; A becomes G = [[2, 0], [1, 2]]. Boxed: "Cholesky: Σ = G Gᵀ, G lower triangular". Caption: "Force the top-right to zero: the equations solve one at a time."
  2. "1. Unique (with a positive diagonal)."
  3. "2. Exists exactly when Σ is symmetric positive definite."
  4. "3. Fast (about n³/3 operations) and numerically stable."
  5. "Why not just A·A = Σ?": G G = [[4, 0], [4, 4]] ≠ Σ ✗, G Gᵀ = Σ ✓. Caption: "The data follows A Aᵀ, so that's the test."

### 2.4a By hand, step 1: LU (ordinary elimination)
`slide_2_4a_lu`
- **Clicks:** Σ = [[4, 2, 2], [2, 5, 3], [2, 3, 3]].
  1. Σ and L = [[1, 0, 0], [?, 1, 0], [?, ?, 1]].
  2. Pivot 4 boxed; R2 ← R2 − ½R1 → [0, 4, 2]; the ½ flies into L.
  3. R3 ← R3 − ½R1 → [0, 2, 2].
  4. Pivot 4 boxed; R3 ← R3 − ½R2 → [0, 0, 1].
  5. The matrix becomes U; "Σ = L U". Caption: "L records the multipliers. U is what's left after elimination."

### 2.4b Step 2: LDV (pull the pivots out)
`slide_2_4b_ldv`
- **Clicks:**
  1. U with its pivots 4, 4, 1 boxed ("pivots").
  2. "Divide each row of U by its pivot": U = D V, then "Σ = L D V". Caption: "Same factorization, with the pivots separated out."

### 2.4c Step 3: LDLᵀ (symmetry gives V for free)
`slide_2_4c_ldlt`
- **Clicks:**
  1. L and V side by side.
  2. A copy of L flips (transpose) and lands on V; green box, "V = Lᵀ".
  3. "Σ = Σᵀ = (LDV)ᵀ = Vᵀ D Lᵀ", "The factorization is unique, so V = Lᵀ.", then "Σ = L D Lᵀ". Caption: "Symmetric matrix, symmetric factorization."

### 2.4d Step 4: split D in half, giving G Gᵀ
`slide_2_4d_split_d`
- **Clicks:**
  1. D = √D √D with √D = diag(2, 2, 1).
  2. Σ = L√D √D Lᵀ = (L√D)(L√D)ᵀ = G Gᵀ.
  3. G = L√D = [[2, 0, 0], [1, 2, 0], [1, 1, 1]]. Caption: "Scale each column of L by the square root of its pivot."
  4. G Gᵀ = Σ ✓. Caption: "Cholesky is just LDLᵀ with the pivots square-rooted."
  5. Dim overlay, boxed: "Cholesky needs every pivot to be strictly positive. / That's exactly what 'positive definite' means." Below: "For [[1, ρ], [ρ, 1]], the pivots are 1 and 1 − ρ². / Watch that second one: it comes back when we push ρ to 1."

### 2.4e The shortcut: solve for G directly
`slide_2_4e_shortcut`
- **Clicks:**
  1. G Gᵀ multiplied out symbolically = Σ, and an empty G of g_ij entries.
  2. g₁₁ = √4 = 2, g₂₁ = 2/2 = 1, g₃₁ = 2/2 = 1, g₂₂ = √(5 − 1²) = 2, g₃₂ = (3 − 1·1)/2 = 1, g₃₃ = √(3 − 1² − 1²) = 1, each boxing its entries and filling G. Caption: "Same G, without the detour. The triangle trick, one column at a time."
  3. The general formulas for g_jj and g_ij, with "Diagonal: what's left of the variance, square-rooted." and "Below: what's left of the covariance, divided by the diagonal."

### 2.5 The algorithm and the hero animation
`slide_2_5_algorithm_and_hero`
- **Point:** four steps; G literally reshapes the plane.
- **Clicks:**
  1. "Back to our 2×2: G = [[2, 0], [1, 2]]" and the algorithm box: 1. Choose μ and Σ. 2. G = chol(Σ). 3. z ~ N(0, I). 4. x = μ + G z.
  2. Plane, grid, blue cloud, tracked point z = (0.5, −1); side panel x = μ + G z with the actual G and μ.
  3. "Step 1: apply G": the grid shears; the point moves to Gz = (1, −1.5).
  4. "Step 2: shift by μ": everything shifts, blue fades to orange, the point lands at x = (2, 0.5) with dashed guides.
  5. The side panel becomes "In NumPy" with the three-line code. **This exact frame carries over into Section 3** (`HeroFrame` in `components.py`).

---

## Section 3: Putting the sampler to work (`scenes/s3_sampler.py`, 26 clicks)

### 3.0 Section slide: Putting the sampler to work
`slide_3_0_section`
- **Clicks:**
  1. Opens on 2.5's final frame, which dims; the card appears on top: "3", "Putting the sampler to work", "Brief part (b): change Σ, watch the data".

### 3.1 Does it work?
`slide_3_1_does_it_work`
- **Clicks:** the orange cloud from 2.5 stays on screen.
  1. The card lifts; the grid, point, guides and code fade; the plane and cloud move left. Caption: "We built a sampler. Does it really produce Σ?"
  2. Target Σ and μ beside Σ̂ and μ̂ from 20,000 samples (2 decimals); matching entries flash green.
  3. The 1σ and 2σ contours.
  4. The 3×3 check: G from 2.4 run through the sampler; Σ beside the measured Σ̂ (flash green). Caption: "The G we computed by hand generates exactly the Σ we asked for."

### 3.2 Turning the dial: brief part (b)
`slide_3_2_turning_the_dial`
- **Clicks:** four panels in a row; "Same z in every panel. Only Σ changes." stays on screen.
  1. Bridge: "Now we trust the machine. Feed it different Σ and watch what G does to the same cloud."
  2. Identity: blue z cloud on a faint grid, Σ, then G is applied (grid deforms, cloud turns orange). "G does nothing. Circle stays a circle."
  3. Unequal variances [[4, 0], [0, 1]]: "Diagonal G: the 1D rule on each axis. Stretch by √4 and √1."
  4. Positive ρ = 0.8: "An off-diagonal entry shears the grid. Tilt up-right."
  5. Negative ρ = −0.8: "Flip its sign, flip the tilt."
  6. "G's diagonal sets the spread. G's off-diagonal sets the tilt."

### 3.3 The eigen view: reading the shape
`slide_3_3_eigen_view`
- **Clicks:**
  1. Σ = [[5, 4], [4, 5]] and its cloud. "Which directions does this cloud stretch along, and by how much?"
  2. A small test plane: (1, 0) → (5, 4). "Most directions get rotated."
  3. (1, 1) → (9, 9) = 9·(1, 1). "This one only stretches."
  4. (1, −1) → (1, −1) = 1·(1, −1). "And so does this one."
  5. Boxed: Σ v = λ v. "Directions Σ only stretches: eigenvectors. The stretch factors: eigenvalues."
  6. The 2σ ellipse with eigenvector arrows of length 2√λ (6 and 2) landing on its axes, plus λ, √λ, v rows. "...√eigenvalues set the spread: 3 one way, 1 the other."
  7. Morph to [[8, 2], [2, 5]]: cloud, ellipse and arrows follow live. "Not just a 45° trick: eigenvalues 9 and 4, so spreads 3 and 2."

### 3.4 Turning the dial until it breaks
`slide_3_4_until_it_breaks`
- **Point:** positive pivots work, a zero pivot collapses the data, a negative pivot is impossible.
- **Clicks:** live cloud (fixed z) on the left; live Cholesky pipeline on the right (Σ, elimination row [0, 1 − ρ²], D, G with G₂₂ boxed "the only entry giving the cloud its thickness", and the pivot gauge).
  1. ρ = 0.
  2. ρ → −0.9
  3. ρ → 0
  4. ρ → 0.99
  5. ρ → 0.999. "Pivot ≈ 0.002: a hairline cloud, but Cholesky still works." (green)
  6. ρ = 1: pivot 0, gauge amber, G = [[1, 0], [1, 0]], x₂ = x₁. "Pivot = 0. NumPy's Cholesky refuses: it needs every pivot strictly positive." (amber)
  7. ρ = 1.2: the cloud fades out, pivot −0.44, gauge red, G₂₂ = √−0.44 in red. "No data in the world has this covariance."
  8. Summary box: "Positive pivots: Cholesky works. / Zero pivot: the data collapses. / Negative pivot: the Σ is impossible."

### 3.5 Bonus: the same recipe in 3D (off by default)
`slide_3_5_three_d_bonus` in class `S3bThreeD`; only shown when `ENABLE_3D_BONUS = True` in `config.py`.
- **Clicks:**
  1. The 3D cloud for the 3×3 Σ from 2.4.
  2. A slow camera orbit.

---

## Section 4: Wrap-up (`scenes/s4_wrapup.py`, 12 clicks)

### 4.0 Section slide: Wrap-up
`slide_4_0_section`
- **Clicks:**
  1. Big "4", "Wrap-up", subtitle "Recap, limits and questions".

### 4.1 Recap
`slide_4_1_recap`
- **Clicks:**
  1. "Cholesky turns independent noise / into the correlation structure you chose." Blue N(0, I) cloud, yellow arrow x = μ + G z, orange N(μ, Σ) cloud with its 2σ ellipse.

### 4.2 Bookend: the smooth curves, explained
`slide_4_2_bookend`
- **Clicks:**
  1. "x = G z, where G is the 60 × 60 Cholesky factor of the kernel", then the same three jagged-to-smooth morphs as 1.1 (same z vectors). Caption: "Same trick, 60 dimensions instead of 2."

### 4.3 Limits, and what we've already seen
`slide_4_3_limits`
- **Clicks:** one point per click; the previous point dims.
  1. "Needs strictly positive pivots." "At ρ = 1 the pivot is 0 and Cholesky stops." Thumbnail of the 3.4 gauge at pivot 0 (amber).
  2. "Alternatives: eigendecomposition and SVD." "ρ = 1: λ = 2 and 0, √0 = 0 is fine (eigen view)", "NumPy's multivariate_normal defaults to SVD. Slower."
  3. "Gaussian only." "Real returns have fat tails: the portfolio risk can still be understated." "Fixes (multivariate t, copulas) still use x = μ + G z". Thumbnail of the 1.1 histogram with its VaR line, tagged "from the start".

### 4.4 Closing
`slide_4_4_closing`
- **Clicks:**
  1. "Thank you", "Questions?"

### A.1 Appendix (backup): the derivation is the same transformation
`slide_a_1_derivation`
- **Clicks:**
  1. f_x(x) = f_z(G⁻¹(x − μ)) / |det G|
  2. |det G| = |Σ|^(1/2)
  3. ‖G⁻¹(x − μ)‖² = (x − μ)ᵀ Σ⁻¹ (x − μ)
  4. "Together: exactly the density we started with."
  5. Check box at x = (2, 0.5) from z = (0.5, −1): quadratic form = 1.25, ‖z‖² = 1.25, f(x) ≈ 0.0213. Caption: "One equation derives the distribution and generates the data."
