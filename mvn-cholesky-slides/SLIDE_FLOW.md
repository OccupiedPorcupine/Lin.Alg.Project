# Slide flow

The deck as it is now, slide by slide. Each **click** is one press of the
right arrow in the presenter. Numbers on screen come from `config.py`.

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

---

## Section 1: Problem (`scenes/s1_problem.py`, 23 clicks)

### 1.0 Title slide
`slide_1_0_title`
- **Point:** name the topic and method.
- **Clicks:**
  1. "Simulating the Multivariate Normal / with a Given Covariance Matrix", subtitle "Cholesky decomposition and covariance structures", presenter name.

### 0.0 Section slide: Motivating problems (where can this be implemented)
### 0.1 Where this is used
`slide_0_1_where_used`
- **Clicks:**
  go deeper for each topic, one slide for each one. with the title being the use case. 
  1. Monte Carlo risk and portfolio simulation
    - put in some graphs that that's relavant to having correlated multivariate data 
  2. Gaussian processes
    - what are gaussian processes 
  3. Synthetic data generation
    - give a use case for this

### 1.0 Section slide: Problem
`slide_1_0_section`
- **Point:** open section 1.
- **Clicks:**
  1. Big "1", "Problem", subtitle "What we have, what we want, and why it is not obvious".

### 1.1 Real variables move together
`slide_1_1_hook`
- **Point:** real variables are correlated, so we need draws that move together.
- **Clicks:**
  1. "Stock A vs Stock B returns (simulated)": blue round cloud (independent) beside an orange tilted cloud (correlated, rho = 0.8). Caption: "To simulate a portfolio, we need random draws that move together."

### 1.2 What computers give us
`slide_1_2_what_computers_give`
- **Point:** computers give us independent standard normals, N(0, I). (label the normal distribution. We dont want to rely on people remembering the exact form fo the normal distribution, so we shoul write it there)
- **Clicks:**
  1. NumPy code (`rng.standard_normal((800, 2))`), the 1D density N(0, 1): phi(z) = e^(-z^2/2) / sqrt(2 pi), then z = (z1, z2) ~ N(0, I), with the blue round cloud on the right. Caption: "Independent. No correlation."

### 1.3 What we want
`slide_1_3_what_we_want`
- **Point:** we want N(mu, Sigma); Sigma is the blueprint. <- just have N(mu, Sigma) instead of the entire multivariate formula
- **Clicks:**
  1. The MVN density formula f(x) in display form.
  2. mu = [mu_x; mu_y] and Sigma = [sigma_xx, sigma_xy; sigma_yx, sigma_yy] (symbols). Diagonal boxed: "variances". Off-diagonal dashed: "covariances".
  3. Orange tilted cloud for the example mu, Sigma with its 2-sigma ellipse.

### 1.4 What each entry does
`slide_1_4_what_each_entry_does`
- **Point:** each entry of Sigma has a visible effect on the cloud.
- **Clicks:** Sigma shown as symbols = live numbers; the cloud and ellipse redraw from the same z as Sigma changes; changed entries are boxed.
  1. Start at Sigma = [[4, 2], [2, 5]].
  2. sigma_xx 4 to 9. "Bigger sigma_xx: the cloud spreads out along x"
  3. sigma_yy 5 to 1.5. "Smaller sigma_yy: the cloud squeezes in y"
  4. sigma_xy 2 to -3. "Negative sigma_xy: the tilt flips"
  5. sigma_xy -3 to 0. "sigma_xy = 0: no tilt, the ellipse lines up with the axes"
  6. Back to [[4, 2], [2, 5]]. "Back to our example"

### 1.5 The question
`slide_1_5_the_question`
- **Point:** state the problem.
- **Clicks:**
  1. Blue round cloud, an arrow with "?", orange tilted cloud (same scale), labeled N(0, I) and N(mu, Sigma). Caption: "How do we change the distribution of the points?"

### 1.6 Warm-up: we already do this in 1D
`slide_1_6_one_d_warmup`
- **Point:** in 1D we stretch by sigma, then shift by mu.
- **Clicks:** mean line and sigma markers follow the curve throughout.
  1. Blue bell curve z ~ N(0, 1) with mu = 0, sigma = 1 markers.
  2. "1. Stretch by sigma = 2": curve widens; 2z ~ N(0, 4).
  3. "2. Shift by mu = 3": curve moves and turns orange; x = 3 + 2z ~ N(3, 4).

### 1.7 The trap: scaling by the variance
`slide_1_7_the_trap`
- **Point:** scale by the square root of the variance, not the variance.
- **Clicks:**
  1. Check mark: "Scale by sigma = sqrt(4) = 2", 3 + 2z ~ N(3, 4), orange curve with sigma = 2 markers.
  2. Cross: "Scale by sigma^2 = 4", 3 + 4z ~ N(3, 16), red dashed curve with sigma = 4 markers.
  3. Green box: "Scale by the square root of the variance", x = mu + sqrt(sigma^2) z = mu + sigma z.

### 1.8 What is the square root of a matrix?
`slide_1_8_matrix_square_root`
- **Point:** element-wise thinking fails in nD.
- **Clicks:** target Sigma shown at the top throughout.
  1. Attempt 1: element-wise root A = [[2, sqrt2], [sqrt2, sqrt5]], arrow, cloud vs dashed target ellipse, arrow, A^2 approx [[6.03, 6.07], [6.07, 7.13]], red cross.
  2. Attempt 2: multiply by Sigma, same layout, Sigma^2 approx [[19.99, 18.26], [18.26, 29.70]], red cross.
  3. Caption: "Element-wise thinking fails. We need a better definition."

---

## Section 2: Solution (`scenes/s2_solution.py`, 28 clicks)

### 2.0 Section slide: Solution
`slide_2_0_section`
- **Clicks:**
  1. Big "2", "Solution", subtitle "A square root for matrices".

### 2.1 What a matrix does to covariance
`slide_2_1_covariance_of_az`
- **Point:** Cov(Az) = A A^T.
- **Clicks:** each derivation line morphs out of the one above.
  1. "x = Az, mu = 0", then "z ~ N(0, I): independent standard normals, so E[zz^T] = I", then Cov(x) = E[xx^T].
  2. = E[A z z^T A^T]
  3. = A E[z z^T] A^T
  4. = A I A^T
  5. = A A^T (boxed)
  6. Caption: "x is still normal: a linear transform of a Gaussian stays Gaussian."
  7. Side box "Back to the failed square roots:" A A^T = [[6, 5.99], [5.99, 7]], Sigma Sigma^T = [[20, 18], [18, 29]].

### 2.2 Defining a "square root" of Sigma
`slide_2_2_square_root_of_sigma`
- **Point:** any A with A A^T = Sigma works, so many exist.
- **Clicks:**
  1. sigma times sigma = sigma^2 beside A A^T = Sigma. "Any A with A A^T = Sigma works."
  2. (AQ)(AQ)^T = A Q Q^T A^T = A A^T, "Q any rotation". Blue round cloud rotates 90 degrees and stays round. Caption: "Many square roots exist. We need a rule to pick one."

### 2.3 Cholesky decomposition
`slide_2_3_cholesky`
- **Point:** Cholesky is the rule we pick.
- **Clicks:**
  1. Sigma = L L^T, with a 3x3 lower-triangular L (zeros greyed), "lower triangular, positive diagonal".
  2. "1. Unique."
  3. "2. Exists exactly when Sigma is symmetric positive definite."
  4. "3. Fast (about n^3/3 operations) and numerically stable."

### 2.4 Computing it
`slide_2_4_computing_it`
- **Point:** compute L by hand; the general formula is the same pattern.
- **Clicks:** Sigma and L = [[?, 0], [?, ?]] on the left; entries fill in as computed.
  1. Sigma and the empty L.
  2. L11 = sqrt(Sigma11) = sqrt4 = 2
  3. L21 = Sigma21 / L11 = 2/2 = 1
  4. L22 = sqrt(Sigma22 - L21^2) = sqrt(5 - 1) = 2
  5. Verify: L L^T = Sigma, green check.
  6. Clear; the general formulas for L_jj and L_ij. Caption: "Same pattern, one column at a time".

### 2.5 Intuition: L is a mixing recipe
`slide_2_5_mixing_recipe`
- **Point:** x2 borrows part of x1's randomness; that shared part is the correlation.
- **Clicks:**
  1. Sigma(rho) = [[1, rho], [rho, 1]] to L = [[1, 0], [rho, sqrt(1 - rho^2)]].
  2. x1 = z1 and x2 = rho z1 + sqrt(1 - rho^2) z2 (z1 blue, z2 purple).
  3. Stacked bar for rho = 0.8: 0.64 shared with x1, 0.36 fresh noise. Caption: "x2 borrows a fraction of x1's randomness. The shared part is the correlation."

### 2.6 The algorithm
`slide_2_6_algorithm_and_hero`
- **Point:** four steps; L literally reshapes the plane.
- **Clicks:**
  1. Algorithm box: 1. Choose mu and Sigma. 2. L = chol(Sigma). 3. z ~ N(0, I). 4. x = mu + L z.
  2. Grid, blue cloud and the tracked point z = (0.5, -1); side panel x = mu + L z with the actual L and mu.
  3. "Step 1: apply L": the grid shears and the circle becomes an ellipse; point at Lz = (1, -1.5).
  4. "Step 2: shift by mu": everything moves and the cloud turns orange; point lands at x = (2, 0.5) with dashed guides and the 2-sigma ellipse.
  5. Side panel replaced by the NumPy code (`np.linalg.cholesky`, `rng.standard_normal`, `X = mu + Z @ L.T`).

---

## Section 3: Demo (`scenes/s3_demo.py`, 16 clicks)

### 3.0 Section slide: Demo
`slide_3_0_section`
- **Clicks:**
  1. Big "3", "Demo", subtitle "Brief part (b): change Sigma, watch the data".

### 3.1 Does the sampler work?
`slide_3_1_verification`
- **Point:** the sampler reproduces Sigma.
- **Clicks:**
  1. Orange cloud with 1-sigma and 2-sigma contours.
  2. Table: target Sigma and mu beside the values measured from 20,000 samples. Caption: "The sampler reproduces Sigma."

### 3.2 Covariance structures side by side
`slide_3_2_structures_side_by_side`
- **Point:** with the same z, Sigma alone sets the shape. Caption throughout: "Same z samples in every panel. Only Sigma changes."
- **Clicks:** a 2x2 grid, one panel per click.
  1. Identity: "Circle"
  2. Unequal variances [[4, 0], [0, 1]]: "Ellipse aligned with the axes"
  3. Positive rho = 0.8: "Tilted up-right"
  4. Negative rho = -0.8: "Tilted down-right"

### 3.3 Correlation sweep
`slide_3_3_correlation_sweep`
- **Point:** correlation tilts and squeezes the cloud; at rho = 1 Cholesky fails.
- **Clicks:** live readouts for rho and sqrt(1 - rho^2), plus L.
  1. Cloud at rho = 0.
  2. rho to -0.9
  3. rho to 0
  4. rho to 0.99
  5. rho to 0.999 (almost a line)
  6. rho = 1: sqrt(1 - rho^2) = 0 in red, Sigma = [[1, 1], [1, 1]], det(Sigma) = 0. Red caption: "Cholesky fails: Sigma is not positive definite."

### 3.4 The eigen view
`slide_3_4_eigen_view`
- **Point:** eigenvectors set the ellipse's axes, eigenvalues the spread.
- **Clicks:**
  1. Cloud, 2-sigma ellipse, Sigma.
  2. Arrow lambda1, "lambda1 approx 6.56, v1 approx (0.62, 0.79)".
  3. Arrow lambda2, "lambda2 approx 2.44, v2 approx (0.79, -0.62)". Caption: "Eigenvectors set the ellipse's axes. Eigenvalues set the spread along them."

### 3.5 Bonus: the same recipe in 3D (off by default)
`slide_3_5_three_d_bonus` in class `S3bThreeD`; only shown when `ENABLE_3D_BONUS = True` in `config.py`.
- **Clicks:**
  1. A 3D correlated cloud on 3D axes.
  2. A slow camera orbit.

---

## Section 4: Wrap-up (`scenes/s4_wrapup.py`, 14 clicks)

### 4.0 Section slide: Wrap-up
`slide_4_0_section`
- **Clicks:**
  1. Big "4", "Wrap-up", subtitle "Recap, uses and limits".

### 4.1 Recap
`slide_4_1_recap`
- **Clicks:**
  1. "Cholesky turns independent noise / into the correlation structure you chose." Blue cloud, arrow labeled x = mu + L z, orange cloud.

### 4.3 Limits
`slide_4_3_limits`
- **Clicks:**
  1. Needs a positive definite Sigma.
  2. Only Gaussian data: no skew and no fat tails.
  3. Alternatives: eigendecomposition and SVD ("Slower, but more robust near singular Sigma."). Caption: "NumPy's multivariate_normal defaults to SVD."

### 4.4 Bonus: the derivation is the same transformation
`slide_4_4_bonus_derivation`
- **Point:** x = mu + L z also derives the density.
- **Clicks:**
  1. f_x(x) = f_z(L^-1 (x - mu)) / |det L|
  2. |det L| = |Sigma|^(1/2)
  3. ||L^-1 (x - mu)||^2 = (x - mu)^T Sigma^-1 (x - mu)
  4. "Together: exactly the density we started with."
  5. Check box at x = (2, 0.5): quadratic form = 1.25, ||z||^2 = 1.25, f(x) approx 0.0213. Caption: "One equation derives the distribution and generates the data."

### 4.5 Closing
`slide_4_5_closing`
- **Clicks:**
  1. "Thank you", "Questions?"
