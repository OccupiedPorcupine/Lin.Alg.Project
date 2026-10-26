"""Every tweakable number, matrix and timing for the deck lives here.

Scene files read from this module only. Change a value here, re-render, and
every slide that depends on it (matrices, readouts, axis ranges) updates.
"""

# --- Presenter and randomness ------------------------------------------------

PRESENTER_NAME = "Presenter name"  # shown on the title slide
RANDOM_SEED = 42  # one seed for the whole deck, so every render is identical

# --- Sample sizes ------------------------------------------------------------

N_DISPLAY = 800  # dots actually drawn on screen (Manim gets slow with thousands of Dots)
N_STATS = 20_000  # samples used only for numerical checks (sample mean, covariance, correlation)
N_PRINTED_ROWS = 3  # rows of Z printed in the code on slide 1.2, each flown to its dot

# --- 1.1 hook: portfolio panel (two unit-variance stocks, 50/50 weights) -----

PORTFOLIO_WEIGHTS = [0.5, 0.5]  # share of the portfolio in stock A and stock B
PORTFOLIO_RHO_INDEPENDENT = 0.0  # the "ignore correlation" assumption (blue)
PORTFOLIO_RHO_CORRELATED = 0.8  # the realistic, correlated stocks (orange)
VAR_LEVEL = 0.05  # 5% worst-case loss line (Value at Risk)
PORTFOLIO_HIST_BINS = 36  # bars in each histogram of portfolio returns

# --- 1.1 hook: Gaussian process panel (replayed on 4.2) ----------------------

GP_N_POINTS = 60  # points along the x axis; one curve is one 60-dimensional normal draw
GP_X_RANGE = (0.0, 10.0)  # x range the curve is drawn over
GP_LENGTH_SCALE = 1.0  # how far apart two points can be and still move together
GP_JITTER = 1e-6  # added to the kernel diagonal; Cholesky fails without it
GP_N_CURVES = 3  # how many jagged-to-smooth morphs are shown

# --- 1.1 hook: synthetic data panel (illustrative "original" correlation) ----

SYNTH_VARIABLES = ["Math", "Physics", "English"]  # column names of the fake exam table
SYNTH_CORRELATION = [
    [1.0, 0.7, 0.3],
    [0.7, 1.0, 0.35],
    [0.3, 0.35, 1.0],
]  # the "original data" correlations the synthetic copy should reproduce
SYNTH_TABLE_ROWS = 4  # rows of the synthetic table shown ticking in under the heatmaps
SYNTH_SCORE_MEAN = 65.0  # synthetic rows are shown as exam scores: mean + sd * x
SYNTH_SCORE_SD = 12.0

# --- 1.6 1D warm-up: N(0, 1) -> N(3, 4) via x = 3 + 2z -----------------------

ONE_D_MU = 3.0  # mean of the target 1D normal
ONE_D_VARIANCE = 4.0  # the scale used is sqrt(ONE_D_VARIANCE) = 2
ONE_D_NAIVE_SCALE = 4.0  # the wrong attempt: multiplying by the variance
ONE_D_AXIS_N_SD = 2.5  # the 1D axes hold the widest curve (the naive one) out to this many sds

# --- 1.7 failed square roots ---------------------------------------------------

MEASURED_DECIMALS = 1  # measured covariances on 1.7 are estimates, so one decimal is honest

# --- Main worked example, used across sections 1 to 3 and the appendix -------

MAIN_MU = [1.0, 2.0]  # target mean vector
MAIN_SIGMA = [
    [4.0, 2.0],
    [2.0, 5.0],
]  # target covariance; its Cholesky factor G is [[2, 0], [1, 2]]
WORKED_Z = [0.5, -1.0]  # the tracked point; maps to x = mu + G z = [2.0, 0.5]

# --- 2.4 hand computation (top-left 2x2 block equals MAIN_SIGMA) -------------

EXAMPLE_3X3_SIGMA = [
    [4.0, 2.0, 2.0],
    [2.0, 5.0, 3.0],
    [2.0, 3.0, 3.0],
]  # G is [[2, 0, 0], [1, 2, 0], [1, 1, 1]]

# --- 3.2 covariance structures, shown side by side (zero mean) ---------------

DEMO_STRUCTURES = [
    ("Identity", [[1.0, 0.0], [0.0, 1.0]]),
    ("Unequal variances", [[4.0, 0.0], [0.0, 1.0]]),
    ("Positive rho = 0.8", [[1.0, 0.8], [0.8, 1.0]]),
    ("Negative rho = -0.8", [[1.0, -0.8], [-0.8, 1.0]]),
]
# One caption per panel, same order. {root_x} and {root_y} are filled in from
# the panel's Sigma (for example "√4"), so the text follows the matrix.
DEMO_CAPTIONS = [
    "G does nothing. Circle stays a circle.",
    "Diagonal G: the 1D rule on each axis. Stretch by {root_x} and {root_y}.",
    "An off-diagonal entry shears the grid. Tilt up-right.",
    "Flip its sign, flip the tilt.",
]
DEMO_GRID_HALF_WIDTH = 3  # the faint grid G deforms in each 3.2 panel covers z in [-3, 3]^2

# --- 3.3 eigen view ----------------------------------------------------------

EIGEN_EXAMPLE_SIGMA = [[5.0, 4.0], [4.0, 5.0]]  # eigenvalues 9 and 1
EIGEN_EXAMPLE_SIGMA_2 = [[8.0, 2.0], [2.0, 5.0]]  # eigenvalues 9 and 4
EIGEN_TEST_DIRECTIONS = [[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]]  # arrows multiplied by Sigma
ENABLE_EIGEN_SECOND_EXAMPLE = True  # reveal 4: morph to EIGEN_EXAMPLE_SIGMA_2
EIGEN_ARROW_K = 2.0  # eigenvector arrows are scaled by k * sqrt(lambda)

# --- 3.4 correlation sweep until it breaks -----------------------------------

RHO_SWEEP_PATH = [0.0, -0.9, 0.0, 0.99, 0.999]  # values visited in order
RHO_SWEEP_SECONDS_PER_LEG = 6.0  # seconds spent moving between two values
RHO_ZERO_PIVOT = 1.0  # the pivot 1 - rho^2 is exactly zero here
RHO_NEGATIVE_PIVOT = 1.2  # the pivot is negative here: no such data exists
ENABLE_NEGATIVE_PIVOT_REVEAL = True  # show the rho = 1.2 step
SWEEP_DECIMALS = 3  # live readouts need 3 places so 0.999 does not show as 1.00

# --- 2.2 rotation and 2.5 hero animation -------------------------------------

ROTATION_DEMO_DEGREES = 90.0  # how far the round cloud is rotated on slide 2.2
HERO_GRID_HALF_WIDTH = 2  # the moving grid covers z in [-2, 2]^2; keep mu + G(grid) on the axes

# --- Ellipses and axes -------------------------------------------------------

CONTOUR_KS = [1.0, 2.0]  # k-sigma density contours drawn on slides 1.4 and 3.1
QUESTION_ELLIPSE_K = 2.0  # k-sigma ellipse drawn around target clouds
FORMULA_POINT_DISTANCE = 2.0  # slide 1.4: ruler distance of points A and B from mu
FORMULA_ZOOM_HALF_WIDTH = 4.0  # slide 1.4: the A and B plot shows mu +/- this much
AXIS_N_SD = 3.5  # axes span mean +/- this many standard deviations
AXIS_ROUND_STEP = 1.0  # axis limits are rounded outward to a multiple of this
DISPLAY_DECIMALS = 2  # decimal places for every number shown on screen
DENSITY_DECIMALS = 4  # the density value in the appendix is small, so it gets more places

# --- Optional 3D bonus (slide 3.5) -------------------------------------------

ENABLE_3D_BONUS = False  # optional stretch slide, see 3.5; uses EXAMPLE_3X3_SIGMA
N_DISPLAY_3D = 400  # 3D dots are spheres, much slower to draw than 2D dots
THREE_D_ROTATION_DEGREES = 120.0  # how far the camera orbits
THREE_D_ROTATION_SECONDS = 16.0  # duration of the slow camera rotation

# --- Animation timing constants (seconds) ------------------------------------

T_SHORT = 1.2  # small reveals: a caption, a check mark
T_MEDIUM = 2.4  # standard reveals: a matrix, a cloud
T_LONG = 5.0  # big moments: the hero transform, curve stretches
T_SETTLE = 0.1  # hold at the end of each slide so it ends on the exact final frame
