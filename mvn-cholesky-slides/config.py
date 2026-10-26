"""Every tweakable number, matrix and timing for the deck lives here.

Scene files read from this module only. Change a value here, re-render, and
every slide that depends on it (matrices, readouts, axis ranges) updates.
"""

# --- Presenter and randomness ------------------------------------------------

PRESENTER_NAME = "Presenter name"  # shown on the title slide
RANDOM_SEED = 42  # one seed for the whole deck, so every render is identical

# --- Sample sizes ------------------------------------------------------------

N_DISPLAY = 800  # dots actually drawn on screen (Manim gets slow with thousands of Dots)
N_STATS = 20_000  # samples used only for numerical checks (sample mean and covariance)

# --- 1D warm-up: N(0, 1) -> N(3, 4) via x = 3 + 2z (slide 1.6) ---------------

ONE_D_MU = 3.0  # mean of the target 1D normal
ONE_D_SIGMA = 2.0  # standard deviation, so the variance is 4

# --- Main worked example, used across sections 1 to 4 -----------------------

MAIN_MU = [1.0, 2.0]  # target mean vector
MAIN_SIGMA = [
    [4.0, 2.0],
    [2.0, 5.0],
]  # target covariance; its Cholesky factor is [[2, 0], [1, 2]]
WORKED_Z = [0.5, -1.0]  # the tracked point; maps to x = mu + L z = [2.0, 0.5]

# --- Hook slide (1.1): two simulated "stock returns" -------------------------

HOOK_RHO = 0.8  # correlation of the right-hand (correlated) stock pair

# --- Entry-by-entry walk through Sigma (slide 1.4) ----------------------------

# Starting from MAIN_SIGMA, Sigma moves to each matrix in turn (one click each).
# Every matrix must be positive definite; a straight path between two positive
# definite matrices stays positive definite, so every in-between frame is valid.
# Captions use Pango markup so the subscripts render properly.
SIGMA_WALK = [
    ("Bigger σ<sub>xx</sub>: the cloud spreads out along x", [[9.0, 2.0], [2.0, 5.0]]),
    ("Smaller σ<sub>yy</sub>: the cloud squeezes in y", [[9.0, 2.0], [2.0, 1.5]]),
    ("Negative σ<sub>xy</sub>: the tilt flips", [[9.0, -3.0], [-3.0, 1.5]]),
    ("σ<sub>xy</sub> = 0: no tilt, the ellipse lines up with the axes", [[9.0, 0.0], [0.0, 1.5]]),
    ("Back to our example", MAIN_SIGMA),
]
SIGMA_WALK_SECONDS = 4.0  # seconds for each step of the walk
SIGMA_WALK_DECIMALS = 1  # decimal places of the live Sigma readout

# --- Part (b) covariance structures, shown side by side (zero mean, slide 3.2)

DEMO_STRUCTURES = [
    ("Identity", [[1.0, 0.0], [0.0, 1.0]]),
    ("Unequal variances", [[4.0, 0.0], [0.0, 1.0]]),
    ("Positive rho = 0.8", [[1.0, 0.8], [0.8, 1.0]]),
    ("Negative rho = -0.8", [[1.0, -0.8], [-0.8, 1.0]]),
]
DEMO_CAPTIONS = [  # one-line caption per panel, same order as DEMO_STRUCTURES
    "Circle",
    "Ellipse aligned with the axes",
    "Tilted up-right",
    "Tilted down-right",
]

# --- Square roots and the hero animation (slides 2.2 and 2.6) ---------------

ROTATION_DEMO_DEGREES = 90.0  # how far the round cloud is rotated on slide 2.2
HERO_GRID_HALF_WIDTH = 2  # the moving grid covers z in [-2, 2]^2; keep mu + L(grid) on the axes

# --- Mixing recipe (slide 2.5) -----------------------------------------------

MIXING_DEMO_RHO = 0.8  # rho used for the stacked variance-share bar

# --- Correlation sweep (slide 3.3) -------------------------------------------

RHO_SWEEP_PATH = [0.0, -0.9, 0.0, 0.99, 0.999]  # values visited in order
RHO_SWEEP_SECONDS_PER_LEG = 6.0  # seconds spent moving between two values
RHO_FAILURE = 1.0  # final value, where Cholesky breaks down
SWEEP_DECIMALS = 3  # live readouts need 3 places so 0.999 does not show as 1.00

# --- Ellipses, eigenvectors and axes -----------------------------------------

CONTOUR_KS = [1.0, 2.0]  # k-sigma density contours drawn on slide 3.1
QUESTION_ELLIPSE_K = 2.0  # k-sigma ellipse drawn around the target cloud (1.3 to 1.5)
EIGEN_ARROW_K = 2.0  # eigenvector arrows are scaled by k * sqrt(lambda) (slide 3.4)
AXIS_N_SD = 3.5  # axes span mean +/- this many standard deviations
AXIS_ROUND_STEP = 1.0  # axis limits are rounded outward to a multiple of this
DISPLAY_DECIMALS = 2  # decimal places for every number shown on screen
DENSITY_DECIMALS = 4  # the density value on slide 4.4 is small, so it gets more places

# --- Optional 3D bonus (slide 3.5) -------------------------------------------

ENABLE_3D_BONUS = False  # optional stretch slide, see 3.5
THREE_D_MU = [0.0, 0.0, 0.0]  # mean of the 3D cloud
THREE_D_SIGMA = [
    [1.0, 0.7, 0.3],
    [0.7, 1.0, 0.5],
    [0.3, 0.5, 1.0],
]  # covariance of the 3D cloud (must be positive definite)
N_DISPLAY_3D = 400  # 3D dots are spheres, much slower to draw than 2D dots
THREE_D_ROTATION_DEGREES = 120.0  # how far the camera orbits
THREE_D_ROTATION_SECONDS = 16.0  # duration of the slow camera rotation

# --- Animation timing constants (seconds) ------------------------------------

T_SHORT = 1.2  # small reveals: a caption, a check mark
T_MEDIUM = 2.4  # standard reveals: a matrix, a cloud
T_LONG = 5.0  # big moments: the hero transform, curve stretches
T_SETTLE = 0.1  # hold at the end of each slide so it ends on the exact final frame
