# Speaker notes

Generated from the scene files by `python notes/build_notes.py`. Edit the notes in the scenes, then rerun it (or `make notes`).

## S1Problem

### 1.0  Title

Hi everyone. Our project is about simulating random vectors that follow a multivariate normal distribution with a covariance matrix we choose. The tool that makes it work is the Cholesky decomposition, and at the end we will play with different covariance structures.

### 1.0  Section slide: Problem

Let's start with the problem. We will look at what a computer can give us, what we actually want, and why the obvious fixes fail.

### 1.1  Real variables move together

Here are simulated daily returns for two stocks. On the left they are independent, so the cloud is round. On the right they tend to move together, which tilts the cloud. If we want to simulate a portfolio honestly, we need draws like the ones on the right.

### 1.2  What computers give us

This is what a computer gives us for free. One line of NumPy draws from the 1D standard normal, the familiar bell curve with mean 0 and variance 1. Each row here is two independent draws, so the vector z is N of 0 and I, and that independence is why the cloud is round.

### 1.3  What we want

What we actually want is a multivariate normal with a mean vector mu and a covariance matrix Sigma. The diagonal of Sigma holds the variances of each variable, and the off-diagonal entries hold how much they move together. Data drawn from it looks like this tilted orange cloud. How we made it is the rest of the talk.

### 1.4  What each entry does

Here is what each entry actually does, using the same random points the whole time. Making sigma xx bigger stretches the cloud along x, and making sigma yy smaller squeezes it in y. The covariance sets the tilt: negative tilts it down, and zero lines the ellipse up with the axes.

### 1.5  The question

So here is the whole problem in one picture. On the left is the round cloud we can generate, and on the right is the tilted cloud we want. Both plots use the same scale. So the question is: how do we change the distribution of the points to get from the left to the right?

### 1.6  Warm-up: we already do this in 1D

Let's warm up in one dimension, where we already know the answer. Take z from N of 0 and 1, stretch it by the standard deviation, then shift it by the mean; watch the mean line and the sigma markers move. The result is exactly N of 3 and 4.

### 1.7  The trap: scaling by the variance

Here is the trap. If we scale by the variance, 4, instead of the standard deviation, 2, the spread doubles and the variance becomes 16, four times too big. So the rule is: scale by the square root of the variance. Remember this, because in more dimensions we will need the square root of a matrix.

### 1.8  What is the square root of a matrix?

In more than one dimension, what is the square root of a matrix? Let's just try things and measure. Taking the square root of each entry gives a cloud that is too thin and the wrong covariance. Multiplying by Sigma itself overshoots badly, so we need a better idea.

## S2Solution

### 2.0  Section slide: Solution

Now the solution. We will find the right idea of a square root for a matrix, compute it with Cholesky, and turn it into a four line algorithm.

### 2.1  What a matrix does to covariance

Here is the one fact that explains everything. Remember z is the vector of independent standard normals the computer gives us, so its covariance is the identity. If x equals A times z, its covariance is A times I times A transpose, which is just A A transpose, and that explains the strange numbers from the square root attempts.

### 2.2  Defining a "square root" of Σ

So the right notion of a square root is any matrix A with A A transpose equal to Sigma, just like sigma times sigma is sigma squared in 1D. But it is not unique. Rotating the round cloud first changes nothing, so A times any rotation Q works too.

### 2.3  Cholesky decomposition

The rule we pick is the Cholesky decomposition: write Sigma as L L transpose, where L is lower triangular with a positive diagonal. It is unique, it exists exactly when Sigma is symmetric positive definite, and it is fast and stable, which is why every library uses it.

### 2.4  Computing it

Let's compute L for our example by hand. The first entry is the square root of 4, the entry below it is 2 divided by 2, and the last entry is whatever variance is left after removing that. Multiplying back gives Sigma, and the general formulas are the same pattern applied one column at a time.

### 2.5  Intuition: L is a mixing recipe

Here is the intuition for what L does. With unit variances, x1 is just z1, and x2 takes a slice of z1 plus some fresh noise z2. The slice of z1 is exactly the part x2 shares with x1, and that sharing is what we call correlation.

### 2.6  The algorithm

Putting it together, the algorithm is four lines: choose mu and Sigma, factor Sigma, draw independent normals, and compute mu plus L z. Watch the grid: L shears the whole plane, the circle becomes an ellipse, and then everything shifts by mu. Our tracked point lands exactly where the hand calculation says, and in NumPy it is three lines.

## S3Demo

### 3.0  Section slide: Demo

Time for the demo, which is part b of the brief. First we check that the sampler works, then we change Sigma and watch how the shape and spread of the data respond.

### 3.1  Does the sampler work?

Before playing with it, let's check that it works. We drew 20,000 samples and measured their mean and covariance; the dots you see are the first 800 of them. The measured values match the targets to about two decimal places, and the cloud sits inside the 1 and 2 sigma contours of the true density.

### 3.2  Covariance structures side by side

Now part b of the brief: different covariance structures. Every panel uses exactly the same z samples, so the only thing changing is Sigma. Equal variances give a circle, unequal variances stretch it along an axis, and a positive or negative correlation tilts it up or down.

### 3.3  Correlation sweep

Here we sweep the correlation live. The samples z never change; only rho does, through the mixing recipe from earlier. As rho approaches 1 the fresh-noise term shrinks and the cloud collapses onto a line. At exactly 1 there is no fresh noise left, Sigma is singular, and Cholesky fails.

### 3.4  The eigen view

One more way to read Sigma. Its eigenvectors point along the axes of the ellipse, and the eigenvalues say how much variance lies along each one. The arrows here are two standard deviations long, so they reach exactly to the 2 sigma contour.

## S3bThreeD

### 3.5  Bonus: the same recipe in 3D

The same recipe works in any dimension. Here is a three dimensional Sigma, factored with Cholesky and applied to 3D standard normals. As the camera moves, you can see the cloud is a tilted ellipsoid.

## S4WrapUp

### 4.0  Section slide: Wrap-up

To finish, a quick recap, where this is used, its limits, and one bonus connection back to the density formula from the start.

### 4.1  Recap

To recap: we start from independent noise that any computer can produce, factor our chosen Sigma into L times L transpose, and apply x equals mu plus L z. The result has exactly the correlation structure we asked for.

### 4.2  Where this is used

This is not just a classroom trick. Banks use it to simulate correlated asset returns for risk, Gaussian processes use it to draw random functions, and it is a standard way to generate realistic synthetic data with a known correlation structure.

### 4.3  Limits

There are limits. Cholesky needs Sigma to be strictly positive definite, as we saw with rho equal to 1. The output is always Gaussian, so it cannot produce skewed or fat-tailed data. When Sigma is close to singular, eigendecomposition or SVD are safer, and that is what NumPy's multivariate normal uses by default.

### 4.4  Bonus: the derivation is the same transformation

A bonus for the curious. If we run x equals mu plus L z backwards with the change of variables formula, the determinant of L gives the square root of the determinant of Sigma, and the squared length of z becomes the quadratic form. That is exactly the density from the start, and at our worked point both sides give the same numbers.

### 4.5  Closing

Thank you for listening. We are happy to take any questions, and we can jump back to any slide if it helps.
