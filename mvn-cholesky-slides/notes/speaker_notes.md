# Speaker notes

Generated from the scene files by `python notes/build_notes.py`. Edit the notes in the scenes, then rerun it (or `make notes`).

## S1Problem

### 1.0  Title

Hi everyone. Our project is about simulating random vectors that follow a multivariate normal distribution with a covariance matrix we choose. The tool that makes it work is the Cholesky decomposition, and at the end we will play with different covariance structures.

### 1.0  Section slide: Problem

Let's start with the problem. We will see where correlated randomness matters, what a computer can give us, what we actually want, and why the obvious fixes fail.

### 1.1  Hook

Before any math, here are two places where correlated randomness matters. A bank simulating a portfolio underestimates its risk if it treats the stocks as independent, and synthetic data is only useful if it keeps the real correlations. Both need random numbers that move together.

### 1.2  What computers give us

Here's what a computer gives us for free. One line of numpy and we get 800 rows, and each row is a point. Every draw is independent, and the two columns have nothing to do with each other, so the correlation is zero and the cloud is round. The rest of this talk is about mixing those two columns together.

### 1.3  What we want

What we actually want is a multivariate normal with a mean vector mu and a covariance matrix Sigma. Mu is the center of the cloud. The diagonal of Sigma holds the variances, how spread out each variable is, and the off-diagonal entry is the covariance, how the two move together. How we made this cloud is the rest of the talk.

### 1.4  Reading the formula: same idea as 1D

This formula looks scary, but you already know it. Here's the 1D bell curve, and if I rearrange it a little, it has exactly the same shape as the multivariate one. Numbers become vectors, σ² becomes Σ, and dividing becomes multiplying by an inverse. The only genuinely new idea is the distance term: it measures distance in units of spread, so a point along the long axis counts as closer than one the same ruler distance along the short axis.

### 1.5  The question

So here is the whole problem in one picture. On the left is the round cloud a computer gives us, and on the right is the tilted cloud we want, drawn to the same scale. The question for the rest of the talk is how to turn one into the other.

### 1.6  Warm-up: we already do this in 1D

Let's start with something we already know. We want a normal with mean 3 and variance 4. The obvious move is to multiply by 4, but the curve overshoots and the variance jumps to 16, because stretching by a multiplies the variance by a squared. So we stretch by the square root of 4, which is 2, then shift by 3. To get variance sigma squared, you scale by its square root.

### 1.7  The nD catch: the variance becomes a matrix

Now go up a dimension. The variance is no longer one number, it's a whole matrix. If we make the same mistake as before and multiply by Σ, we get Σ squared. And taking the square root of each entry fails too. So what does the square root of a matrix even mean?

## S2Solution

### 2.0  Section slide: Solution

Now the solution. We will work out what a matrix does to the spread of the data, use that to define the square root of a matrix, compute it by hand, and turn it into a four line algorithm.

### 2.1  What a matrix does to the spread

Let's name our characters. z is the random point we start with, A is the matrix we multiply by, and x is the point we get out. Each new variable is a mix of z₁ and z₂, independent spreads add, and covariance only comes from shared ingredients. Put those in a matrix and it's exactly A times A transpose, the nD version of 'scaling by a multiplies variance by a²'. So the real question is: what A gives A Aᵀ = Σ?

### 2.2  The plan: Cholesky decomposition

A A transpose equals Σ gives three equations, but there are four unknowns, so we get one free choice. Cholesky's choice is to make A lower triangular, and we call that matrix G. Here is the plan for finding it: eliminate to get L U, pull the pivots out into D, use symmetry to get L D L transpose, then split D in half. Next we do each step by hand on a 3 by 3. If anyone asks: with a positive diagonal G is unique, and it costs about n cubed over 3 operations.

### 2.3a  By hand, step 1: LU (ordinary elimination)

We already know how to factor a matrix: Gaussian elimination. Each step subtracts a multiple of one row from another. We record those multipliers in L, and what's left is U.

### 2.3b  Step 2: LDV (pull the pivots out)

U's diagonal holds the pivots. If we pull them out into a diagonal matrix D, we get L D V.

### 2.3c  Step 3: LDLᵀ (symmetry gives V for free)

Because Σ is symmetric, V turns out to be exactly L transposed. So Σ = L D Lᵀ.

### 2.3d  Step 4: split D in half, giving G Gᵀ

Now the trick: split D into two square roots and give one to each side. L √D is our G, and Σ = G Gᵀ. That's Cholesky. It needs every pivot to be strictly positive, which is exactly the positive definite condition.

### 2.4  The algorithm

Back to our 2 by 2. Its Σ is the top-left corner of the 3 by 3, so its G is the top-left corner of the G we just found: 2, 0, 1, 2. The whole algorithm is four lines: choose μ and Σ, factor Σ, draw independent normals, and compute μ plus G z. Watch the grid: G shears the whole plane, the circle becomes an ellipse, and then everything shifts by μ. Our tracked point lands exactly at 2, 0.5, and in NumPy it is three lines.

## S3Sampler

### 3.0  Section slide: Putting the sampler to work

The sampler is built, so let's put it to work. This section is part b of the brief: we check that it really produces Sigma, then turn the dial on Sigma and watch the data respond.

### 3.1  Does it work?

We built a sampler, but does it really produce the Sigma we asked for? Measured from 20,000 samples, the mean and covariance match the targets to about two decimal places, and the dots fill the 1 and 2 sigma contours just as the density predicts. The 3 by 3 G we computed by hand in 2.3 passes the same test.

### 3.2  Turning the dial

Now we trust the machine, so let's feed it different Sigmas. Every panel starts from exactly the same z, so the only thing changing is G. The identity does nothing, a diagonal G stretches each axis by a square root, and an off-diagonal entry shears the grid and tilts the cloud one way or the other.

### 3.3  Turning the dial until it breaks

Let's run the same machine from 2.3 while turning rho up. Elimination leaves a second pivot of 1 minus rho squared, and its square root is the only entry in G that gives the cloud any thickness. At exactly rho equals 1 the pivot hits zero, x2 is just a copy of x1, and NumPy refuses to factor it. Push past 1 and the pivot goes negative: there's no square root at all, because no real data could have that covariance.

## S3bThreeD

### 3.4  Bonus: the same recipe in 3D

The same recipe works in any dimension. Here is the 3 by 3 Sigma from our hand calculation, factored with Cholesky and applied to 3D standard normals. As the camera moves, you can see the cloud is a tilted ellipsoid.

## S4WrapUp

### 4.0  Section slide: Wrap-up

To finish, a quick recap, a second look at the smooth curves from the start, and the limits of the method.

### 4.1  Recap

To recap: we start from independent noise that any computer can produce, factor our chosen Sigma into G times G transpose, and apply x equals mu plus G z. The result has exactly the correlation structure we asked for.

### 4.2  Limits

Three limits. First, Cholesky needs every pivot strictly positive; at rho equal to 1 the pivot hit zero and it stopped. Second, the eigen square root (the eigen view, in the appendix) still works there, because the square root of a zero eigenvalue is just zero, which is why NumPy's multivariate normal uses SVD by default, even though it is slower. Third, this only makes Gaussian data, and real returns have fat tails, but the usual fixes, like multivariate t and Gaussian copulas, still get their correlation from the same x equals mu plus G z.

### 4.3  Closing

Thank you for listening. We are happy to take any questions, and we can jump back to any slide if it helps.

### A.1  Appendix: the derivation is the same transformation

A backup slide, in case anyone asks where the density formula comes from. If we run x equals mu plus G z backwards with the change of variables formula, the determinant of G gives the square root of the determinant of Sigma, and the squared length of z becomes the quadratic form. That is exactly the density from slide 1.4, and at our worked point both sides give the same numbers.

### A.2  Appendix: the eigen view

Cholesky told us how to build this cloud. Eigenvectors tell us how to read it. Most arrows turn when I multiply them by Sigma, but two directions only stretch: the diagonal by 9 and the other diagonal by 1. Their square roots, 3 and 1, are exactly how spread out the cloud is along each axis.
