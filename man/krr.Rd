\name{krr}
\alias{krr}
\title{
  Kernel Ridge Regression
}
\description{
  Fit kernel ridge regression, i.e. reproducing kernel Hilbert space regression.
}
\usage{
krr(x, y, group = NULL)
}
\arguments{
  \item{x}{
    a matrix of predictors.
  }
  \item{y}{
    a vector of response.
  }
  \item{group}{
    an optional vector of the same length as \code{y} which specifies the 
    group membership. The regression model is fitted separately for each
    group of observations but with the same scaling factors as well as 
    penalty amount. If omitted, a single group is assumed.
  }
}
\details{
  \code{krr} minimizes the sum of squared loss plus a penalty term on the 
  squared norm of the regression function. Gaussian kernel is used.
  Tuning parameters are chosen by minimizing the leave-one-out cross validated
  mean squared error. See the mathematical formulation in the reference.
}
\value{
  An object of class \code{krr}.
}
\references{
  \url{https://arxiv.org/abs/1606.01472}
}
\seealso{
  \code{\link{predict.krr}}
}
\examples{
  x <- matrix(rnorm(200 * 10), 200, 10)
  y <- x[, 1] + x[, 2] ^ 2 + x[, 3] * x[, 4] + rnorm(200)
  obj <- krr(x, y)
  
  xnew <- matrix(rnorm(1000 * 10), 1000, 10)
  ynew <- predict(obj, xnew)

  ytrue <- xnew[, 1] + xnew[, 2] ^ 2 + xnew[, 3] * xnew[, 4]
  mean((ynew - ytrue) ^ 2)  # MSE
}
\keyword{ models }
\keyword{ regression }
