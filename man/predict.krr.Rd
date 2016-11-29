\name{predict.krr}
\alias{predict.krr}
\title{
Prediction for Kernel Ridge Regression
}
\description{
Predict response given predictors, based on a \code{krr} object.
}
\usage{
\method{predict}{krr}(object, xnew, \dots)
}
\arguments{
  \item{object}{
    an object of class\code{krr}, usually the return value of 
    function \code{krr}
  }
  \item{xnew}{
    a matrix of predictors.
  }
  \item{\dots}{
    further arguments passed to or from other methods.
  }
}
\value{
  A vector of predicted response.
}
\seealso{
  \code{\link{krr}}
}
\examples{
  # see examples for krr
}
\keyword{ models }
\keyword{ regression }
