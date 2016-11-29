\name{predict.listdtr}
\alias{predict.listdtr}
\title{
Treatment Recommendation Dictated by 
List-based Optimal Dynamic Treatment Regime
}
\description{
Provide treatment recommendation at a given stage using the features,
based on a given \code{listdtr} object.
}
\usage{
\method{predict}{listdtr}(object, xnew, stage, ...)
}
\arguments{
  \item{object}{
    an object of class \code{listdtr}, 
    usually the return value of function \code{listdtr}.
  }
  \item{xnew}{
    a matrix of features.
  }
  \item{stage}{
    an integer that specifies the stage.
  }
  \item{\dots}{
    further arguments passed to or from other methods.
  }
}
\value{
A factor vector that gives the estimated optimal treatment
for the features presented in each row of \code{xnew}.
The length is the same as 
the number of rows in \code{xnew}.
The levels are the unique values of treatments actually received
at that stage.
}
\seealso{
  \code{\link{listdtr}}, \code{\link{apply.rule}}
}
\examples{
  # see examples for listdtr
}
\keyword{ tree }
