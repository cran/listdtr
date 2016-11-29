\name{print.listdtr}
\alias{print.listdtr}
\title{
Representation of List-based Regimes in Words
}
\description{
Describe the given regime in words and print them.
}
\usage{
\method{print}{listdtr}(x, stages = NULL, digits = 3L, \dots)
}
\arguments{
  \item{x}{
    an object of class \code{listdtr}.
  }
  \item{stages}{
    an integer scalar / vector that specifies the stage(s) of interest.
    Default to all stages.
  }
  \item{digits}{
    number of decimal digits to show.
  }
  \item{\dots}{
    further arguments passed to or from other methods.
  }
}
\value{
Return \code{object} invisibly.
}
\seealso{
  \code{\link{listdtr}},
  \code{\link{show.rule}}
}
\examples{
  # see examples for listdtr
}
\keyword{ tree }
