\name{plot.listdtr}
\alias{plot.listdtr}
\title{
Representation of List-based Regimes in Diagrams
}
\description{
Describe the given regime in diagrams and plot them.
}
\usage{
\method{plot}{listdtr}(x, stages = NULL, digits = 3L, \dots)
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
  \code{\link{draw.rule}}
}
\examples{
  # see examples for listdtr
}
\keyword{ tree }
