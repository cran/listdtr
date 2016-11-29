\name{listdtr}
\alias{listdtr}
\title{
Estimation of List-based Optimal Dynamic Treatment Regime
}
\description{
Estimate the optimal dynamic treatment regime 
/ individualized treatment rule, in the form
of decision list, namely, a sequence of if-then clauses.
}
\usage{
listdtr(y, a, x, stage.x, 
  seed = NULL, kfolds = 5L, fold = NULL, 
  maxlen = 10L, zeta.choices = NULL, eta.choices = NULL)
}
\arguments{
  \item{y}{
    a matrix of immediate outcomes, 
    of size \code{n.obs} by \code{n.stage},
    where \code{n.obs} is the number of observations
    and \code{n.stage} is the number of stages.
    Assume larger outcomes are more favorable.
  }
  \item{a}{
    a matrix of treatments/interventions actually received
    at each stage, of size \code{n.obs} by \code{n.stage}.
  }
  \item{x}{
    a matrix of features (such as demographics, biomarkers, 
    confounders), of size \code{n.obs} by \code{n.feature},
    where \code{n.feature} is the number of features
    measured at any of the stages.
  }
  \item{stage.x}{
    a vector of length \code{n.feature},
    with values in 1, ..., \code{n.stage} that gives
    the stage at which each feature is measured.
  }
  \item{seed}{
    seed for random number generator to obtain \code{fold}.
    Omitted if \code{fold} is not null.
  }
  \item{kfolds}{
    number of folds to perform cross validation.
  }
  \item{fold}{
    a vector of length \code{n.obs}
    that specifies fold membership for each observation.
  }
  \item{maxlen}{
    maximum length of the decision list in each stage.
    Should be a scalar.
  }
  \item{zeta.choices}{
    Choices for the tuning parameter zeta.
    Larger value of zeta tends to construct a condition
    that covers more observations in each if-then clause.
    Should be null or a numeric vector.
  }
  \item{eta.choices}{
    Choices for the tuning parameter eta.
    Larger value of eta tends to construct a condition
    that uses less features in each if-then clause.
    Should be null or a numeric vector.
  }
}
\details{
  The algorithm is quite complicated. See the reference if interested.
}
\value{
  An object of class \code{listdtr}.
}
\references{
  \url{https://arxiv.org/abs/1606.01472}
}
\seealso{
  \code{\link{predict.listdtr}}, 
  \code{\link{print.listdtr}}, 
  \code{\link{plot.listdtr}},
  \code{\link{build.rule.cv}}
}
\examples{
  
  # an example for one-stage study
  x <- matrix(rnorm(500), 100, 5)
  stage.x <- rep(1, 5)
  a <- rbinom(100, 1, 0.5)
  y <- a * x[, 1] + rnorm(100, 0, 0.1)
  dtr <- listdtr(y, a, x, stage.x)
  
  dtr                # display the regime in words
  plot(dtr)          # display the regime in diagrams
  yrec <- predict(dtr, x, 1)
  yopt <- ifelse(x[, 1] > 0, 1, 0)
  table(yrec, yopt)  # discrepancy between recommended and optimal

  # an example for two-stage study
  x <- matrix(rnorm(500), 100, 5)
  stage.x <- c(1, 1, 1, 2, 2)
  a1 <- rbinom(100, 1, 0.5)
  a2 <- rbinom(100, 1, 0.5)
  y1 <- rep(0, 100)
  y2 <- 9 * a1 * sin(x[, 1] * pi / 8) - 2 * a2 * x[, 4] + rnorm(100)
  dtr <- listdtr(cbind(y1, y2), cbind(a1, a2), x, stage.x)
  
  dtr                # display the regime in words
  plot(dtr)          # display the regime in diagrams
  yrec <- predict(dtr, x, 1)
  yopt <- ifelse(x[, 1] > 0, 1, 0)
  table(yrec, yopt)  # discrepancy between recommended and optimal

}
\keyword{ tree }
