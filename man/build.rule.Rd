\name{build.rule}
\alias{build.rule}
\alias{build.rule.cv}
\alias{apply.rule}
\alias{show.rule}
\alias{verbalize.rule}
\alias{draw.rule}
\title{
Low-level Functions for Handling List-based Rules
}
\description{
Build, apply and visualize list-based rules directly using features and losses.
}
\usage{
build.rule(x, y, 
  maxlen = 10L, zeta = 0.1 * mean(y), eta = 0.05 * sum(y))

build.rule.cv(x, y, 
  kfolds = 5L, fold = NULL, 
  maxlen = 10L, zeta.choices = NULL, eta.choices = NULL, 
  cv.only = FALSE)

apply.rule(object, xnew, what = c("label", "index"))

show.rule(object, digits = 3L)

verbalize.rule(object, digits = 3L)

draw.rule(object, digits = 3L, filepath = NULL)
}
\arguments{
  \item{x}{
    A matrix of features.
  }
  \item{y}{
    A matrix of losses; \code{y[i, j]} gives the loss if 
    the \code{i}-th observation receives the \code{j}-th treatment.
  }
  \item{maxlen}{
    A scalar for the maximum length of the list.
  }
  \item{zeta}{
    A scalar for tuning paramter zeta.
    Larger zeta tends to construct condition 
    that covers more observations in each if-then clause.
  }
  \item{eta}{
    A scalar for tuning paramter eta.
    Larger eta tends to construct condition 
    that uses less features in each if-then clause.
  }
  \item{kfolds}{
    A scalar for the number of folds for cross validation.
  }
  \item{fold}{
    An integer vector consisting of fold membership.
  }
  \item{zeta.choices}{
    A numeric vector for possible values of zeta
    in cross validation.
  }
  \item{eta.choices}{
    A numeric vector for possible values of eta
    in cross validation.
  }
  \item{cv.only}{
    A boolean scalar. If true, only cross validated losses are computed.
    Otherwise, the list built using the optimal zeta and eta is also
    computed.
  }
  \item{object}{
    Return value of \code{build.rule}, or \code{build.rule.cv} with
    \code{cv.only = FALSE}.
  }
  \item{xnew}{
    A matrix of features for prediction.
  }
  \item{what}{
    A scalar that determines the form in which the recommended treatment
    is represented.
  }
  \item{digits}{
    A scalar for the number of decimal digits to show.
  }
  \item{filepath}{
    A character scalar, if not null, that gives the location 
    that the diagram will save to.
  }
}
\details{
  See the reference if interested in the algorithm.
}
\value{
  \code{build.rule} returns a list.
  
  \code{build.rule.cv} returns a list as well as 
    cross validated losses.
  
  \code{apply.rule} returns a vector
    of recommended actions.
  
  \code{show.rule} prints the rule in words
    and returns it invisibly.
  
  \code{verbalize.rule} returns a data.frame
    that contains conditions and actions separately
    for each if-then clause.
  
  \code{draw.rule} returns a ggplot2 object
    that contains the diagram.
}
\references{
  \url{https://arxiv.org/abs/1606.01472}
}
\note{
  Use these functions only when it is really necessary. 
}
\seealso{
  \code{\link{listdtr}}
}
\examples{
  x <- matrix(rnorm(200 * 10), 200, 10)
  y <- cbind(
    a1 = as.double(x[, 1] < 0) + rnorm(200, 0, 0.1),
    a2 = as.double(x[, 2] > 0) + rnorm(200, 0, 0.1))
  y[y < 0] <- 0
  
  obj <- build.rule(x, y)
  show.rule(obj)
  draw.rule(obj)
  
  xnew <- matrix(rnorm(1000 * 10), 1000, 10)
  ynew <- apply.rule(obj, xnew)
  table(factor(xnew[, 1] < 0) : factor(xnew[, 2] < 0), ynew)
}
\keyword{ tree }
