krr <- function(x, y, group = NULL)
{
  x <- as.matrix(x)
  if (is.null(group)) {
    options <- "0"
    group <- rep_len(0L, length(y))
  } else {
    if (!is.factor(group)) {
      group <- factor(group)
    }
    options <- levels(group)
    group <- as.integer(group) - 1L
  }
  
  var.x <- colMeans(x ^ 2L) - colMeans(x) ^ 2L
  var.x[var.x < 1e-8] <- Inf
  scaling <- 1.0 / var.x
  model <- .Call("R_kernel_train", x, y,
    group, length(options), scaling)
  
  object <- list(model = model, options = options)
  class(object) <- "krr"
  object
}




predict.krr <- function(object, xnew, ...)
{
  xnew <- as.matrix(xnew)
  ynew <- .Call("R_kernel_predict", object$model, xnew)
  colnames(ynew) <- object$options
  ynew
}




get.regrets <- function(outcomes)
{
  regrets <- .Call("R_get_regrets_from_outcomes", outcomes)
  colnames(regrets) <- colnames(outcomes)
  regrets
}



