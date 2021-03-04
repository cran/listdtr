listdtr <- function(y, a, x, stage.x, seed = NULL,
  kfolds = 5L, fold = NULL,
  maxlen = 10L, zeta.choices = NULL, eta.choices = NULL)
{
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  if (!is.data.frame(a)) {
    a <- as.data.frame(a)
  }
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  if (is.null(colnames(x))) {
    colnames(x) <- paste0("x", seq_len(ncol(x)))
  }
  stopifnot(nrow(y) == nrow(a) && nrow(y) == nrow(x))
  n <- nrow(y)
  stopifnot(ncol(y) == ncol(a))
  n.stage <- ncol(y)
  dtr <- vector("list", n.stage)
  future.y <- double(n)

  if (is.null(colnames(a))) {
    colnames(a) <- paste0("a", 1L : n.stage)
  }
  a.mm <- lapply(1L : n.stage, function(j)
    model.matrix(as.formula(sprintf("~ -1 + %s", colnames(a)[j])), a))
  stage.a.mm <- rep.int(1L : n.stage, sapply(a.mm, ncol))
  a.mm <- do.call("cbind", a.mm)

  if (is.null(fold)) {
    if (!is.null(seed)) {
      set.seed(seed)
    }
    fold <- rep_len(1L : kfolds, n)[sample.int(n)]
  }

  for (i.stage in n.stage : 1L) {
    current.x <- cbind(
      x[, which(stage.x <= i.stage), drop = FALSE],
      a.mm[, which(stage.a.mm < i.stage), drop = FALSE],
      y[, seq_len(i.stage - 1L), drop = FALSE])
    if (ncol(current.x) < 2L) {
      current.x <- cbind(x = current.x, dummy_ = 0.0)
    }
    current.a <- a[, i.stage]
    current.y <- y[, i.stage] + future.y

    model <- krr(current.x, current.y, current.a)
    options <- model$options
    outcomes <- predict(model, current.x)
    regrets <- get.regrets(outcomes)

    obj <- build.rule.cv(current.x, regrets,
      kfolds, fold, maxlen, zeta.choices, eta.choices)
    dtr[[i.stage]] <- obj
    future.y <- outcomes[cbind(1L : n, obj$action)]
  }

  class(dtr) <- "listdtr"
  dtr
}




predict.listdtr <- function(object, xnew, stage, ...)
{
  stopifnot(stage >= 1L && stage <= length(object))
  if (!is.matrix(xnew) || ncol(xnew) < 2L) {
    xnew <- cbind(x = xnew, dummy_ = 0.0)
  }

  action <- apply.rule(object[[stage]], xnew, "label")
  action
}




print.listdtr <- function(x, stages = NULL, digits = 3L, ...)
{
  object <- x
  if (is.null(stages)) {
    stages <- seq_along(object)
  }
  for (i.stage in stages) {
    cat(sprintf("=====  Stage %d  =====\n", i.stage))
    show.rule(object[[i.stage]], digits)
  }
  invisible(object)
}




plot.listdtr <- function(x, stages = NULL, digits = 3L, ...)
{
  object <- x
  if (is.null(stages)) {
    stages <- seq_along(object)
  }
  figures <- lapply(stages, function(i.stage)
    draw.rule(object[[i.stage]], digits))
  if (length(stages) <= 1L) {
    print(figures[[1L]] + ggtitle("Stage 1"))
  } else {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(1L, length(stages))))

    for (i in seq_len(length(stages))) {
      print(figures[[i]] + ggtitle(sprintf("Stage %d", stages[i])),
        vp = viewport(layout.pos.row = 1L, layout.pos.col = i))
    }
  }
  invisible(object)
}
