build.rule <- function(x, y, maxlen = 10L,
  zeta = 0.1 * mean(y), eta = 0.05 * sum(y))
{
  if (!is.matrix(x) || ncol(x) < 2) {
    x <- cbind(x = x, dummy_ = 0)
  }
  y <- as.matrix(y)
  stopifnot(ncol(y) > 1)
  stopifnot(nrow(x) == nrow(y))
  n <- nrow(x)

  zeta <- as.double(zeta)
  eta <- as.double(eta)
  maxlen <- as.integer(maxlen)
  action <- integer(n)

  variables <- colnames(x)
  if (is.null(variables)) {
    variables <- paste0("x", seq_len(ncol(x)))
  }
  options <- colnames(y)
  if (is.null(options)) {
    options <- as.character(seq_len(ncol(y)))
  }
  rule <- .Call("R_find_rule", x, y, zeta, eta, maxlen, action)
  action <- action + 1L

  object <- list(rule = rule, options = options, 
    variables = variables, action = action)
  object
}




build.rule.cv <- function(x, y, kfolds = 5L, fold = NULL,
  maxlen = 10L, zeta.choices = NULL, eta.choices = NULL, 
  cv.only = FALSE)
{
  if (!is.matrix(x) || ncol(x) < 2) {
    x <- cbind(x = x, dummy_ = 0)
  }
  y <- as.matrix(y)
  stopifnot(ncol(y) > 1)
  stopifnot(nrow(x) == nrow(y))
  n <- nrow(x)

  simple.loss <- min(colMeans(y))
  if (simple.loss < 1e-8) {
    zeta.selected <- simple.loss * n
    eta.selected <- simple.loss * n
  } else {
    if (is.null(zeta.choices) || is.null(eta.choices)) {
      zeta.grid <- simple.loss * c(2, 0.75, 0.3, 0.12, 0.05)
      eta.grid <- simple.loss * n * c(0.3, 0.1, 0.03)
      zeta.choices <- rep(zeta.grid, times = 3L)
      eta.choices <- rep(eta.grid, each = 5L)
    }
    if (is.null(fold)) {
      kfolds <- as.integer(kfolds)
      fold <- rep_len(1L : kfolds, n)[sample.int(n)]
    } else {
      fold <- as.integer(fold)
      kfolds <- max(fold)
      if (any(tabulate(fold, kfolds) <= 5L)) {
        stop("Some fold(s) have too few observations.")
      }
    }
    fold <- fold - 1L

    cv.loss <- .Call("R_cv_tune_rule", x, y,
      zeta.choices, eta.choices, maxlen, fold, kfolds)
    min.cv.loss <- min(cv.loss)
    
    if (min.cv.loss > simple.loss - 1e-8) {
      zeta.selected <- simple.loss * n
      eta.selected <- simple.loss * n
    } else {
      index <- which(cv.loss - min.cv.loss - 1e-8 <= 0)[1L]
      zeta.selected <- zeta.choices[index]
      eta.selected <- eta.choices[index]
    }
  }

  cv <- list(
    zeta.selected = zeta.selected, 
    eta.selected = eta.selected,
    metrics = data.frame(
      zeta.choices = zeta.choices,
      eta.choices = eta.choices,
      cv.loss = cv.loss))
  if (cv.only) {
    object <- list(cv = cv)
  } else {
    object <- build.rule(x, y, maxlen, zeta.selected, eta.selected)
    object$cv <- cv
  }
  
  object
}




apply.rule <- function(object, xnew, 
  what = c("label", "index"))
{
  what <- match.arg(what)
  if (!is.matrix(xnew) || ncol(xnew) < 2) {
    xnew <- cbind(x = xnew, dummy_ = 0)
  }

  action <- .Call("R_apply_rule", object$rule, xnew) + 1L
  if (what == "label") {
    action <- factor(object$options, object$options)[action]
  }
  
  action
}




verbalize.rule <- function(object, digits = 3L)
{
  options <- object$options
  variables <- object$variables
  
  d <- object$rule
  op1 <- ifelse(substring(d[, "type"], 1L, 1L) == "L", "<=", ">")
  op2 <- ifelse(substring(d[, "type"], 2L, 2L) == "L", "<=", ">")
  var1 <- variables[d[, "j1"]]
  var2 <- variables[d[, "j2"]]
  term1 <- ifelse(is.finite(d[, "c1"]), 
    paste(var1, op1, formatC(d[, "c1"], digits, 0L, "f")), "")
  term2 <- ifelse(is.finite(d[, "c2"]), 
    paste(var2, op2, formatC(d[, "c2"], digits, 0L, "f")), "")
  cond <- ifelse(nchar(term1) > 0L & nchar(term2) > 0L,
    paste(term1, "and", term2), paste0(term1, term2))
  act <- options[d[, "a"] + 1L]
  
  data.frame(conditions = cond, actions = act, 
    stringsAsFactors = FALSE)
}




show.rule <- function(object, digits = 3L)
{
  v <- verbalize.rule(object, digits)
  if (nrow(v) == 1L) {
    cat("Always ", v[1L, 2L], ".\n", sep = "")
    return(invisible(object))
  }
  
  clauses <- apply(v, 1, function(clause) {
    ifelse(nchar(clause[1L]) > 0L, 
      paste0("If ", clause[1L], " then ", clause[2L], ";"), 
      paste0("Else ", clause[2L], "."))
  })
  cat(clauses, sep = "\n")
  invisible(object)
}




draw.rule <- function(object, digits = 3L, filepath = NULL)
{
  v <- verbalize.rule(object, digits)
  conditions <- v$conditions
  actions <- v$actions

  # starts plot
  p <- ggplot() + theme_bw() +
    theme(line = element_blank(), rect = element_blank(),
      axis.text = element_blank(), axis.title = element_blank())
  scale <- 25.4  # 25.4 / 0.35277777777778
  
  len <- length(conditions) - 1L
  
  ratio <- 1.2
  bg.condition <- rgb(240, 230, 210, maxColorValue = 255)
  bg.action <- rgb(210, 230, 240, maxColorValue = 255)
  bg.arrow <- rgb(0, 100, 0, maxColorValue = 255)
  
  # computes sizes
  h <- max(strheight(c(" ", "T", "F"), "inches"),
    sapply(conditions, strheight, units = "inches"),
    sapply(actions, strheight, units = "inches")) * ratio
  w1 <- max(strheight(c(" ", "T", "F"), "inches"),
    sapply(conditions, strwidth, units = "inches")) * ratio
  w2 <- max(strheight(c(" ", "T", "F"), "inches"),
    sapply(actions, strwidth, units = "inches")) * ratio
  horigap <- strwidth(" T ", "inches") * ratio
  vertgap <- (strheight("F", "inches") 
    + 0.4 * strheight(" ", "inches")) * ratio
  line.width <- min(w1 * 0.03, w2 * 0.03, h * 0.1)
  
  h0 <- (h + vertgap) * (len + 1L)
  w0 <- w1 + horigap + w2
  p <- p + coord_cartesian(c(0, w0), c(0, h0))
  
  x1 <- rep.int(0, len)
  y1 <- h0 - (h + vertgap) * seq_len(len)
  x2 <- x1 + w1
  y2 <- y1 + h
  x3 <- (x1 + x2) / 2
  y3 <- (y1 + y2) / 2
  if (len) {
    df.condition <- data.frame(
      x1 = x1, y1 = y1, x2 = x2, y2 = y2, x3 = x3, y3 = y3,
      h = h, w1 = w1, w2 = w2, 
      horigap = horigap, vertgap = vertgap, 
      line.width = line.width)
  }

  x4 <- rep.int(w1 + horigap, len + 1L)
  y4 <- c(y1, 0)
  x5 <- x4 + w2
  y5 <- y4 + h
  x6 <- (x4 + x5) / 2
  y6 <- (y4 + y5) / 2
  df.action <- data.frame(
    x4 = x4, y4 = y4, x5 = x5, y5 = y5, x6 = x6, y6 = y6,
    h = h, w1 = w1, w2 = w2, 
    horigap = horigap, vertgap = vertgap)
  
  if (len) {
    # draws arrows
    p <- p + geom_segment(
      aes(x = x3, y = y1 + h + vertgap, xend = x3, yend = y1 + h), 
      df.condition,
      size = line.width * scale, col = bg.arrow,
      arrow = arrow(length = unit(vertgap / 3, "inches")))
  
    p <- p + geom_segment(
      aes(x = x2, y = y3, xend = x2 + horigap, yend = y3), 
      df.condition,
      size = line.width * scale, col = bg.arrow,
      arrow = arrow(length = unit(horigap / 3, "inches")))
    
    p <- p + geom_segment(
      aes(x = w1 / 2, y = h + vertgap, xend = w1 / 2, yend = h / 2),
      df.condition,
      size = line.width * scale, col = bg.arrow)
  
    p <- p + geom_segment(
      aes(x = w1 / 2, y = h / 2, xend = w1 + horigap, yend = h / 2),
      df.condition,
      size = line.width * scale, col = bg.arrow,
      arrow = arrow(length = unit(horigap / 3, "inches")))
    
    # draws true and false
    p <- p + geom_text(
      aes(x = w1 / 2 + h / 5, y = y1 - vertgap / 2, label = "F"),
      df.condition,
      col = bg.arrow,
      size = h * scale * 0.75)
  
    p <- p + geom_text(
      aes(x = x2 + horigap / 2, y = y3 + h / 5, label = "T"),
      df.condition,
      col = bg.arrow,
      size = h * scale * 0.75)
  
    # draws conditions
    p <- p + geom_rect(
      aes(xmin = x1, ymin = y1, xmax = x2, ymax = y2), 
      df.condition,
      fill = bg.condition)
    
    p <- p + geom_text(
      aes(x = x3, y = y3, label = conditions[seq_len(len)]), 
      df.condition,
      size = h * scale)
  } else {  # len == 0
    p <- p + geom_segment(
      aes(x = w1 / 2, y = h0, xend = w1 / 2, yend = y6),
      size = line.width * scale, col = bg.arrow)
    p <- p + geom_segment(
      aes(x = w1 / 2, y = y6, xend = x4, yend = y6),
      size = line.width * scale, col = bg.arrow,
      arrow = arrow(length = unit(horigap / 3, "inches")))
  }

  # draws actions
  p <- p + geom_rect(aes(xmin = x4, ymin = y4, xmax = x5, ymax = y5),
    df.action,
    fill = bg.action)
  
  p <- p + geom_text(aes(x = x6, y = y6, label = actions), 
    df.action, 
    size = h * scale)

  if (!is.null(filepath)) {
    ggsave(filepath, p, width = w0 * 2.54, height = h0 * 2.54)
  }
  p
}



