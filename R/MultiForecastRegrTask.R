#' Create an task object for multivariate forecasting.
#'
#' @rdname Task
#' @description Creates a task that is used for multivariate forecasting. Target the name of a single series
#' or 'All'. Targeting a single series will produce a single output similar to univariate tasks while
#' treating all other variables endogeneously. targeting 'All' will forecast all variables forward.
#' @export
makeMultiForecastRegrTask = function(id = deparse(substitute(data)), data, target,
  weights = NULL, blocking = NULL, date.col = "dates", frequency = 1L, fixup.data = "warn",
  check.data = TRUE) {

  assertString(target)
  assertChoice(target, c("all", colnames(data)))

  assertString(id)
  assertClass(data, "data.frame")
  assertString(date.col)
  frequency = asCount(frequency)
  assertChoice(fixup.data, choices = c("no", "quiet", "warn"))
  assertFlag(check.data)
  is.target.all = target == "all"
  col.names = colnames(data)
  # Need to check that dates
  # 1. Exist
  # 2. Are unique
  # 3. Follow POSIXct convention
  dates = data[, date.col, drop = FALSE]
  if (check.data) {
    if (!is.target.all) {
      assertNumeric(data[[target]], any.missing = FALSE, finite = TRUE, .var.name = target)
    } else if (is.target.all) {
      lapply(data, assertNumeric, any.missing = FALSE, finite = TRUE)
    }
    if (any(duplicated(dates)))
      stop(catf("Multiple observations for %s. Dates must be unique.", dates[any(duplicated(dates)), ]))
    if (!is.POSIXt(dates[, 1]))
      stop(catf("Dates are of type %s, but must be in a POSIXt format", class(dates[, 1])))
  }
  if (fixup.data != "no" && all(!is.target.all)) {
    if (is.integer(data[[target]]))
      data[[target]] = as.double(data[[target]])
    if (is.unsorted(dates[, 1])) {
      if (fixup.data == "warn")
        warning("Dates and data will be sorted in ascending order")
      date.order = order(dates)
      data = data[date.order, , drop = FALSE]
      dates = dates[date.order, , drop = FALSE]
    }
  } else if (fixup.data != "no") {
    is.int = vlapply(data, is.integer)
    if (all(is.int)) {
      data = lapply(data, as.double)
      data = as.data.frame(row.names = row.names, data)
    }
  }
  # Remove the date column and add it as the rownames
  data = data[, date.col != colnames(data), drop = FALSE]

  if (is.target.all)
    target = colnames(data)

  task = makeSupervisedTask("mfcregr", data, target, weights, blocking, fixup.data = fixup.data, check.data = check.data)
  task$task.desc = makeMultiForecastRegrTaskDesc(id, data, target, weights, blocking, frequency, dates)
  addClasses(task, c("MultiForecastRegrTask", "TimeTask"))
}

makeMultiForecastRegrTaskDesc = function(id, data, target, weights, blocking, frequency, dates) {
  td = makeTaskDescInternal("mfcregr", id, data, target, weights, blocking)
  td$dates = dates
  td$frequency = frequency
  td$col.names = colnames(data)
  addClasses(td, c("MultiForecastRegrTaskDesc", "SupervisedTaskDesc"))
}


#' @export
print.MultiForecastRegrTask = function(x, print.weights = TRUE, ...) {
  td = getTaskDesc(x)
  catf("Task: %s", td$id)
  catf("Type: %s", td$type)
  catf("Target: %s", stri_paste(td$target, collapse = " "))
  catf("Observations: %i", td$size)
  catf("Dates:\n Start: %s \n End:   %s", td$dates[1, ], td$dates[nrow(td$dates), ])
  catf("Frequency: %i", td$frequency)
  catf("Features:")
  catf(printToChar(td$n.feat, collapse = "\n"))
  catf("Missings: %s", td$has.missings)
  if (print.weights)
    catf("Has weights: %s", td$has.weights)
  catf("Has blocking: %s", td$has.blocking)
}

