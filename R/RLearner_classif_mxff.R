#' @export
makeRLearner.classif.mxff = function() {
  makeRLearnerClassif(
    cl = "classif.mxff",
    package = "mxnet",
    par.set = makeParamSet(
      # architectural hyperparameters
      makeIntegerLearnerParam(id = "layers", lower = 1L, upper = 4L, default = 1L),
      makeIntegerLearnerParam(id = "num.layer1", lower = 1L, default = 1L),
      makeIntegerLearnerParam(id = "num.layer2", lower = 1L, default = 1L,
        requires = quote(layers > 1)),
      makeIntegerLearnerParam(id = "num.layer3", lower = 1L, default = 1L,
        requires = quote(layers > 2)),
      makeIntegerLearnerParam(id = "num.layer4", lower = 1L, default = 1L,
        requires = quote(layers > 3)),
      makeDiscreteLearnerParam(id = "act1", default = "tanh",
        values = c("tanh", "relu", "sigmoid", "softrelu")),
      makeDiscreteLearnerParam(id = "act2", default = "tanh",
        values = c("tanh", "relu", "sigmoid", "softrelu"),
        requires = quote(layers > 1)),
      makeDiscreteLearnerParam(id = "act3", default = "tanh",
        values = c("tanh", "relu", "sigmoid", "softrelu"),
        requires = quote(layers > 2)),
      makeDiscreteLearnerParam(id = "act4", default = "tanh",
        values = c("tanh", "relu", "sigmoid", "softrelu"),
        requires = quote(layers > 3)),
      makeDiscreteLearnerParam(id = "act.out", default = "softmax",
        values = c("rmse", "softmax", "logistic")),
      makeLogicalLearnerParam(id = "conv.layer1", default = FALSE),
      makeLogicalLearnerParam(id = "conv.layer2", default = FALSE,
        requires = quote(layers > 1 && conv.layer1 == TRUE)),
      makeLogicalLearnerParam(id = "conv.layer3", default = FALSE,
        requires = quote(layers > 2 && conv.layer2 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.data.shape",
        requires = quote(conv.layer1 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.kernel1",
        requires = quote(conv.layer1 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.kernel2",
        requires = quote(conv.layer2 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.kernel3",
        requires = quote(conv.layer3 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.stride1",
        requires = quote(conv.layer1 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.stride2",
        requires = quote(conv.layer2 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.stride3",
        requires = quote(conv.layer3 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.dilate1",
        requires = quote(conv.layer1 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.dilate2",
        requires = quote(conv.layer2 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.dilate3",
        requires = quote(conv.layer3 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.pad1",
        requires = quote(conv.layer1 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.pad2",
        requires = quote(conv.layer2 == TRUE)),
      makeIntegerVectorLearnerParam(id = "conv.pad3",
        requires = quote(conv.layer3 == TRUE)),
      makeIntegerVectorLearnerParam(id = "pool.kernel1",
        requires = quote(conv.layer1 == TRUE)),
      makeIntegerVectorLearnerParam(id = "pool.kernel2",
        requires = quote(conv.layer2 == TRUE)),
      makeIntegerVectorLearnerParam(id = "pool.kernel3",
        requires = quote(conv.layer3 == TRUE)),
      makeIntegerVectorLearnerParam(id = "pool.stride1",
        requires = quote(conv.layer1 == TRUE)),
      makeIntegerVectorLearnerParam(id = "pool.stride2",
        requires = quote(conv.layer2 == TRUE)),
      makeIntegerVectorLearnerParam(id = "pool.stride3",
        requires = quote(conv.layer3 == TRUE)),
      makeIntegerVectorLearnerParam(id = "pool.pad1",
        requires = quote(conv.layer1 == TRUE)),
      makeIntegerVectorLearnerParam(id = "pool.pad2",
        requires = quote(conv.layer2 == TRUE)),
      makeIntegerVectorLearnerParam(id = "pool.pad3",
        requires = quote(conv.layer3 == TRUE)),
      makeDiscreteLearnerParam(id = "pool.type1", default = "max",
        values = c("max", "avg", "sum"),
        requires = quote(conv.layer1 == TRUE)),
      makeDiscreteLearnerParam(id = "pool.type2", default = "max",
        values = c("max", "avg", "sum"),
        requires = quote(conv.layer2 == TRUE)),
      makeDiscreteLearnerParam(id = "pool.type3", default = "max",
        values = c("max", "avg", "sum"),
        requires = quote(conv.layer3 == TRUE)),
      # other hyperparameters
      makeIntegerVectorLearnerParam(id = "validation.set"),
      makeIntegerLearnerParam(id = "early.stop.badsteps", lower = 1),
      makeLogicalLearnerParam(id = "early.stop.maximize", default = TRUE),
      makeNumericLearnerParam(id = "dropout", lower = 0, upper = 1 - 1e-7),
      makeUntypedLearnerParam(id = "ctx", default = mx.ctx.default(), tunable = FALSE),
      makeIntegerLearnerParam(id = "begin.round", default = 1L),
      makeIntegerLearnerParam(id = "num.round", default = 10L),
      makeDiscreteLearnerParam(id = "optimizer", default = "sgd",
        values = c("sgd", "rmsprop", "adam", "adagrad", "adadelta")),
      makeUntypedLearnerParam(id = "initializer", default = NULL),
      makeUntypedLearnerParam(id = "eval.data", default = NULL, tunable = FALSE),
      makeUntypedLearnerParam(id = "eval.metric", default = NULL, tunable = FALSE),
      makeUntypedLearnerParam(id = "epoch.end.callback", default = NULL, tunable = FALSE),
      makeUntypedLearnerParam(id = "batch.end.callback", default = NULL, tunable = FALSE),
      makeIntegerLearnerParam(id = "array.batch.size", default = 128L),
      makeDiscreteLearnerParam(id = "array.layout", default = "rowmajor",
        values = c("auto", "colmajor", "rowmajor"), tunable = FALSE),
      makeUntypedLearnerParam(id = "kvstore", default = "local", tunable = FALSE),
      makeLogicalLearnerParam(id = "verbose", default = FALSE, tunable = FALSE),
      makeUntypedLearnerParam(id = "arg.params", tunable = FALSE),
      makeUntypedLearnerParam(id = "aux.params", tunable = FALSE),
      makeUntypedLearnerParam(id = "symbol", tunable = FALSE),
      # optimizer specific hyperhyperparameters
      makeNumericLearnerParam(id = "rho", default = 0.9, requires = quote(optimizer == "adadelta")),
      makeNumericLearnerParam(id = "epsilon",
        requires = quote(optimizer %in% c("adadelta", "adagrad", "adam"))),
      makeNumericLearnerParam(id = "wd", default = 0,
        requires = quote(optimizer %in% c("adadelta", "adagrad", "adam", "rmsprop", "sgd"))),
      makeNumericLearnerParam(id = "rescale.grad", default = 1,
        requires = quote(optimizer %in% c("adadelta", "adagrad", "adam", "rmsprop", "sgd"))),
      makeNumericLearnerParam(id = "clip_gradient",
        requires = quote(optimizer %in% c("adadelta", "adagrad", "adam", "rmsprop", "sgd"))),
      makeFunctionLearnerParam(id = "lr_scheduler",
        requires = quote(optimizer %in% c("adagrad", "adam", "rmsprop", "sgd"))),
      makeNumericLearnerParam(id = "learning.rate",
        requires = quote(optimizer %in% c("adagrad", "adam", "rmsprop", "sgd"))),
      makeNumericLearnerParam(id = "beta1", default = 0.9, requires = quote(optimizer == "adam")),
      makeNumericLearnerParam(id = "beta2", default = 0.999, requires = quote(optimizer == "adam")),
      makeNumericLearnerParam(id = "gamma1", default = 0.95,
        requires = quote(optimizer == "rmsprop")),
      makeNumericLearnerParam(id = "gamma2", default = 0.9,
        requires = quote(optimizer == "rmsprop")),
      makeNumericLearnerParam(id = "momentum", default = 0, requires = quote(optimizer == "sgd"))
    ),
    properties = c("twoclass", "multiclass", "numerics", "prob"),
    par.vals = list(learning.rate = 0.1, array.layout = "rowmajor", verbose = FALSE),
    name = "Feedforward Neural Network",
    short.name = "mxff",
    note = "Default of `learning.rate` set to `0.1`. Default of `array.layout` set to `'rowmajor'`.
    Default of `verbose` is set to `FALSE`. If `symbol` is specified, it will be passed to mxnet
    ignoring other architectural specifications. Default of `initializer` is set to NULL, which
    results in the default mxnet initializer being called when training a model. Number of output
    nodes is detected automatically. The upper bound for dropout is set to `1 - 1e-7` as in `mx.mlp`
    in the `mxnet` package. `validation.set` gives the indices of training data that will not
    be used for training but as validation data similar to the data provided in `eval.data`.
    If `eval.data` is specified, `validation.set` will be ignored. 
    If `early.stop.badsteps` is specified and `epoch.end.callback` is not specified,
    early stopping will be used using `mx.callback.early.stop` as `epoch.end.callback` with the
    learner's `eval.metric`. In this case, `early.stop.badsteps` gives the number of `bad.steps` in
    `mx.callback.early.stop` and `early.stop.maximize` gives the `maximize` parameter in 
    `mx.callback.early.stop`. Please note that when using `early.stop.badsteps`, `eval.metric` and
    either `eval.data` or `validation.set` should be specified.
    "
  )
}

#' @export
trainLearner.classif.mxff = function(.learner, .task, .subset, .weights = NULL,
  layers = 1L, num.layer1 = 1L, num.layer2 = 1L, num.layer3 = 1L, num.layer4 = 1L,
  act1 = "tanh", act2 = "tanh", act3 = "tanh", act4= "tanh", act.out = "softmax",
  conv.data.shape = NULL, conv.layer1 = FALSE, conv.layer2 = FALSE, conv.layer3 = FALSE,
  conv.kernel1 = NULL, conv.kernel2 = NULL, conv.kernel3 = NULL,
  conv.stride1 = NULL, conv.stride2 = NULL, conv.stride3 = NULL,
  conv.dilate1 = NULL, conv.dilate2 = NULL, conv.dilate3 = NULL,
  conv.pad1 = NULL, conv.pad2 = NULL, conv.pad3 = NULL,
  pool.kernel1 = NULL, pool.kernel2 = NULL, pool.kernel3 = NULL,
  pool.stride1 = NULL, pool.stride2 = NULL, pool.stride3 = NULL,
  pool.pad1 = NULL, pool.pad2 = NULL, pool.pad3 = NULL,
  pool.type1 = "max", pool.type2 = "max", pool.type3 = "max",
  dropout = NULL, symbol = NULL, validation.set = NULL, eval.data = NULL, early.stop.badsteps = NULL,
  epoch.end.callback = NULL, early.stop.maximize = TRUE, array.layout = "rowmajor", ...) {
  # transform data in correct format
  d = getTaskData(.task, subset = .subset, target.extra = TRUE)
  y = as.numeric(d$target) - 1
  X = data.matrix(d$data)
  
  # construct validation data
  if (is.null(eval.data) & !is.null(validation.set)) {
    eval.data = list()
    eval.data$label = y[validation.set,]
    y = y[-validation.set]
    eval.data$data = X[validation.set,]
    X = X[-validation.set,]
  }
  
  # if convolution is used, prepare the data dimensionality
  if (conv.layer1) {
    l = length(conv.data.shape)
    if (!(l %in% 1:4)) {
      stop("Length of conv.data.shape should be between 1 and 4!")
    }
    dims = switch(l,
      c(conv.data.shape, 1, 1, nrow(X)),
      c(conv.data.shape, 1, nrow(X)),
      c(conv.data.shape, nrow(X)),
      conv.data.shape)
    X = array(aperm(X), dim = dims)
    # adapt array.layout for mx.model.FeedForward.create
    array.layout = "colmajor"
    # adapt validation data if necessary
    if (!is.null(validation.set)) {
      eval.data$data = array(aperm(eval.data$data), dim = dims)
    }
  }
  
  # early stopping
  if (is.null(epoch.end.callback) & is.numeric(early.stop.badsteps)) {
    epoch.end.callback = mx.callback.early.stop(bad.steps = early.stop.badsteps,
      maximize = early.stop.maximize)
  }
  
  # construct vectors with #nodes and activations
  if (!is.null(symbol)) {
    out = symbol
  } else {
    sym = mx.symbol.Variable("data")
    act = c(act1, act2, act3, act4)[1:layers]
    nums = c(num.layer1, num.layer2, num.layer3, num.layer4)[1:layers]
    convs = c(conv.layer1, conv.layer2, conv.layer3, FALSE)[1:layers]
    # if layers equals 4 a NULL is appended to the lists automatically
    conv.kernels = list(conv.kernel1, conv.kernel2, conv.kernel3)[1:layers]
    conv.strides = list(conv.stride1, conv.stride2, conv.stride3)[1:layers]
    conv.dilates = list(conv.dilate1, conv.dilate2, conv.dilate3)[1:layers]
    conv.pads = list(conv.pad1, conv.pad2, conv.pad3)[1:layers]
    pool.kernels = list(pool.kernel1, pool.kernel2, pool.kernel3)[1:layers]
    pool.strides = list(pool.stride1, pool.stride2, pool.stride3)[1:layers]
    pool.pads = list(pool.pad1, pool.pad2, pool.pad3)[1:layers]
    pool.types = list(pool.type1, pool.type2, pool.type3)[1:layers]
    
    # construct hidden layers using symbols
    for (i in seq_len(layers)) {
      if (convs[i]) {
        # construct convolutional layer with pooling
        conv.inputs = list(data = sym, kernel = conv.kernels[[i]], stride = conv.strides[[i]],
          dilate = conv.dilates[[i]], pad = conv.pads[[i]], num_filter = nums[i])
        sym = do.call(mx.symbol.Convolution, conv.inputs[!sapply(conv.inputs, is.null)])
        sym = mx.symbol.Activation(sym, act_type = act[i])
        pool.inputs = list(data = sym, kernel = pool.kernels[[i]], pool.type = pool.types[[i]],
          stride = pool.strides[[i]], pad = pool.pads[[i]])
        sym = do.call(mx.symbol.Pooling, pool.inputs[!sapply(pool.inputs, is.null)])
      } else {
        # construct fully connected layer
        if (i > 1) {
          if (convs[i - 1]) {
            sym = mx.symbol.flatten(sym)
          }
        }
        sym = mx.symbol.FullyConnected(sym, num_hidden = nums[i])
        sym = mx.symbol.Activation(sym, act_type = act[i])
      }
    }
    
    # add dropout if specified
    if (!is.null(dropout)) {
      sym = mx.symbol.Dropout(sym, p = dropout)
    }
    
    # construct output layer
    nodes.out = switch(act.out,
      softmax = nlevels(d$target),
      logistic = 1,
      stop("Output activation not supported yet."))
    sym = mx.symbol.FullyConnected(sym, num_hidden = nodes.out)
    out = switch(act.out,
      # rmse = mx.symbol.LinearRegressionOutput(sym),
      softmax = mx.symbol.SoftmaxOutput(sym),
      logistic = mx.symbol.LogisticRegressionOutput(sym),
      stop("Output activation not supported yet."))
  }
  
  # create model
  model = mx.model.FeedForward.create(out, X = X, y = y, eval.data = eval.data,
    epoch.end.callback = epoch.end.callback, array.layout = array.layout, ...)
  return(model)
}

#' @export
predictLearner.classif.mxff = function(.learner, .model, .newdata, ...) {
  X = data.matrix(.newdata)
  array.layout = .model$learner$par.vals$array.layout
  if (.learner$par.vals$conv.layer1) {
    l = length(.learner$par.vals$conv.data.shape)
    dims = switch(l,
      c(.learner$par.vals$conv.data.shape, 1, 1, nrow(X)),
      c(.learner$par.vals$conv.data.shape, 1, nrow(X)),
      c(.learner$par.vals$conv.data.shape, nrow(X)),
      .learner$par.vals$conv.data.shape)
    X = array(aperm(X), dim = dims)
    array.layout = "colmajor"
  }
  p = predict(.model$learner.model, X = X, array.layout = array.layout)
  if (.learner$predict.type == "response") {
    p = apply(p, 2, function(i) {
      w = which.max(i)
      return(ifelse(length(w > 0), w, NaN))
    })
    p = factor(p, exclude = NaN)
    levels(p) = .model$task.desc$class.levels
    return(p)
  }
  if (.learner$predict.type == "prob") {
    p = t(p)
    colnames(p) = .model$task.desc$class.levels
    return(p)
  }
}

