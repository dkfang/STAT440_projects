# ExtraTree Classifier
##REF:https://rdrr.io/cran/extraTrees/man/extraTrees.html
library(extraTrees)
set.seed(2016)
#######################################
## Regression with ExtraTrees:
n <- 1000  ## number of samples
p <- 5     ## number of dimensions
x <- matrix(runif(n*p), n, p)
y <- (x[,1]>0.5) + 0.8*(x[,2]>0.6) + 0.5*(x[,3]>0.4) +
  0.1*runif(nrow(x))
## using ExtraTrees
et <- extraTrees(x, y, nodesize=3, mtry=p, numRandomCuts=2)
yhat <- predict(et, x)
head(y,10)
head(yhat,10)
#######################################

#######################################
## Multi-task regression with ExtraTrees:
n <- 1000  ## number of samples
p <- 5     ## number of dimensions
x <- matrix(runif(n*p), n, p)
task <- sample(1:10, size=n, replace=TRUE)
## y depends on the task: 
y <- 0.5*(x[,1]>0.5) + 0.6*(x[,2]>0.6) + 0.8*(x[cbind(1:n,(task %% 2) + 3)]>0.4)
et <- extraTrees(x, y, nodesize=3, mtry=p-1, numRandomCuts=2, tasks=task)
yhat <- predict(et, x, newtasks=task)
#######################################

#######################################
## Quantile regression with ExtraTrees (with test data)
make.qdata <- function(n) {
  p <- 4
  f <- function(x) (x[,1]>0.5) + 0.8*(x[,2]>0.6) + 0.5*(x[,3]>0.4)
  x <- matrix(runif(n*p), n, p)
  y <- as.numeric(f(x))
  return(list(x=x, y=y))
}
train <- make.qdata(400)
test  <- make.qdata(200)

## learning extra trees:
et <- extraTrees(train$x, train$y, quantile=TRUE)
## estimate median (0.5 quantile)
yhat0.5 <- predict(et, test$x, quantile = 0.5)
## estimate 0.8 quantile (80%)
yhat0.8 <- predict(et, test$x, quantile = 0.8)
######################################

