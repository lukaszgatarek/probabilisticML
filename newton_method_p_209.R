require("MASS")

sigmoid <- function(w, x){
  
  # evaluates the standard sigmoid function in the engine of the logistic regression

  # w vector of params
  # x vector of observations
  # y single observation of dependent
  
  sigmoid_evaluation <- 1 / (1 + exp(-1 * t(w) %*% x ) )
  
  return (sigmoid_evaluation)
  
}

gradient <- function(X, mu, y){
  
  # mu corresponds to the vector of sigmoid evaluations for X
  
  return(t(X) %*% (mu - y))
  
}

Hessian <- function(X, mu){
  
  S <- diag(mu * (1-mu))
  # define S as in formula 8.7 page 2471
  return(t(X) %*% S %*% X)
  
}


# ----------- simulate the model

# number of observations 

N = 1000
# simulate x's from normal
mu <- 0
Sigma <- 1

# simulate from normal / that can be changed to any other distribution
X <- mvrnorm(n = N, mu, Sigma)

# select w
w <- matrix(0.7)

# compute mu i.e. vector of sigmoid evaluations 
# for each observation X under the true value of w
for (i in 1:N){
  mu[i] <- sigmoid(w, X[i, ])
}

# derive y from mu based on a fixed threshold
y <- mu > 0.5

# change 0 to -1
y[y == 0] <- -1

# ------------ Newton's method
# maximal number of steps fro the algorithm
maxNumSteps <- 20
# prepare the holder for the subsequent estimates of w and initialize w
estimated_w <- array(NA, dim = c(dim(w)[1], dim(w)[2], maxNumSteps )   )
estimated_w[1,1,1] <- 1

# record gradients 
g <- array(NA, dim = c(dim(w)[1], dim(w)[2], maxNumSteps )   )
# record Hessians
H <- array(NA, dim = c(dim(w)[1], dim(w)[1], maxNumSteps )   )
# record directions
d <- array(NA, dim = c(dim(w)[1], dim(w)[2], maxNumSteps )   )

NLL <- rep(NA, maxNumSteps)

for (i in 1: maxNumSteps){

  # evaluate gradient at the current value of w
  current_mu_given_w <- rep(NA, N)
  
  for (j in 1:N){
    #print(sigmoid(estimated_w[,,i], X[j,]))
    current_mu_given_w[j] <- sigmoid(estimated_w[,,i], y[j] * X[j,])
  }
  
  g[,,i] <- gradient(X, current_mu_given_w, y)
  H[,,i] <- Hessian(X, current_mu_given_w)
  
  d[,,i] <- H[,,i]^(-1) %*% -g[,,i]
  
  nu <- 0.001
  estimated_w[,,i+1] <- estimated_w[,,i] + nu * d[,,i]
  
  
  NLL_contribution <- rep(NA, N)
  for (j in 1:N){
    #print(sigmoid(estimated_w[,,i], X[j,]))
    NLL_contribution[j] <- log(1 + exp(-y[j] * t( estimated_w[,,i]) %*% X[j,]  )   )
  }
  NLL[i] <- sum(NLL_contribution)
  print(NLL[i])
}


plot(estimated_w)



