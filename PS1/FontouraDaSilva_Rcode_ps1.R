"

NUMERICAL METHODS 1/2023
João Pedro Fontoura da Silva

~* R version
"

ρ = 0.95    # persistence of AR process
σ = 0.007   # std. deviation
N = 9       # baseline number of grid points
m = 3       # number of std. devs
μ = 0       # mean

set.seed(123)

t = 10000   # number of 'observations'
obs = 1:t

ϵ = rnorm(t, μ, σ) # compiling the noise vector

"FUNCTIONS"

AR1_series = function(ρ, t){
  y = rep(0, t)    # filling a vector of length n with zeros
  for (i in 1:(t-1)){
    y[i+1] = ρ*y[i] + ϵ[i+1]
  }
  return(y)
}

Tauchen_discretization = function(ρ, μ, σ, m, N, t){
  θ_N = m * σ / sqrt(1 - ρ^2)
  θ_1 = - θ_N
  zgrid = seq(from=θ_1, to=θ_N, length.out = N)
  Δθ = (θ_N - θ_1) / (N - 1)
  
  # Filling out probability matrix P
  P = matrix(0, N, N) # NOTE: always fill with 0.0, if you use '0' code will interpret as integers-only
  for (i in 1:N){
    for (j in 1:N){
      if (j == 1){
        P[i,j] = pnorm((zgrid[j] + Δθ/2 - (1 - ρ) * μ - ρ * zgrid[i]), mean=μ, sd=σ)
      } else if (j == N){
        P[i,j] = 1 - pnorm((zgrid[j] - Δθ/2 - (1 - ρ) * μ - ρ * zgrid[i]), mean=μ, sd=σ)
      } else {
        P[i,j] = pnorm((zgrid[j] + Δθ/2 - (1 - ρ) * μ - ρ * zgrid[i]), mean=μ, sd=σ) - pnorm((zgrid[j] - Δθ/2 - (1 - ρ) * μ - ρ * zgrid[i]), mean=μ, sd=σ)
      }
    }
  }
  
  "Having the raw values of P, now we do the cummulative sum of columns so that in
  the probabilities add up to 1 as we move further along right. =#
  "
  for (j in 2:N){
    P[,j] = P[,j] + P[,j-1]
  }
  
  "
  (2)
  Having this probability matrix, now I want to discretize the noise process.
  I'll do this by picking the θ_i from the grid which is closest to wherever
    it is that I am in the sequence.

  (3)
    Afterwards, we compare the pnorm of the realized epsilon to the pnorm of each entry
    from the selected vector and pick the j-th value which is closest to it. Thus,
    we´ll select the j-th θ from the grid and continue until we have the full sequence.
  "
  
  theta = rep(0, t)
  theta[1] = zgrid[which.min(abs(ϵ[1] - zgrid))] # gives the index of zgrid θ closest to ϵ[1]
  
  for (s in 2:t){
    # (2)
    if(theta[s-1] <= zgrid[1]){
      i = 0
    }else{
      i = length(zgrid[zgrid < theta[s-1]])
    }
    prob = P[i+1,]
      
    # (3)
    if(pnorm(ϵ[s], mean=μ, sd=σ) <= prob[1]){
      j = 0
    } else {
      j = length(zgrid[prob < pnorm(ϵ[s], mean=μ, sd=σ)])
    }
    theta[s] = zgrid[j+1]
  }
  
  par(mar=c(5, 4, 4, 7))
  plot(obs, ar1, type="l", col="blue", main="Tauchen's Method",
       ylab="", ylim=c(1.2*min(ar1),1.2*max(ar1)))
  abline(h = zgrid[1:N], col="gray", lty = 2,lwd=1)
  lines(obs, theta, type="l", col="red")
  legend("topright",
         c(paste("AR(1) ρ=",ρ),paste("θ discr. N=",N)),
         fill=c("blue","red"), inset=c(-0.29, 0.3), bty = "n", xpd=TRUE)
  return(theta)
}

Rouwenhorst_discretization = function(ρ, μ, σ, m, N, t){
  θ_N = sqrt(N - 1) * σ / sqrt(1 - ρ^2)
  θ_1 = - θ_N
  zgrid = seq(from=θ_1, to=θ_N, length.out = N)
  
  p = (1 + ρ) / 2
  
  P = matrix(c(p, 1-p, 1-p, p),2,2)
  
  for (s in 2:(N-1)){
    zrow = rep(0,s)
    zcol = t(rep(0,s+1))
    A = cbind(P, zrow)
    A = rbind(A, zcol)
  
    B = cbind(P, zrow)
    B = rbind(zcol, B)
  
    C = cbind(zrow, P)
    C = rbind(C, zcol)
  
    D = cbind(zrow, P)
    D = rbind(zcol, D)
  
    P = p*A + (1-p)*B + (1-p)*C + p*D
  }
  
  Pnorm = matrix(, nrow=N, ncol=N)
  for (i in 1:N){
    Pnorm[i,] = P[i,] / sum(P[i,])
  }
  
  "(1)
  Having the raw values of P, now we do the cummulative sum of columns so that in
  the probabilities add up to 1 as we move further along right."
  for (j in 2:N){
    Pnorm[,j] = Pnorm[,j] + Pnorm[,j-1]
  }
  
  "(2)
  Having this probability matrix, now I want to discretize the noise process.
  I'll do this by picking the θ_i from the grid which is closest to wherever
  it is that I am in the sequence.

   (3)
  Afterwards, we compare the cdf of the realized epsilon to the cdf of each entry
  from the selected vector and pick the j-th value which is closest to it. Thus,
  we'll select the j-th θ from the grid and continue until we have the full sequence.
  "
  theta = rep(0, t)
  theta[1] = zgrid[which.min(abs(ϵ[1] - zgrid))] # gives the index of zgrid θ closest to ϵ[1]
  
  for (s in 2:t){
    # (2)
    if(theta[s-1] <= zgrid[1]){
      i = 0
    }else{
      i = length(zgrid[zgrid < theta[s-1]])
    }
    prob = Pnorm[i+1,]
    
    # (3)
    if(pnorm(ϵ[s], mean=μ, sd=σ) <= prob[1]){
      j = 0
    } else {
      j = length(zgrid[prob < pnorm(ϵ[s], mean=μ, sd=σ)])
    }
    theta[s] = zgrid[j+1]
  }
  
  par(mar=c(5, 4, 4, 7))
  plot(obs, ar1, type="l", col="blue", main="Rouwenhorst's Method",
       ylab="", ylim=c(1.2*min(ar1),1.2*max(ar1)))
  abline(h = zgrid[1:N], col="gray", lty = 2,lwd=1)
  lines(obs, theta, type="l", col="red")
  legend("topright",
         c(paste("AR(1) ρ=",ρ),paste("θ discr. N=",N)),
         fill=c("blue","red"), inset=c(-0.29, 0.3), bty = "n", xpd=TRUE)
  return(theta)
}

"BASELINE, N=9, ρ = 0.95"

ar1 = AR1_series(ρ, t)
theta_T95_09 = Tauchen_discretization(ρ, μ, σ, m, N, t)
theta_R95_09 = Rouwenhorst_discretization(ρ, μ, σ, m, N, t)

"MODIFIED, N=15, ρ = 0.95"

ar1 = AR1_series(ρ, t)
theta_T95_15 = Tauchen_discretization(ρ, μ, σ, m, N=15, t)
theta_R95_15 = Rouwenhorst_discretization(ρ, μ, σ, m, N=15, t)

"MODIFIED, N=20, ρ = 0.95"

ar1 = AR1_series(ρ, t)
theta_T95_20 = Tauchen_discretization(ρ, μ, σ, m, N=20, t)
theta_R95_20 = Rouwenhorst_discretization(ρ, μ, σ, m, N=20, t)

"MODIFIED, N=9, ρ = 0.99"

ar1 = AR1_series(ρ=0.99, t)
theta_T99_09 = Tauchen_discretization(ρ=0.99, μ, σ, m, N, t)
theta_R99_09 = Rouwenhorst_discretization(ρ=0.99, μ, σ, m, N, t)

"MODIFIED, N=15, ρ = 0.99"

ar1 = AR1_series(ρ=0.99, t)
theta_T99_15 = Tauchen_discretization(ρ=0.99, μ, σ, m, N=15, t)
theta_R99_15 = Rouwenhorst_discretization(ρ=0.99, μ, σ, m, N=15, t)

"MODIFIED, N=20, ρ = 0.99"

ar1 = AR1_series(ρ=0.99, t)
theta_T99_20 = Tauchen_discretization(ρ=0.99, μ, σ, m, N=20, t)
theta_R99_20 = Rouwenhorst_discretization(ρ=0.99, μ, σ, m, N=20, t)


"ESTIMATING RHOs"
Estimate_rho = function(theta_series){
  y = theta_series[-1]
  x = theta_series[-t]
  
  data = data.frame(y,x)
  ols = lm(y ~ -1 + x, data)
  rho_est = coef(summary(ols))[1]
  return(rho_est)
}

# ρ = 0.95 & N = 9 points in grid
Estimate_rho(theta_T95_09) # ̂ρ = 0.9467994
Estimate_rho(theta_R95_09) # ̂ρ = 0.950237

# ρ = 0.95 & N = 15 points in grid
Estimate_rho(theta_T95_15) # ̂ρ = 0.9485504
Estimate_rho(theta_R95_15) # ̂ρ = 0.9470625

# ρ = 0.95 & N = 20 points in grid
Estimate_rho(theta_T95_20) # ̂ρ = 0.9466363
Estimate_rho(theta_R95_20) # ̂ρ = 0.9472849

# ρ = 0.99 & N = 9 points in grid
Estimate_rho(theta_T99_09) # ̂ρ = 0.9984355
Estimate_rho(theta_R99_09) # ̂ρ = 0.9904363

# ρ = 0.99 & N = 15 points in grid
Estimate_rho(theta_T99_15) # ̂ρ = 0.9916734
Estimate_rho(theta_R99_15) # ̂ρ = 0.9889262

# ρ = 0.99 & N = 20 points in grid
Estimate_rho(theta_T99_20) # ̂ρ = 0.9850951
Estimate_rho(theta_R99_20) # ̂ρ = 0.9895782
