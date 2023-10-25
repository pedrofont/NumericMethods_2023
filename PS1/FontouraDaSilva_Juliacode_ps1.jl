#=

NUMERICAL METHODS 1/2023
João Pedro Fontoura da Silva

~*. Julia version
=#
using Distributions, Random, Plots, GLM, DataFrames

ρ = 0.95;
σ = 0.007;
N = 9;
m = 3;
μ = 0;

Random.seed!(123)
t = 10000

dist = Normal(μ, σ)
ϵ = rand(dist,t)                    # compiling the noise vector

function AR1_series(;ρ, t)
    y = [0.0 for i in 1:t]          # filling a vector of length n with zeros
    for i in 1:(t-1)
        y[i+1] = ρ*y[i] + ϵ[i+1]
    end
    return y
end

# Visualizing AR(1)
ar1 = AR1_series(;ρ, t);
plot(ar1, label="ρ=0.95")
title!("AR(1) process")

# TAUCHEN'S METHOD

function Tauchen_discretization(; ρ, μ, σ, m, N, t)
    θ_N = m * σ / sqrt(1 - ρ^2)
    θ_1 = - θ_N
    zgrid = range(θ_1, θ_N, length = N)
    Δθ = (θ_N - θ_1) / (N - 1);
    
    # Filling out probability matrix P
    P = fill(0.0, (N, N)) # NOTE: always fill with 0.0, if you use '0' code will interpret as integers-only
    for i in 1:N, j in 1:N
        if j == 1
            P[i,j] = cdf(dist, (zgrid[j] + Δθ/2 - (1 - ρ) * μ - ρ * zgrid[i]))
        elseif j == N
            P[i,j] = 1 - cdf(dist, (zgrid[j] - Δθ/2 - (1 - ρ) * μ - ρ * zgrid[i]))
        else
            P[i,j] = cdf(dist, (zgrid[j] + Δθ/2 - (1 - ρ) * μ - ρ * zgrid[i])) - cdf(dist, (zgrid[j] - Δθ/2 - (1 - ρ) * μ - ρ * zgrid[i]))
        end
    end
    #= (1)
    Having the raw values of P, now we do the cummulative sum of columns so that in
    the probabilities add up to 1 as we move further along right. =#
    for j in 2:N
        P[:,j] = P[:,j] + P[:,j-1]
    end

    #= (2)
    Having this probability matrix, now I want to discretize the noise process.
    I'll do this by picking the θ_i from the grid which is closest to wherever
    it is that I am in the sequence. =#

    #= (3)
    Afterwards, we compare the cdf of the realized epsilon to the cdf of each entry
    from the selected vector and pick the j-th value which is closest to it. Thus,
    we'll select the j-th θ from the grid and continue until we have the full sequence.
    =#
    theta = similar(ϵ)
    theta[1] = zgrid[findmin(abs.(ϵ[1] .- zgrid))[2]] # gives the index of zgrid θ closest to ϵ[1]

    for s in 2:t
        # (2)
        if theta[s-1] <= zgrid[1]
            i = 0
        else
            i = length(zgrid[zgrid .< theta[s-1]])
        end
        prob = P[i+1,:]

        # (3)
        if cdf(dist, ϵ[s]) <= prob[1]
            j = 0
        else
            j = length(zgrid[prob .< cdf(dist,ϵ[s])])
        end
        theta[s] = zgrid[j+1]
    end

    p = plot(ar1, label="AR(1) ρ=$ρ", ylims=(1.2*minimum(ar1),1.2*maximum(ar1)))
    p = plot!(theta, label="θ discr. N=$N")
    p = hline!(p, zgrid[1:N], color = :gray, linestyle = :dash, lw = 1, label = "")
    p = title!("Tauchen's Method")
    display(p)
    return theta
end

# ROUWENHORST'S METHOD

function Rouwenhorst_discretization(; ρ, μ, σ, m, N, t)

    θ_N = sqrt(N - 1) * σ / sqrt(1 - ρ^2);
    θ_1 = - θ_N;
    zgrid = range(θ_1, θ_N, length = N);

    p = (1 + ρ) / 2;

    P = [p 1-p; 1-p p]

    for s in 2:(N-1)
        zrow = zeros(s,1)
        zcol = zeros(1,s+1)
        A = hcat(P, zrow)
        A = vcat(A, zcol)

        B = hcat(P, zrow)
        B = vcat(zcol, B)

        C = hcat(zrow, P)
        C = vcat(C, zcol)

        D = hcat(zrow, P)
        D = vcat(zcol, D)

        P = p*A + (1-p)*B + (1-p)*C + p*D
    end
    Pnorm = P ./ sum(P, dims=2)

    #= (1)
    Having the raw values of P, now we do the cummulative sum of columns so that in
    the probabilities add up to 1 as we move further along right. =#
    for j in 2:N
        Pnorm[:,j] = Pnorm[:,j] + Pnorm[:,j-1]
    end

    #= (2)
    Having this probability matrix, now I want to discretize the noise process.
    I'll do this by picking the θ_i from the grid which is closest to wherever
    it is that I am in the sequence. =#

    #= (3)
    Afterwards, we compare the cdf of the realized epsilon to the cdf of each entry
    from the selected vector and pick the j-th value which is closest to it. Thus,
    we'll select the j-th θ from the grid and continue until we have the full sequence.
    =#
    theta = similar(ϵ)
    theta[1] = zgrid[findmin(abs.(ϵ[1] .- zgrid))[2]] # gives the index of zgrid θ closest to ϵ[1]

    for s in 2:t
        # (2)
        if theta[s-1] <= zgrid[1]
            i = 0
        else
            i = length(zgrid[zgrid .< theta[s-1]])
        end
        prob = Pnorm[i+1,:]

        # (3)
        if cdf(dist, ϵ[s]) <= prob[1]
            j = 0
        else
            j = length(zgrid[prob .< cdf(dist,ϵ[s])])
        end
        theta[s] = zgrid[j+1]
    end

    p = plot(ar1, label="AR(1) ρ=$ρ", ylims=(1.2*minimum(ar1),1.2*maximum(ar1)))
    p = plot!(theta, label="θ discr. N=$N")
    p = hline!(p, zgrid[1:N], color = :gray, linestyle = :dash, lw = 1, label = "")
    p = title!("Rouwenhorst's Method")
    display(p)
    return theta
end


# Baseline, ρ = 0.95 & N = 9
ar1 = AR1_series(;ρ=0.95, t)
theta_T95_09 = Tauchen_discretization(; ρ, μ, σ, m, N, t)
theta_R95_09 = Rouwenhorst_discretization(; ρ, μ, σ, m, N, t)

# trying for ρ = 0.95 & N = 15
ar1 = AR1_series(;ρ=0.95, t)
theta_T95_15 = Tauchen_discretization(; ρ, μ, σ, m, N=15, t)
theta_R95_15 = Rouwenhorst_discretization(; ρ, μ, σ, m, N=15, t)

# trying for ρ = 0.95 & N = 20
ar1 = AR1_series(;ρ=0.95, t)
theta_T95_20 = Tauchen_discretization(; ρ, μ, σ, m, N=20, t)
theta_R95_20 = Rouwenhorst_discretization(; ρ, μ, σ, m, N=20, t)

# trying for ρ = 0.99 & N = 9
ar1 = AR1_series(;ρ=0.99, t)
theta_T99_09 = Tauchen_discretization(; ρ=0.99, μ, σ, m, N=9, t)
theta_R99_09 = Rouwenhorst_discretization(; ρ=0.99, μ, σ, m, N=9, t)

# trying for ρ = 0.99 & N = 15
ar1 = AR1_series(;ρ=0.99, t)
theta_T99_15 = Tauchen_discretization(; ρ=0.99, μ, σ, m, N=15, t)
theta_R99_15 = Rouwenhorst_discretization(; ρ=0.99, μ, σ, m, N=15, t)

# trying for ρ = 0.99 & N = 20
ar1 = AR1_series(;ρ=0.99, t)
theta_T99_20 = Tauchen_discretization(; ρ=0.99, μ, σ, m, N=20, t)
theta_R99_20 = Rouwenhorst_discretization(; ρ=0.99, μ, σ, m, N=20, t)


# From baseline, run OLS regression to estimate ρ
function Estimate_rho(theta_series)
    y = copy(theta_series)
    x = copy(theta_series)

    popfirst!(y)
    pop!(x)

    data = DataFrame(y = y, x = x)
    ols = lm(@formula(y ~ -1 + x), data)
    rho_est = coef(ols)[1]
    return rho_est
end

# ρ = 0.95 & N = 9 points in grid
Estimate_rho(theta_T95_09) # ̂ρ = 0.9513622523626913
Estimate_rho(theta_R95_09) # ̂ρ = 0.9492781520692933

# ρ = 0.95 & N = 15 points in grid
Estimate_rho(theta_T95_15) # ̂ρ = 0.9487922395991596
Estimate_rho(theta_R95_15) # ̂ρ = 0.9471931771715707

# ρ = 0.95 & N = 20 points in grid
Estimate_rho(theta_T95_20) # ̂ρ = 0.9519764676916789
Estimate_rho(theta_R95_20) # ̂ρ = 0.9481421095537376

# ρ = 0.99 & N = 9 points in grid
Estimate_rho(theta_T99_09) # ̂ρ = 0.9988583088396236
Estimate_rho(theta_R99_09) # ̂ρ = 0.98960214791309

# ρ = 0.99 & N = 15 points in grid
Estimate_rho(theta_T99_15) # ̂ρ = 0.9912624284423184
Estimate_rho(theta_R99_15) # ̂ρ = 0.9905748815525974

# ρ = 0.99 & N = 20 points in grid
Estimate_rho(theta_T99_20) # ̂ρ = 0.9883715773133648
Estimate_rho(theta_R99_20) # ̂ρ = 0.9898778732343325