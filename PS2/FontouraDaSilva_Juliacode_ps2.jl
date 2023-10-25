#=

NUMERICAL METHODS 1/2023
João Pedro Fontoura da Silva

~*. Second Problem Set
=#

#=
..... USER GUIDE .....

>> Run code up to line 128 to initialize parameters, packages, functions and other general utility objects
>> Each method's section should be self contained. I took care to redefine variables when needed, in the set up portions
    at the start of each method's code.
>> Problems and issues are all my fault. Or Julia's.

>> I'd like to thank Rafael Vetromille and Lucas Greco for the invaluable help, especially with regards to EGM and Euler Equation Errors.
=#

using Distributions, DataFrames, Plots, Polynomials, Roots, Interpolations, LaTeXStrings

# Baseline parameters
β = 0.987
μ = 2
α = 1/3
δ = 0.012

function u(c)       # utility function
    if c > 0
        return (c^(1 - μ) - 1) / (1 - μ)
    else
        return -1000
    end
end

function umg(c)     # marginal utility function
    if c > 0
        return c^(-μ)
    else
        return 0
    end
end

# For the stochastic process
zgrid_N = 7
m = 3
ρ = 0.95
σ = 0.007

# Setting up shocks grid
dist = Normal(0, σ)
function Tauchen_discretization(; ρ, μ, σ, m, N)
    θ_N = m * σ / sqrt(1 - ρ^2)
    θ_1 = - θ_N
    zgrid = LinRange(θ_1, θ_N, N)
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
    
    θπ = (zgrid, P)
    return θπ
end

# Tauchen[1] returns zgrid while Tauchen[2] returns transition matrix
Tauchen = Tauchen_discretization(; ρ, μ=0, σ, m, N=zgrid_N);

# Setting up capital grid
k_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α)) # steady state capital

kgrid_N = 500
kgrid_min = 0.75 * k_ss
kgrid_max = 1.25 * k_ss
kgrid = LinRange(kgrid_min, kgrid_max, kgrid_N)

# Setting up income grid
function income(;zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)
    y = zeros(kgrid_N, zgrid_N)
    for z in 1:zgrid_N
        for k in 1:kgrid_N
            y[k, z] = exp(zgrid[z]) * (kgrid[k])^α + (1 - δ) * kgrid[k]
        end
    end
    return y
end

# Euler Equation Errors
function EEE(;cnew)
    eulerror = zeros(kgrid_N, zgrid_N)
    for z ∈ 1:zgrid_N
        pi = Tauchen[2][z,:]
        for k ∈ 1:kgrid_N
            y = abs(1 - ((β * umg.(cnew[next_capital[k,z].==kgrid,:]) * (exp(Tauchen[1][z]) * α * next_capital[k,z]^(α-1) + 1-δ) * pi).^(-1/μ))[1]/cnew[k,z])
            eulerror[k,z] = log(10, y)
        end
    end
    return eulerror
end

function EEE_egm(;cnew,g)
    eulerror = zeros(kgrid_N, zgrid_N)
    for z ∈ 1:zgrid_N
        pi = Tauchen[2][z,:]
        for k ∈ 1:kgrid_N
            # this gives the index which is closest to the value in the kgrid for a given (k,z) pair
            idx = findfirst(g[k,z] .== kgrid)
            # Breaking down components just so it's easier to visualize.
            a = umg.(cnew[idx,:])
            b = exp.(Tauchen[1]) .* α .* g[k,:].^(α-1) .+ (1-δ)
            eee = β * transpose(a .* b) * pi
            eulerror[k,z] = log(10, abs(1 - ((eee)^(-1/μ))/cnew[k,z]))
        end
    end
    return eulerror
end

# End of initial set up

### *~ BAREBONES VFI (no tricks included)
# Setting up the VFI
    tol = 10^(-5)
    max_iter = 10000;
    norm = 10 # initial value (disregard)

    V_old = zeros(kgrid_N, zgrid_N);            # initial guess for value function (to be made better)
    g_old = zeros(Int32, kgrid_N, zgrid_N);     # initial guess for policy function (to be made better)

    # Making better guesses for the value and policy functions...
    for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
        g_old[k, z] = k
        V_old[k, z] = u(exp(Tauchen[1][z]) * kgrid[k]^α - δ * kgrid[k])
    end
#

function VFI_barebones(; Tauchen, kgrid, kgrid_N, tol, V_old, g_old, β, α, μ, δ)
    V_new = copy(V_old)
    g_new = copy(g_old)
    k_prime = zeros(kgrid_N, zgrid_N);
    c = zeros(kgrid_N, zgrid_N);

    y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)
    iter = 0
    while iter < max_iter
        for z ∈ 1:zgrid_N
            pi = Tauchen[2][z,:]    # selecting column from P according to state TFP z (thus, probabilities of going to state z)
            for k ∈ 1:kgrid_N
                value = u.(exp(Tauchen[1][z])*kgrid[k]^α + (1-δ)*kgrid[k] .- kgrid) + β * V_old * pi
                V_new[k,z] = maximum(value)
                g_new[k,z] = argmax(value)      # gives index of k which maximized value!
            end
            k_prime[:,z] = kgrid[g_new[:,z]]
        end

        norm = maximum(abs.(V_new - V_old))
        if norm < tol
            break
        else
            V_old = copy(V_new)
            g_old = copy(g_new)
            println("Currently in $iter th iteration. Error is $norm")
            iter += 1
        end
    end

    println("Converged in: $iter iterations!")

    c = y - k_prime
    return V_new, g_new, c, k_prime
end

# Took 847 iterations, 538 seconds to run
VFI_barebones_results = @time VFI_barebones(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)

# Results and plots for VFI barebones
    value=copy(VFI_barebones_results[1])
    policy=copy(VFI_barebones_results[2])
    consumption=copy(VFI_barebones_results[3])
    next_capital=copy(VFI_barebones_results[4])

    plot(kgrid,value,title="Value - VFI Barebones version",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"Initial $k$",legend=:bottomright)
    plot(kgrid,consumption,title="Consumption - VFI Barebones version",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$",legend=:bottomright)
    plot(kgrid,next_capital,title="k' - VFI Barebones version",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$",legend=:bottomright)
    
    # 3D Plot
    surface(exp.(Tauchen[1]),kgrid,value,title="VFI Barebones", xlabel=L"$z$ shock", ylabel=L"initial $k$")
    
    # Euler Equation Errors
    plot(kgrid,EEE(cnew=consumption),title="Euler Equation Errors - VFI barebones version",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|",legend=:bottomright)
#

# .................................................................................................................................

### *~ MONOTONICITY VFI
# Setting up the VFI
    tol = 10^(-5)
    max_iter = 10000;
    norm = 10 # initial value (disregard)

    V_old = zeros(kgrid_N, zgrid_N);            # initial guess for value function (to be made better)
    g_old = zeros(Int32, kgrid_N, zgrid_N);     # initial guess for policy function (to be made better)

    # Making better guesses for the value and policy functions...
    for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
        g_old[k, z] = k
        V_old[k, z] = u(exp(Tauchen[1][z]) * kgrid[k]^α - δ * kgrid[k])
    end
#

# This function uses the monotonicity trick! That way we save on computational time...
function VFI_monotonicity(; Tauchen, kgrid, kgrid_N=kgrid_N, tol, V_old, g_old, β, α, μ, δ)
    V_new = copy(V_old)
    g_new = copy(g_old)
    k_prime = zeros(kgrid_N, zgrid_N);
    c = zeros(kgrid_N, zgrid_N);

    # Reminder: y[k, z] = TFP * (kgrid[k])^α + (1 - δ) * kgrid[k]
    y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)
    iter = 0
    while iter < max_iter
        for z ∈ 1:zgrid_N
            π = Tauchen[2][z,:] # selecting line from P according to state TFP z
            for k ∈ 0:kgrid_N-1
                if k == 0
                    value = u.(y[k+1,z] .- kgrid) + β * V_old * π
                    V_new[k+1,z] = maximum(value)
                    g_new[k+1,z] = findmax(value)[2]
                    k_prime[k+1,z] = kgrid[g_new[k+1,z]]
                else
                    w = count(x->x<=k_prime[k,z], kgrid)     # checks the number of points in kgrid which are <= k' from previous k-step!
                    value = u.(y[k+1,z] .- kgrid[w:kgrid_N]) + β * V_old[w:kgrid_N,:] * π
                    V_new[k+1,z] = maximum(value)
                    g_new[k+1,z] = findmax(value)[2] + w - 1
                    k_prime[k+1,z] = kgrid[g_new[k+1,z]]
                end
            end
        end
    
        norm = maximum(abs.(V_new - V_old))
        if norm < tol
            break
        else
            V_old = copy(V_new)
            g_old = copy(g_new)
            println("Currently in $iter th iteration. Error is $norm")
            iter += 1
        end
    end

    println("Converged in: $iter iterations!")

    for z ∈ 1:zgrid_N # Computing k_prime
        k_prime[:,z] = kgrid[g_new[:,z]]
    end

    c = y - k_prime
    return V_new, g_new, c, k_prime
end

# VFI with monotonicity too 847 iterations, 281 seconds to run
VFI_monotonicity_results = @time VFI_monotonicity(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)

# Results and plots for VFI monotonicity
    value=copy(VFI_monotonicity_results[1])
    policy=copy(VFI_monotonicity_results[2])
    consumption=copy(VFI_monotonicity_results[3])
    next_capital=copy(VFI_monotonicity_results[4])

    plot(kgrid,value,title="Value - VFI with Monotonicity",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"Initial $k$",legend=:bottomright)
    plot(kgrid,consumption,title="Consumption - VFI with Monotonicity",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$",legend=:bottomright)
    plot(kgrid,next_capital,title="k' - VFI with Monotonicity",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$",legend=:bottomright)
    
    # 3D Plot
    surface(exp.(Tauchen[1]),kgrid,value,title="Value - VFI with Monotonicity", xlabel=L"$z$ shock", ylabel=L"initial $k$")
    
    # Euler Equation Errors
    plot(kgrid,EEE(cnew=consumption),title="Euler Equation Errors - VFI w/monotonicity version",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|",legend=:bottomright)
#

# .................................................................................................................................

### *~ CONCAVITY VFI
# Setting up the VFI
    tol = 10^(-5)
    max_iter = 10000;
    norm = 10 # initial value (disregard)

    V_old = zeros(kgrid_N, zgrid_N);            # initial guess for value function (to be made better)
    g_old = zeros(Int32, kgrid_N, zgrid_N);     # initial guess for policy function (to be made better)

    # Making better guesses for the value and policy functions...
    for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
        g_old[k, z] = k
        V_old[k, z] = u(exp(Tauchen[1][z]) * kgrid[k]^α - δ * kgrid[k])
    end
#

# This function uses the concavity trick!
function VFI_concavity(; Tauchen, kgrid, kgrid_N=kgrid_N, tol, V_old, g_old, β, α, μ, δ)
    V_new = copy(V_old)
    g_new = copy(g_old)
    k_prime = zeros(kgrid_N, zgrid_N);
    c = zeros(kgrid_N, zgrid_N);

    H = zeros(kgrid_N,1);

    # Reminder: y[k, z] = TFP * (kgrid[k])^α + (1 - δ) * kgrid[k]
    y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)
    iter = 0
    while iter < max_iter
        for z ∈ 1:zgrid_N
            π = Tauchen[2][z,:] # selecting line from P according to state TFP z
            for k ∈ 1:kgrid_N
        
                # First for k_prime = k_1
                k_prime_N = 1
                H[k_prime_N] = u(y[k,z] - kgrid[k_prime_N]) + β * transpose(V_old[k_prime_N,:]) * π
        
                # now for the remaining k_prime values
                for k_prime_N ∈ 2:kgrid_N
                    H[k_prime_N] = u(y[k,z] - kgrid[k_prime_N]) + β * transpose(V_old[k_prime_N,:]) * π
        
                    # Checking concavity condition
                    if H[k_prime_N] < H[k_prime_N-1]
                        V_new[k,z] = H[k_prime_N-1]
                        g_new[k,z] = k_prime_N-1
                        break
                    elseif k_prime_N == kgrid_N
                        V_new[k,z] = H[k_prime_N]
                        g_new[k,z] = k_prime_N
                        break
                    end
                end
            end
        end

        norm = maximum(abs.(V_new - V_old))
        if norm < tol
            break
        else
            V_old = copy(V_new)
            g_old = copy(g_new)
            println("Currently in $iter th iteration. Error is $norm")
            iter += 1
        end
    end

    println("Converged in: $iter iterations!")

    for z ∈ 1:zgrid_N # Computing k_prime
        k_prime[:,z] = kgrid[g_new[:,z]]
    end

    c = y - k_prime
    return V_new, g_new, c, k_prime
end

# VFI with concavity is taking 847 iterations, 284 seconds to run
VFI_concavity_results = @time VFI_concavity(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)

# Results and plots for VFI concavity
    value=copy(VFI_concavity_results[1])
    policy=copy(VFI_concavity_results[2])
    consumption=copy(VFI_concavity_results[3])
    next_capital=copy(VFI_concavity_results[4])

    plot(kgrid,value,title="Value - VFI with Concavity",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"Initial $k$",legend=:bottomright)
    plot(kgrid,consumption,title="Consumption - VFI with Concavity",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$",legend=:bottomright)
    plot(kgrid,next_capital,title="k' - VFI with Concavity",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$",legend=:bottomright)
    
    # 3D Plot
    surface(exp.(Tauchen[1]),kgrid,value,title="VFI with Concavity", xlabel=L"$z$ shock", ylabel=L"initial $k$")
    
    # Euler Equation Errors
    plot(kgrid,EEE(cnew=consumption),title="Euler Equation Errors - VFI w/concavity version",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|",legend=:bottomright)
#

# .................................................................................................................................

### *~ ACCELERATOR FUNCTION
# Setting up the VFI
    tol = 10^(-5)
    max_iter = 10000;
    norm = 10 # initial value (disregard)

    V_old = zeros(kgrid_N, zgrid_N);            # initial guess for value function (to be made better)
    g_old = zeros(Int32, kgrid_N, zgrid_N);     # initial guess for policy function (to be made better)

    # Making better guesses for the value and policy functions...
    for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
        g_old[k, z] = k
        V_old[k, z] = u(exp(Tauchen[1][z]) * kgrid[k]^α - δ * kgrid[k])
    end
#

function VFI_accelerator(; Tauchen, kgrid, kgrid_N, tol, V_old, g_old, β, α, μ, δ)
    V_new = copy(V_old)
    g_new = copy(g_old)
    k_prime = zeros(kgrid_N, zgrid_N);
    c = zeros(kgrid_N, zgrid_N);

    y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)
    iter = 0
    while iter < max_iter
        if mod(iter, 10) == 0
            for z ∈ 1:zgrid_N
                pi = Tauchen[2][z,:]    # selecting column from P according to state TFP z (thus, probabilities of going to state z)
                for k ∈ 1:kgrid_N
                    value = u.(exp(Tauchen[1][z])*kgrid[k]^α + (1-δ)*kgrid[k] .- kgrid) + β * V_old * pi
                    V_new[k,z] = maximum(value)     # gives maximum value of previous vector for that particular combination of k and z, taking into account all possible k_prime
                    g_new[k,z] = argmax(value)      # gives index of k which maximized 'value'!
                end
                k_prime[:,z] = kgrid[g_new[:,z]]    # gives which k_prime was optimal
            end
    
            norm = maximum(abs.(V_new - V_old))
            if norm < tol
                break
            else
                V_old = copy(V_new)
                g_old = copy(g_new)
                println("Currently in $iter th iteration, error is $norm")
                iter += 1
            end
        else
            for z ∈ 1:zgrid_N
                pi = Tauchen[2][z,:]    # selecting column from P according to state TFP z (thus, probabilities of going to state z)
                for k ∈ 1:kgrid_N
                    V_new[k,z] = u(exp(Tauchen[1][z])*kgrid[k]^α + (1-δ)*kgrid[k] - k_prime[k,z]) + β * transpose(V_old[g_new[k,z],:]) * pi
                end
            end

            norm = maximum(abs.(V_new - V_old))
            if norm < tol
                break
            else
                V_old = copy(V_new)
                g_old = copy(g_new)
                println("Currently in $iter th iteration, error is $norm")
                iter += 1
            end
        end
    end

    println("Converged in: $iter iterations!")

    c = y - k_prime
    return V_new, g_new, c, k_prime
end

# VFI with accelerator took 848 iterations, 30 seconds to run
VFI_accelerator_results = @time VFI_accelerator(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)

# Results and plots VFI accelerator
    value=copy(VFI_accelerator_results[1])
    policy=copy(VFI_accelerator_results[2])
    consumption=copy(VFI_accelerator_results[3])
    next_capital=copy(VFI_accelerator_results[4])

    plot(kgrid,value,title="Value - VFI with Accelerator",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    plot(kgrid,consumption,title="Consumption - VFI with Accelerator",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    plot(kgrid,next_capital,title="k' - VFI with Accelerator",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    
    # 3D Plot
    surface(exp.(Tauchen[1]),kgrid,value,title="Value - VFI with Accelerator", xlabel=L"$z$ shock", ylabel=L"initial $k$")
    
    # Euler Equation Errors
    plot(kgrid,EEE(cnew=consumption),title="Euler Equation Errors - VFI w/accelerator version",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"], ylabel=L"\log_{10} |EEE|",legend=:bottomright)
#

# .................................................................................................................................

### *~ MULTIGRID METHOD
#=
First, solve the problem using a grid with 100 points, then 500 and, finally, 5000. For each successive
grid, use the previous solution as your initial guess (you will need to interpolate). 
=#

# Setting up the VFI
    tol = 10^(-5)
    max_iter = 10000;
    norm = 10 # initial value (disregard)

    # Setting up capital grid
    kgrid_min = 0.75 * k_ss
    kgrid_max = 1.25 * k_ss
#

# method = "barebones", "monotonicity", "accelerator"
function VFI_multigrid(; Tauchen, multigrid_k, tol, β, α, μ, δ, method)

    # N = 100
    kgrid_N = multigrid_k[1]
    kgrid = LinRange(kgrid_min, kgrid_max, kgrid_N)
    V_old = zeros(kgrid_N, zgrid_N);            # initial guess for value function (to be made better)
    g_old = zeros(Int32, kgrid_N, zgrid_N);     # initial guess for policy function (to be made better)
    
    # Better initial guess
    for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
        g_old[k, z] = k
        V_old[k, z] = u(exp(Tauchen[1][z]) * kgrid[k]^α - δ * kgrid[k])
    end
    
    if method == "barebones"
        VFI_barebones_results = VFI_barebones(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)
        value=copy(VFI_barebones_results[1])
    elseif method == "monotonicity" 
        VFI_monotonicity_results = VFI_monotonicity(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)
        value=copy(VFI_monotonicity_results[1])
    else method == "accelerator" 
        VFI_accelerator_results = VFI_accelerator(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)
        value=copy(VFI_accelerator_results[1])
    end

    # N = 500
    kgrid_N = multigrid_k[2]
    kgrid_new = LinRange(kgrid_min, kgrid_max, kgrid_N)
    V_old = zeros(kgrid_N, zgrid_N)
    g_old = zeros(Int32, kgrid_N, zgrid_N)
    
    for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
        V̂ = CubicSplineInterpolation(kgrid, value[:,z], extrapolation_bc=Line())
        V_old[:,z] = V̂.(kgrid_new)
        g_old[k, z] = k
    end
    
    kgrid = copy(kgrid_new)
                
    if method == "barebones"
        VFI_barebones_results = VFI_barebones(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)
        value=copy(VFI_barebones_results[1])
    elseif method == "monotonicity" 
        VFI_monotonicity_results = VFI_monotonicity(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)
        value=copy(VFI_monotonicity_results[1])
    else method == "accelerator" 
        VFI_accelerator_results = VFI_accelerator(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)
        value=copy(VFI_accelerator_results[1])
    end

    # N = 5000
    kgrid_N = multigrid_k[3]
    kgrid_new = LinRange(kgrid_min, kgrid_max, kgrid_N)
    V_old = zeros(kgrid_N, zgrid_N)
    g_old = zeros(Int32, kgrid_N, zgrid_N)
    
    for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
        V̂ = CubicSplineInterpolation(kgrid, value[:,z], extrapolation_bc=Line())
        V_old[:,z] = V̂.(kgrid_new)
        g_old[k, z] = k
    end
    
    kgrid = copy(kgrid_new)
                
    if method == "barebones"
        VFI_barebones_results = VFI_barebones(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)
        value=copy(VFI_barebones_results[1])
        policy=copy(VFI_barebones_results[2])
        consumption=copy(VFI_barebones_results[3])
        next_capital=copy(VFI_barebones_results[4])
    elseif method == "monotonicity" 
        VFI_monotonicity_results = VFI_monotonicity(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)
        value=copy(VFI_monotonicity_results[1])
        policy=copy(VFI_monotonicity_results[2])
        consumption=copy(VFI_monotonicity_results[3])
        next_capital=copy(VFI_monotonicity_results[4])
    else method == "accelerator" 
        VFI_accelerator_results = VFI_accelerator(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, g_old=g_old, β=β, α=α, μ=μ, δ=δ)
        value=copy(VFI_accelerator_results[1])
        policy=copy(VFI_accelerator_results[2])
        consumption=copy(VFI_accelerator_results[3])
        next_capital=copy(VFI_accelerator_results[4])
    end

    return value, policy, consumption, next_capital, kgrid
end

multigrid_k = [100, 500, 5000]
# VFI multigrid w/accelerator method took 11 minutes to run
VFI_multigrid_results = @time VFI_multigrid(; Tauchen, multigrid_k, tol, β, α, μ, δ, method="accelerator")

# Results and plots VFI multigrid
    value=copy(VFI_multigrid_results[1])
    policy=copy(VFI_multigrid_results[2])
    consumption=copy(VFI_multigrid_results[3])
    next_capital=copy(VFI_multigrid_results[4])
    kgrid=copy(VFI_multigrid_results[5])

    plot(kgrid,value,title="Value - VFI with Multigrid",xlabel=L"Initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    plot(kgrid,consumption,title="Consumption - VFI with Multigrid",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    plot(kgrid,next_capital,title="k' - VFI with Multigrid",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    
    # 3D Plot
    surface(exp.(Tauchen[1]),kgrid,value,title="Value - VFI with Multigrid", xlabel=L"$z$ shock", ylabel=L"initial $k$")
    
    # Euler Equation Errors
    function EEE_multi(;cnew)
        eulerror = zeros(multigrid_k[3], zgrid_N)
        for z ∈ 1:zgrid_N
            pi = Tauchen[2][z,:]
            for k ∈ 1:multigrid_k[3]
                # next_capital[k,z]==kgrid used to be simply 'k'
                y = abs(1 - ((β * umg.(cnew[next_capital[k,z].==kgrid,:]) * (exp(Tauchen[1][z]) * α * next_capital[k,z]^(α-1) + 1-δ) * pi).^(-1/μ))[1]/cnew[k,z])
                eulerror[k,z] = log(10, y)
            end
        end
        return eulerror
    end
    plot(kgrid,EEE_multi(cnew=consumption),title="Euler Equation Errors - VFI w/multigrid version",xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|",legend=:bottomright)
#

# .................................................................................................................................

### *~ ENDOGENOUS GRID METHOD
# Setting up EGM
    tol = 10^(-5)
    max_iter = 10000;
    norm = 10 # initial value (disregard)

    # Setting up capital grid
    k_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α)) # steady state capital

    kgrid_N = 500
    kgrid_min = 0.75 * k_ss
    kgrid_max = 1.25 * k_ss
    kgrid = LinRange(kgrid_min, kgrid_max, kgrid_N)

    # Empty matrices
    g_old = zeros(Int32, kgrid_N, zgrid_N);
    g_new = zeros(Int32, kgrid_N, zgrid_N);

    c_old = zeros(kgrid_N, zgrid_N)
    k_prime = zeros(kgrid_N, zgrid_N)

    # Income for each k,z combination
    y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)

    # Making better guesses for the policy and consumption functions...
    for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
        g_old[k, z] = k
    end

    for z ∈ 1:zgrid_N
        k_prime[:,z] = kgrid[g_old[:,z]]        # this 'enlightened' guess is such that k_prime[:,z] = kgrid[k].
    end
    
    c_old = y - k_prime                         # initial guess for consumption matrix;; c0(k,z)
#

function EGM(; Tauchen, kgrid, kgrid_N, tol, c_old, k_prime, β, α, μ, δ)
    c_new = zeros(kgrid_N, zgrid_N);
    k_EGM = zeros(kgrid_N, zgrid_N);
    k_prime_new = zeros(kgrid_N, zgrid_N);

    y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)
    iter = 0
    while iter < max_iter    
        for z ∈ 1:zgrid_N
            pi = Tauchen[2][z,:]
            for k ∈ 1:kgrid_N
                # solve for x to obtain endogenous k grid
                #euler(x) = exp(Tauchen[1][z]) * (x)^α + (1-δ) * (x) - k_prime[k,z] - (β * transpose(umg.(c_old[k,:])) * pi).^(-1/μ)
                euler(x) = exp(Tauchen[1][z]) * (x)^α + (1-δ) * (x) - k_prime[k,z] - (β * transpose(umg.(c_old[k,:]) .* (exp.(Tauchen[1]) .* α .* k_prime[k,:].^(α-1) .+ (1-δ))) * pi).^(-1/μ)
                k_EGM[k,z] = find_zero(euler, kgrid[k])
            end        
        end

        # Interpolation step; fitting the endogenous grid to obtain k' :: k'(k_EGM,z) = k >>> k'(k,z) = '?'
        for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
            G = LinearInterpolation(k_EGM[:,z], kgrid, extrapolation_bc=Line())   # for each z, calling associated k_EGM to obtain function k'(k_EGM,z) = k
            k_prime_new[:,z] = G.(kgrid)                                          # obtaining '?' in k'(k,z) = '?'
        end

        c_new = y - k_prime_new

        norm = maximum(abs.(c_new - c_old))
        if norm < tol
            break
        else
            c_old = copy(c_new)
            println("Currently in $iter th iteration. Error is $norm")
            iter += 1
        end
    end

    println("Converged in: $iter iterations!")
    return c_new, k_prime_new, c_old
end

# EGM with baseline specifications took 174 iterations, 160 seconds (2 minutes 40 seconds) to run
EGM_results = @time EGM(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, c_old=c_old, k_prime=k_prime, β=β, α=α, μ=μ, δ=δ)

# Results and plots EGM
    consumption=copy(EGM_results[1])
    next_capital=copy(EGM_results[2])
    consumption_old=copy(EGM_results[3])

    # policy function for capital using consumption obtained from EGM
    g_policy = zeros(kgrid_N, zgrid_N)
    for z ∈ 1:zgrid_N
        for k ∈ 1:kgrid_N
            g_policy[k,z] = y[k,z] - consumption[k,z]
        end
    end

    # forcing previous g_policy to be on grid
    g_forced = zeros(kgrid_N, zgrid_N)
    for z ∈ 1:zgrid_N
        for k ∈ 1:kgrid_N
            err = abs.(kgrid .- g_policy[k,z])
            x = minimum(err)
            g_forced[k,z] = kgrid[err .== x][1]
        end
    end

    plot(kgrid,consumption,title="Consumption - EGM version",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    plot(kgrid,g_forced,title="k' - EGM version",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    
    # Euler Equation Errors
    plot(kgrid,EEE_egm(cnew=consumption, g=g_forced),title="Euler Equation Errors - EGM version",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|", legend=:bottomright)
#


# Comparison between methods!

#= In order to compare all methods used in questions 3, 4 and 6 to the baseline method,
I'll simply check whether the outputs are close enough such that differences can be
attributed to computational errors or to the finiteness of the iterations.=#

# Value function
maximum(abs.(VFI_barebones_results[1] - VFI_monotonicity_results[1]))       # 8.526512829121202e-14
maximum(abs.(VFI_barebones_results[1] - VFI_concavity_results[1]))          # 1.2789769243681803e-13
maximum(abs.(VFI_barebones_results[1] - VFI_accelerator_results[1]))        # 6.759163611036456e-6
maximum(abs.(VFI_concavity_results[1] - VFI_monotonicity_results[1]))       # 1.2789769243681803e-13
maximum(abs.(VFI_accelerator_results[1] - VFI_monotonicity_results[1]))     # 6.759163611036456e-6
maximum(abs.(VFI_concavity_results[1] - VFI_accelerator_results[1]))        # 6.75916365366902e-6

# Policy function for consumption
maximum(abs.(VFI_barebones_results[3] - VFI_monotonicity_results[3]))       # 0.0
maximum(abs.(VFI_barebones_results[3] - VFI_concavity_results[3]))          # 0.0
maximum(abs.(VFI_barebones_results[3] - VFI_accelerator_results[3]))        # 0.0
maximum(abs.(VFI_concavity_results[3] - VFI_monotonicity_results[3]))       # 0.0
maximum(abs.(VFI_accelerator_results[3] - VFI_monotonicity_results[3]))     # 0.0
maximum(abs.(VFI_concavity_results[3] - VFI_accelerator_results[3]))        # 0.0
maximum(abs.(EGM_results[1] - VFI_barebones_results[3]))                    # 0.02869016093792709
maximum(abs.(EGM_results[1] - VFI_monotonicity_results[3]))                 # 0.02869016093792709
maximum(abs.(EGM_results[1] - VFI_concavity_results[3]))                    # 0.02869016093792709
maximum(abs.(EGM_results[1] - VFI_accelerator_results[3]))                  # 0.02869016093792709

# Policy function for capital
maximum(abs.(VFI_barebones_results[4] - VFI_monotonicity_results[4]))       # 0.0
maximum(abs.(VFI_barebones_results[4] - VFI_concavity_results[4]))          # 0.0
maximum(abs.(VFI_barebones_results[4] - VFI_accelerator_results[4]))        # 0.0
maximum(abs.(VFI_concavity_results[4] - VFI_monotonicity_results[4]))       # 0.0
maximum(abs.(VFI_accelerator_results[4] - VFI_monotonicity_results[4]))     # 0.0
maximum(abs.(VFI_concavity_results[4] - VFI_accelerator_results[4]))        # 0.0
maximum(abs.(EGM_results[2] - VFI_barebones_results[4]))                    # 0.02869016093792709
maximum(abs.(EGM_results[2] - VFI_monotonicity_results[4]))                 # 0.02869016093792709
maximum(abs.(EGM_results[2] - VFI_concavity_results[4]))                    # 0.02869016093792709
maximum(abs.(EGM_results[2] - VFI_accelerator_results[4]))                  # 0.02869016093792709