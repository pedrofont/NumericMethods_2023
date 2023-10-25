#=

NUMERICAL METHODS 1/2023
João Pedro Fontoura da Silva

~*. Third Problem Set
=#

using Distributions, DataFrames, Plots, Polynomials, Roots, Interpolations, LaTeXStrings, NLsolve

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
function EEE_egm(;cnew,g)
    eulerror = zeros(kgrid_N, zgrid_N)
    for z ∈ 1:zgrid_N
        pi = Tauchen[2][z,:]
        for k ∈ 1:kgrid_N
            # this gives the index which is closest to the value in the kgrid for a given (k,z) pair
            idx = findfirst(g[k,z] .== kgrid)

            cons = ĉ(γ_star[:,z], g, z, d)
            # Breaking down components just so it's easier to visualize.
            a = umg.(cons[idx,:])
            b = exp.(Tauchen[1]) .* α .* g[k,:].^(α-1) .+ (1-δ)
            eee = β * transpose(a .* b) * pi
            eulerror[k,z] = log(10, abs(1 - ((eee)^(-1/μ))/cnew[k,z]))
        end
    end
    return eulerror
end

function EEE_fe(;cnew,g)
    eulerror = zeros(kgrid_N, zgrid_N)
    for z ∈ 1:zgrid_N
        pi = Tauchen[2][z,:]
        for k ∈ 1:kgrid_N
            # this gives the index which is closest to the value in the kgrid for a given (k,z) pair
            idx = findfirst(g[k,z] .== kgrid)

            cons = zeros(kgrid_N,zgrid_N)
            for w ∈ 1:zgrid_N, t ∈ 1:kgrid_N
                cons[t,w] = ĉ_fe(A_star[:,z], g[t,w], z, N)
            end
            # Breaking down components just so it's easier to visualize.
            a = umg.(cons[idx,:])
            b = exp.(Tauchen[1]) .* α .* g[k,:].^(α-1) .+ (1-δ)
            eee = β * transpose(a .* b) * pi
            eulerror[k,z] = log(10, abs(1 - ((eee)^(-1/μ))/cnew[k,z]))
        end
    end
    return eulerror
end

# .........................................................................................................................

#= FIRST QUESTION:

For this problem set, you must use projection methods. For this, solve the model using a global projection method.
In particular, use Chebyshev polynomials and the collocation method to solve the model.
=#

# Chebyshev polynomials of d j: Tj(x) = cos(j * cos^(-1)(x))
# roots of ChPol of d m: zi = -cos((2i-1)*π/2m)
# consumption hat: Ĉ(γ,K) = ∑ γj * Tj(K) --> on [-1,1] domain we have Tj(K) = Tj(2*(K-Kmax)/(Kmax-Kmin) - 1)

#= Implementation

(1) Pick k0 and evaluate Ĉ0 = Ĉ(γ,k0) ;; get k1 = k0^α + (1-δ)k0 - Ĉ0
(2) given k1 evaluate again to get Ĉ1 = Ĉ(γ,k1)
(3) compute residual function R(γ,k0) = β[Ĉ1/Ĉ0]^(-η) * (1-δ + αk1^(α-1)) - 1
(4) given there are d+1 unknown coefs, need as many equations from (3)
(5) use zeros of a d+1 ChPol as collocation points and solve system for γ
    That is, use roots in the system as values for k
=#
####
    function kgrid_to_cheby(k)
        kgrid_alt = 2*(k-kgrid_min)/(kgrid_max-kgrid_min) - 1
    end

    function cheby_to_kgrid(k)
        k_og = kgrid_min + (k + 1)*(kgrid_max-kgrid_min)/2
    end

    # It works!
    cheby_to_kgrid.(kgrid_to_cheby.(kgrid))

    function ChebyshevPol(d,x)
        cos.(d * acos.(x))
    end

    function ChPolRoots(d)
        r = zeros(d+1)
        for i ∈ 1:(d+1)
            r[i] = -cos((2*i-1)*π/(2*(d+1)))
        end
        r
    end

    function ĉ(γ, k, z, d)
        x = kgrid_to_cheby.(k)
        chat = 0
        for i ∈ 0:d
            chat = chat .+ γ[i+1] * ChebyshevPol(i, x)
        end
        chat
    end

    function Residual(γ, k, z, d)
        c0 = ĉ(γ, k, z, d)
        k1 = exp(Tauchen[1][z])*k^α + (1-δ)*k - c0
        
        c1 = ĉ(γ, k1, z, d)
        e1 = umg.(c1)
        e2 = ((1-δ) .+ exp.(Tauchen[1]).*α.*k1.^(α-1))
        resid = umg.(c0) .- (β .* transpose(e1 .* e2) * Tauchen[2][z,:])
    end

    function R!(R, γ)
        k_col = cheby_to_kgrid.(ChPolRoots(d))
        for i ∈ 1:(d+1)
            R[i] = Residual(γ, k_col[i], z, d)
        end
    end

    d=5;
    guess = [3.0, 1.0, 0.0, 0.0, 0.0, 0.0]; # d+1 elements
    γ_star = zeros(d+1,zgrid_N);
    for j ∈ 1:zgrid_N
        z = j
        function R!(R, γ)
            k_col = cheby_to_kgrid.(ChPolRoots(d))
            for i ∈ 1:(d+1)
                R[i] = Residual(γ, k_col[i], z, d)[1]
            end
        end
        solution = nlsolve(R!, guess)
        γ_star[:,j] = solution.zero
    end

    # Computing consumption policy
    consumption = zeros(kgrid_N,zgrid_N);
    for j ∈ 1:zgrid_N
        consumption[:,j] = ĉ(γ_star[:,j], kgrid, j, d)
    end

    # Computing capital policy
    g_policy = zeros(kgrid_N, zgrid_N);
    y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N);
    for z ∈ 1:zgrid_N
        for k ∈ 1:kgrid_N
            g_policy[k,z] = y[k,z] - consumption[k,z]
        end
    end

    # forcing previous g_policy to be on grid
    g_forced = zeros(kgrid_N, zgrid_N);
    for z ∈ 1:zgrid_N
        for k ∈ 1:kgrid_N
            err = abs.(kgrid .- g_policy[k,z])
            x = minimum(err)
            g_forced[k,z] = kgrid[err .== x][1]
        end
    end

    # Computing value function
    # Set up
        tol = 10^(-5);
        max_iter = 10000;
        norm = 10; # initial value (disregard)
        V_old = zeros(kgrid_N, zgrid_N);
        for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
            V_old[k, z] = u(exp(Tauchen[1][z]) * kgrid[k]^α - δ * kgrid[k])
        end
    #

    function VFI(; Tauchen, kgrid, kgrid_N, tol, V_old, β, α, μ, δ)
        V_new = copy(V_old)
        y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)

        iter = 0
        while iter < max_iter
            for z ∈ 1:zgrid_N
                pi = Tauchen[2][z,:]    # selecting column from P according to state TFP z (thus, probabilities of going to state z)
                for k ∈ 1:kgrid_N
                    V_new[k,z] = u(consumption[k,z]) + β * pi' * V_old[k,:]
                end
            end

            norm = maximum(abs.(V_new - V_old))
            if norm < tol
                break
            else
                V_old = copy(V_new)
                println("Currently in $iter th iteration. Error is $norm")
                iter += 1
            end
        end

        println("Converged in: $iter iterations!")
        return V_new
    end

    value = @time VFI(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, β=β, α=α, μ=μ, δ=δ)

    # Euler Errors to be used are EGM version
    cons_prime = zeros(kgrid_N,zgrid_N)
    for j ∈ 1:zgrid_N
        cons_prime[:,j] = ĉ(γ_star[:,j], g_forced[:,j], j, d)
    end

    plot(kgrid,consumption,title="Consumption policy function - Chebyshev with collocation",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$",legend=:bottomright)
    plot(kgrid,g_forced,title="Capital policy function - Chebyshev with collocation",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    plot(kgrid,EEE_egm(cnew=cons_prime, g=g_forced),title="Euler Equation Errors - Chebyshev with collocation",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|", legend=:bottomright)
    plot(kgrid,value,title="Value function - Chebyshev with collocation",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|", legend=:bottomright)
    # 3D Plot
    surface(exp.(Tauchen[1]),kgrid,value,title="Value function - Chebyshev with collocation", xlabel=L"$z$ shock", ylabel=L"initial $k$")
    
####

# .........................................................................................................................
#= SECOND QUESTION:

Now, use again a projection method: the finite elements method. Divide the space in several elements. To solve
this, use both the Galerkin and collocation methods.
=#

# FINITE ELEMENTS WITH COLLOCATION
####
    N = 10;
    alt_K = LinRange(kgrid[1], kgrid[length(kgrid)], N)

    function ϕ(i, k)
        if i == 1
            if alt_K[i] <= k && k <= alt_K[i+1]
                ϕ = (alt_K[i+1] - k)/(alt_K[i+1] - alt_K[i])
            else
                ϕ = 0
            end

        elseif i == length(alt_K)
            if alt_K[i-1] <= k && k <= alt_K[i]
                ϕ = (k - alt_K[i-1])/(alt_K[i] - alt_K[i-1])
            else
                ϕ = 0
            end
        else
            if alt_K[i-1] <= k && k <= alt_K[i]
                ϕ = (k - alt_K[i-1])/(alt_K[i] - alt_K[i-1])
            elseif alt_K[i] <= k && k <= alt_K[i+1]
                ϕ = (alt_K[i+1] .- k)/(alt_K[i+1] - alt_K[i])
            else
                ϕ = 0
            end
        end
        return ϕ
    end

    function ĉ_fe(A, k, z, N)
        chat = 0.0
        for j ∈ 1:N
            chat = chat .+ A[j] * ϕ(j, k) # pay attention to the indexes...
        end
        chat
    end

    function Residual_fe(A, k, z, N)
        c0 = ĉ_fe(A, k, z, N)
        k1 = exp(Tauchen[1][z])*k^α + (1-δ)*k - c0
        
        c1 = ĉ_fe(A, k1, z, N)
        e1 = umg.(c1)
        e2 = ((1-δ) .+ exp.(Tauchen[1]).*α.*k1.^(α-1))
        resid = umg.(c0) .- (β .* transpose(e1 .* e2) * Tauchen[2][z,:])
    end

    # attention now!
    function R!(R, A)
        k_col = alt_K        # alt_K are limiting points in intervals
        for i ∈ 1:N
            R[i] = Residual(A, k_col[i], z, N)
        end
    end

    guess = zeros(N);
    for m ∈ 1:N
        guess[m] = m
    end
    A_star = zeros(N,zgrid_N);
    for j ∈ 1:zgrid_N
        z = j
        function R!(R, A)
            k_col = alt_K
            for i ∈ 1:N
                R[i] = Residual_fe(A, k_col[i], z, N)[1]
            end
        end
        solution = nlsolve(R!, guess)
        A_star[:,j] = solution.zero
    end

    # Computing consumption policy
    consumption = zeros(kgrid_N,zgrid_N);
    for j ∈ 1:zgrid_N, i ∈ 1:kgrid_N
        consumption[i,j] = ĉ_fe(A_star[:,j], kgrid[i], j, N)
    end

    # Computing capital policy
    g_policy = zeros(kgrid_N, zgrid_N);
    y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N);
    for z ∈ 1:zgrid_N
        for k ∈ 1:kgrid_N
            g_policy[k,z] = y[k,z] - consumption[k,z]
        end
    end

    # forcing previous g_policy to be on grid
    g_forced = zeros(kgrid_N, zgrid_N);
    for z ∈ 1:zgrid_N
        for k ∈ 1:kgrid_N
            err = abs.(kgrid .- g_policy[k,z])
            x = minimum(err)
            g_forced[k,z] = kgrid[err .== x][1]
        end
    end

    # Computing value function
    # Set up
        tol = 10^(-5);
        max_iter = 10000;
        norm = 10; # initial value (disregard)
        V_old = zeros(kgrid_N, zgrid_N);
        for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
            V_old[k, z] = u(exp(Tauchen[1][z]) * kgrid[k]^α - δ * kgrid[k])
        end
    #

    function VFI(; Tauchen, kgrid, kgrid_N, tol, V_old, β, α, μ, δ)
        V_new = copy(V_old)
        y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)

        iter = 0
        while iter < max_iter
            for z ∈ 1:zgrid_N
                pi = Tauchen[2][z,:]    # selecting column from P according to state TFP z (thus, probabilities of going to state z)
                for k ∈ 1:kgrid_N
                    V_new[k,z] = u(consumption[k,z]) + β * pi' * V_old[k,:]
                end
            end

            norm = maximum(abs.(V_new - V_old))
            if norm < tol
                break
            else
                V_old = copy(V_new)
                println("Currently in $iter th iteration. Error is $norm")
                iter += 1
            end
        end

        println("Converged in: $iter iterations!")
        return V_new
    end

    value = @time VFI(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, β=β, α=α, μ=μ, δ=δ)

    # Euler Errors to be used are EGM version
    cons_prime = zeros(kgrid_N,zgrid_N)
    for i ∈ 1:kgrid_N, j ∈ 1:zgrid_N
        cons_prime[i,j] = ĉ_fe(A_star[:,j], g_forced[i,j], j, N)
    end
    plot(kgrid,consumption,title="Consumption policy function - FE with collocation",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$",legend=:bottomright)
    plot(kgrid,g_forced,title="Capital policy function - FE with collocation",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    plot(kgrid,EEE_fe(cnew=cons_prime, g=g_forced),title="Euler Equation Errors - FE with collocation",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|", legend=:bottomright)
    plot(kgrid,value,title="Value function - FE with collocation",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|", legend=:bottomright)
    # 3D Plot
    surface(exp.(Tauchen[1]),kgrid,value,title="Value function - FE with collocation", xlabel=L"$z$ shock", ylabel=L"initial $k$")
    
####

# FINITE ELEMENTS WITH GALERKIN
####
    N = 10;

    alt_K = LinRange(kgrid[1], kgrid[length(kgrid)], N)

    function ϕ(i, k)
        if i == 1
            if alt_K[i] <= k && k <= alt_K[i+1]
                ϕ = (alt_K[i+1] - k)/(alt_K[i+1] - alt_K[i])
            else
                ϕ = 0.0
            end

        elseif i == length(alt_K)
            if alt_K[i-1] <= k && k <= alt_K[i]
                ϕ = (k - alt_K[i-1])/(alt_K[i] - alt_K[i-1])
            else
                ϕ = 0.0
            end
        else
            if alt_K[i-1] <= k && k <= alt_K[i]
                ϕ = (k - alt_K[i-1])/(alt_K[i] - alt_K[i-1])
            elseif alt_K[i] <= k && k <= alt_K[i+1]
                ϕ = (alt_K[i+1] - k)/(alt_K[i+1] - alt_K[i])
            else
                ϕ = 0.0
            end
        end
        return ϕ
    end

    function ĉ_fe(A, k, z, N)
        chat = 0.0
        for j ∈ 1:N
            chat = chat .+ A[j] * ϕ(j, k) # pay attention to the indexes...
        end
        chat
    end

    function Residual_fe(A, k, z, N)
        c0 = ĉ_fe(A, k, z, N)
        k1 = exp(Tauchen[1][z])*k^α + (1-δ)*k - c0
        
        c1 = ĉ_fe(A, k1, z, N)
        e1 = umg.(c1)
        e2 = ((1-δ) .+ exp.(Tauchen[1]).*α.*k1.^(α-1))
        resid = umg.(c0) .- (β .* transpose(e1 .* e2) * Tauchen[2][z,:])
    end

    # Now I need to create a function which implements a quadrature procedure
    guess = zeros(N);
    for h ∈ 1:N
        guess[h] = h
    end
    A_star = zeros(N,zgrid_N);

    # See that I'm solving the problem for each possible shock z, in sequence.
    # This way I don't have to stack them by shock realization
    for j ∈ 1:zgrid_N
        z = j
        function GaussChebyshevQuad(R,A)
        
            for i ∈ 1:N
        
                # INTEGRALS LIMITS
                # lower bound
                if i == 1
                    LB_a = alt_K[1]
                    LB_b = alt_K[2]
                # upper bound
                elseif i == N
                    UB_a = alt_K[i-1]
                    UB_b = alt_K[i]
                # all else (two integrals)
                else
                    LB_a = alt_K[i-1]
                    LB_b = alt_K[i]
                    UB_a = alt_K[i]
                    UB_b = alt_K[i+1]
                end
        
                # INTEGRAL CALCULATION
                # (i) lower bound
                if i == 1
                    res = 0
                    for j ∈ 1:N-1
                        x = -cos((2*j-1)*π/(2*(N-1)))
                        k = LB_a + (1 + x)*(LB_b - LB_a)/2
                        res = res + Residual_fe(A, k, z, N)[1]*ϕ(i,k)*sqrt(1-x^2)
                    end
                    R[i] = π*(LB_b-LB_a)*res/(2*(N-1))
        
                # (ii) upper bound
                elseif i == N
                    res = 0
                    for j ∈ 1:N-1
                        x = -cos((2*j-1)*π/(2*(N-1)))
                        k = UB_a + (1 + x)*(UB_b - UB_a)/2
                        res = res + Residual_fe(A, k, z, N)[1]*ϕ(i,k)*sqrt(1-x^2)
                    end
                    R[i] = π*(UB_b-UB_a)*res/(2*(N-1))
                # (iii.a) all else - first integral
                else
                    res1 = 0
                    for j ∈ 1:N-1
                        x = -cos((2*j-1)*π/(2*(N-1)))
                        k = LB_a + (1 + x)*(LB_b - LB_a)/2
                        res1 = res1 + Residual_fe(A, k, z, N)[1]*ϕ(i,k)*sqrt(1-x^2)
                    end
                # (iii.b) all else - second integral
                    res2 = 0
                    for j ∈ 1:N-1
                        x = -cos((2*j-1)*π/(2*(N-1)))
                        k = UB_a + (1 + x)*(UB_b - UB_a)/2
                        res2 = res2 + Residual_fe(A, k, z, N)[1]*ϕ(i,k)*sqrt(1-x^2)
                    end
                    R[i] = π*(LB_b-LB_a)*res1/(2*(N-1)) + π*(UB_b-UB_a)*res2/(2*(N-1))
                end
            end
        end
        solution = nlsolve(GaussChebyshevQuad, guess)
        A_star[:,j] = solution.zero
    end

    # Computing consumption policy
    consumption = zeros(kgrid_N,zgrid_N);
    for j ∈ 1:zgrid_N, i ∈ 1:kgrid_N
        consumption[i,j] = ĉ_fe(A_star[:,j], kgrid[i], j, N)
    end

    # Computing capital policy
    g_policy = zeros(kgrid_N, zgrid_N);
    y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N);
    for z ∈ 1:zgrid_N
        for k ∈ 1:kgrid_N
            g_policy[k,z] = y[k,z] - consumption[k,z]
        end
    end

    # forcing previous g_policy to be on grid
    g_forced = zeros(kgrid_N, zgrid_N);
    for z ∈ 1:zgrid_N
        for k ∈ 1:kgrid_N
            err = abs.(kgrid .- g_policy[k,z])
            x = minimum(err)
            g_forced[k,z] = kgrid[err .== x][1]
        end
    end

    # Computing value function
    # Set up
        tol = 10^(-5);
        max_iter = 10000;
        norm = 10; # initial value (disregard)
        V_old = zeros(kgrid_N, zgrid_N);
        for z ∈ 1:zgrid_N, k ∈ 1:kgrid_N
            V_old[k, z] = u(exp(Tauchen[1][z]) * kgrid[k]^α - δ * kgrid[k])
        end
    #

    function VFI(; Tauchen, kgrid, kgrid_N, tol, V_old, β, α, μ, δ)
        V_new = copy(V_old)
        y = income(zgrid=Tauchen[1], kgrid=kgrid, kgrid_N=kgrid_N)

        iter = 0
        while iter < max_iter
            for z ∈ 1:zgrid_N
                pi = Tauchen[2][z,:]    # selecting column from P according to state TFP z (thus, probabilities of going to state z)
                for k ∈ 1:kgrid_N
                    V_new[k,z] = u(consumption[k,z]) + β * pi' * V_old[k,:]
                end
            end

            norm = maximum(abs.(V_new - V_old))
            if norm < tol
                break
            else
                V_old = copy(V_new)
                println("Currently in $iter th iteration. Error is $norm")
                iter += 1
            end
        end

        println("Converged in: $iter iterations!")
        return V_new
    end

    value = @time VFI(Tauchen=Tauchen, kgrid=kgrid, kgrid_N=kgrid_N, tol=tol, V_old=V_old, β=β, α=α, μ=μ, δ=δ)

    # Euler Errors to be used are EGM version
    cons_prime = zeros(kgrid_N,zgrid_N)
    for i ∈ 1:kgrid_N, j ∈ 1:zgrid_N
        cons_prime[i,j] = ĉ_fe(A_star[:,j], g_forced[i,j], j, N)
    end
    plot(kgrid,consumption,title="Consumption policy function - FE with Galerkin",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$",legend=:bottomright)
    plot(kgrid,g_forced,title="Capital policy function - FE with Galerkin",xlabel=L"initial $k$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],legend=:bottomright)
    plot(kgrid,EEE_fe(cnew=cons_prime, g=g_forced),title="Euler Equation Errors - FE with Galerkin",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|", legend=:bottomright)
    plot(kgrid,value,title="Value function - FE with Galerkin",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$"],xlabel=L"initial $k$", ylabel=L"\log_{10} |EEE|", legend=:bottomright)
    # 3D Plot
    surface(exp.(Tauchen[1]),kgrid,value,title="Value function - FE with Galerkin", xlabel=L"$z$ shock", ylabel=L"initial $k$")
    
####

# TEST AREA BEGIN (DETERMINISTIC CASE)
# Creating functions that transform kgrid points into [-1,1] grid and vice versa
function kgrid_to_cheby(k)
    kgrid_alt = 2*(k-kgrid_min)/(kgrid_max-kgrid_min) - 1
end

function cheby_to_kgrid(k)
    k_og = kgrid_min + (k + 1)*(kgrid_max-kgrid_min)/2
end

# It works!
cheby_to_kgrid.(kgrid_to_cheby.(kgrid))

function ChebyshevPol(d,x)
    cos.(d * acos.(x))
end

function ChPolRoots(d)
    z = zeros(d+1)
    for i ∈ 1:(d+1)
        z[i] = -cos((2*i-1)*π/(2*(d+1))) # there was an error here! When saying m-order ChPol, we should think of including the zero as well...
    end
    z
end

function ĉ(γ, k, d)
    x = kgrid_to_cheby.(k)
    chat = 0
    for i ∈ 0:d
        chat = chat .+ γ[i+1] * ChebyshevPol(i, x) # pay attention to the indexes...
    end
    chat
end

function Residual(γ, k, d)
    c0 = ĉ(γ, k, d)
    k1 = k^α + (1-δ)*k - c0
    
    c1 = ĉ(γ, k1, d)
    e1 = (c1./c0).^(-μ)
    e2 = (1-δ + α*k1^(α-1))
    resid = β * (e1 .* e2) .- 1
end

function R!(R, γ)
    k_col = cheby_to_kgrid.(ChPolRoots(d))
    for i ∈ 1:(d+1)
        R[i] = Residual(γ, k_col[i], d)
    end
end

d=5
guess = [3.0, 1.0, 0.0, 0.0, 0.0, 0.0] # d+1 elements
solution = nlsolve(R!, guess)
γ_star = solution.zero

a = ĉ(γ_star, kgrid, d)
plot(kgrid,a)
# TEST AREA END