#=

NUMERICAL METHODS 1/2023
João Pedro Fontoura da Silva

~*. Fourth Problem Set

Special thanks to Lucas Greco and Tomas Martinez.
=#

using Parameters, Distributions, DataFrames, Plots, LaTeXStrings, Roots

# Obtaining grid of shocks and transition probability matrix
function Tauchen_discretization(; ρ, μ, σ, m, N, dist=dist)
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

# Baseline parameters (useful to keep as dictionary)
function BaseParam(;
    agrid_N = 500,
    zgrid_N = 9,
    β = 0.96,
    γ = 1.001,
    m = 3,
    ρ = 0.9,
    σ = 0.01,
    ϵ = 10^(-5)
    )

    # Tauchen[1] returns zgrid while Tauchen[2] returns transition matrix
    dist = Normal(0, σ)
    Tauchen = Tauchen_discretization(ρ=ρ, μ=0, σ=σ, m=m, N=zgrid_N, dist=dist)
    
    # Setting up agrid
    χ_rate = (1/β) - 1      # complete markets rate
    function ϕ_limit(r)
        -((exp(Tauchen[1][1]))/r) # natural debt limit; NPL of lowest possible realization of endowment
    end
    agrid = LinRange(ϕ_limit(χ_rate)+ϵ, -ϕ_limit(χ_rate)-ϵ, agrid_N);

    return (β=β, γ=γ, agrid_N=agrid_N, zgrid_N=zgrid_N, Tauchen=Tauchen, agrid=agrid)
end

param = BaseParam()

function HHProblem(param, r)
    @unpack agrid_N, zgrid_N, Tauchen, β, γ, agrid = param

    function u(c;γ=γ)       # utility function
        if c > 0
            return (c^(1 - γ) - 1) / (1 - γ)
        else
            return -10^(10)
        end
    end

    function umg(c)     # marginal utility function
        if c > 0
            return c^(-γ)
        else
            return 0
        end
    end

    function umg_inv(c)
        return c^(-1/γ)
    end

    #= Setting up VFI =#

    # Computing income + asset wealth matrix
    y = zeros(agrid_N,zgrid_N)
    for z ∈ 1:zgrid_N, a ∈ 1:agrid_N
        y[a,z] = exp(Tauchen[1][z]) + (1+r)*agrid[a]
    end

    V_old = zeros(agrid_N, zgrid_N);            
    g_old = zeros(Int32, agrid_N, zgrid_N);

    # Making better guesses for the value and policy functions...
    for z ∈ 1:zgrid_N, a ∈ 1:agrid_N
        g_old[a, z] = a
        V_old[a, z] = u(y[z])
    end

    # Placeholder matrices
    V_new = copy(V_old)
    g_new = copy(g_old)
    a_prime = zeros(agrid_N,zgrid_N)
    c = zeros(agrid_N, zgrid_N)

    # ============================================= #

    #= VFI starts now! =#
    max_iter = 1000
    tol = 10^(-5)

    iter = 0
    while iter < max_iter
        if mod(iter, 10) == 0
            Threads.@threads for z ∈ 1:zgrid_N
                pi = Tauchen[2][z,:]    # probabilities of going to state z)
                for a ∈ 0:agrid_N-1
                    if a == 0
                        value = u.(y[a+1,z] .- agrid) + β * V_old * pi
                        V_new[a+1,z] = maximum(value)
                        g_new[a+1,z] = findmax(value)[2]
                        a_prime[a+1,z] = agrid[g_new[a+1,z]]
                    else
                        w = count(x->x<=a_prime[a,z], agrid)     # number ofpoints in agrid which are <= a' from previous a-step!
                        value = u.(y[a+1,z] .- agrid[w:agrid_N]) + β * V_old[w:agrid_N,:] * pi
                        V_new[a+1,z] = maximum(value)
                        g_new[a+1,z] = findmax(value)[2] + w - 1
                        a_prime[a+1,z] = agrid[g_new[a+1,z]]
                    end
                end
            end
    
            norm = maximum(abs.(V_new - V_old))
            if norm < tol
                break
            else
                V_old = copy(V_new)
                g_old = copy(g_new)
                iter += 1
            end
        else
            Threads.@threads for z ∈ 1:zgrid_N
                pi = Tauchen[2][z,:]    # probabilities of going to state z
                for a ∈ 1:agrid_N
                    V_new[a,z] = u(y[a,z] - a_prime[a,z]) + β * transpose(V_old[g_new[a,z],:]) * pi
                end
            end

            norm = maximum(abs.(V_new - V_old))
            if norm < tol
                break
            else
                V_old = copy(V_new)
                g_old = copy(g_new)
                iter += 1
            end
        end
    end

    c = y - a_prime

    eulerror = zeros(agrid_N, zgrid_N)
    for z ∈ 1:zgrid_N
        pi = Tauchen[2][z,:]
        for a ∈ 1:agrid_N
            c_prime = c[g_new[a,z],:]
            eee = (1+r)*pi'umg.(c_prime)
            eulerror[a,z] = log(10, abs(1-(umg_inv(β*eee)/c[a,z])))
        end
    end

    return (value = V_new, policy = g_new, consumption = c, a_prime = a_prime, eee = eulerror)
end

# This tryout took 6s to run
testHHP = @time HHProblem(param, 0.04)       # seems ok!

function InvariantDistribution(param, HHChoice)
    @unpack agrid_N, zgrid_N, Tauchen = param
    @unpack policy, = HHChoice

    policy_v=vec(policy')
    transition_a = zeros(agrid_N*zgrid_N, agrid_N)
    Threads.@threads for idx ∈ 1:(agrid_N*zgrid_N)
        transition_a[idx, policy_v[idx]] = 1
    end
    transition_a = kron(transition_a, ones(1, zgrid_N))

    transition_z = repeat(Tauchen[2], agrid_N, agrid_N)
    transition_matrix = transition_z.*transition_a

    tol_λ = 10^(-7)
    λ_0 = ones(agrid_N*zgrid_N)./(agrid_N*zgrid_N)

    for iter ∈ 1:1000
        λ_1 = transition_matrix'*λ_0

        if maximum(abs.(λ_1-λ_0))<tol_λ
            λ_0 = copy(λ_1)
            break
        else
            λ_0 = copy(λ_1)
        end
    end

    λ_0 # 4500-size vector
    λ_star = reshape(λ_0, (zgrid_N,agrid_N))' # reshaping vector into 500x9 matrix
    return λ_star
end

# This tryout took <1s to run
testλ = @time InvariantDistribution(param, testHHP)       # seems ok!

function ExcessCredit(param, r)
    @unpack agrid = param

    # Solving household problem
    HHChoice = HHProblem(param, r)

    # Computing invariant distribution
    λ_star = InvariantDistribution(param, HHChoice)

    # Computing excess credit
    pdf_a = sum(λ_star, dims=2)      # summing by column
    Ea = (pdf_a'agrid)[1]            # single value
    return (Ea, HHChoice, λ_star, pdf_a)
end

function HuggettModel(;ρ, γ, σ, r_max)
    param = BaseParam(ρ=ρ, γ=γ, σ=σ)

    # Bounds of interest rate interval
    r0 = 0.03
    r1 = r_max

    # Initializing iterative process
    max_iter_HM = 100
    tol_HM = 0.01

    iter = 0
    for iter ∈ 1:max_iter_HM
        rguess = (r0+r1)/2
        println("\nTrying interest rate: $rguess")
        (Ea, ) = ExcessCredit(param, rguess)
        println("\nExcess Demand: $Ea")

        if abs(Ea) < tol_HM
            println("Equilibrium found! Excess credit: $Ea")
            break
        end

        if iter == max_iter_HM
            println("Could not find equilibrium. Try again!")
        end

        if Ea < 0
            r0 = rguess
        elseif Ea > 0
            r1 = rguess
        end
    end

    r = (r0+r1)/2
    (Ea, HHChoice, λ, pdf_a) = ExcessCredit(param, r)

    return (HHChoice, λ, r, Ea, pdf_a)
end


# LETTER C
# This tryout took 99 sec to run (1m 39s) for agrid_N = 500
(HHChoice, λ, r, Ea, pdf_a) = @time HuggettModel(ρ=0.9, γ=1.001, σ=0.01, r_max=(1/param.β) - 1)
r_c = r # 0.41575520833.... is the equilibrium interest rate

value=copy(HHChoice[1])
policy=copy(HHChoice[2])
consumption=copy(HHChoice[3]) 
a_prime=copy(HHChoice[4])
eee=copy(HHChoice[5])

# Value function (2d and 3d)
plot(param.agrid,value,title="Value - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)
surface(exp.(param.Tauchen[1]),param.agrid,value,title="Value 3D - Huggett", xlabel=L"$z$ shock", ylabel=L"initial $a$")
# Consumption
plot(param.agrid,consumption,title="Consumption - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)
# K_prime
plot(param.agrid,a_prime,title="k' - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright) 
# Invariant
plot(param.agrid,λ,title="Invariant distribution - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:topright)
surface(exp.(param.Tauchen[1]),param.agrid,λ,title="Invariant 3D - Huggett", xlabel=L"$z$ shock", ylabel=L"initial $a$")
# Distribution of assets
plot(param.agrid,pdf_a,title="Distribution of assets - Huggett",xlabel=L"initial $a$")
# EEE
plot(param.agrid,eee,title="Euler Errors - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)

# LETTER D
# This tryout took 175 sec to run (3m) for agrid_N = 500
(HHChoice, λ, r, Ea, pdf_a) = @time HuggettModel(ρ=0.97, γ=1.001, σ=0.01, r_max=(1/param.β) - 1)
r_d = r # 0.041404622..... is the equilibrium interest rate

value=copy(HHChoice[1])
policy=copy(HHChoice[2])
consumption=copy(HHChoice[3]) 
a_prime=copy(HHChoice[4])
eee=copy(HHChoice[5])

# Value function (2d and 3d)
plot(param.agrid,value,title="Value - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)
surface(exp.(param.Tauchen[1]),param.agrid,value,title="Value 3D - Huggett", xlabel=L"$z$ shock", ylabel=L"initial $a$")
# Consumption
plot(param.agrid,consumption,title="Consumption - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)
# K_prime
plot(param.agrid,a_prime,title="k' - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright) 
# Invariant
plot(param.agrid,λ,title="Invariant distribution - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:topright)
surface(exp.(param.Tauchen[1]),param.agrid,λ,title="Invariant 3D - Huggett", xlabel=L"$z$ shock", ylabel=L"initial $a$")
# Distribution of assets
plot(param.agrid,pdf_a,title="Distribution of assets - Huggett",xlabel=L"initial $a$")
# EEE
plot(param.agrid,eee,title="Euler Errors - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)

# LETTER E
(HHChoice, λ, r, Ea, pdf_a) = @time HuggettModel(ρ=0.9, γ=5.0, σ=0.01, r_max=(1/param.β) - 1)
r_e = r # 0.0408919270...... is the equilibrium interest rate

value=copy(HHChoice[1])
policy=copy(HHChoice[2])
consumption=copy(HHChoice[3]) 
a_prime=copy(HHChoice[4])
eee=copy(HHChoice[5])

# Value function (2d and 3d)
plot(param.agrid,value,title="Value - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)
surface(exp.(param.Tauchen[1]),param.agrid,value,title="Value 3D - Huggett", xlabel=L"$z$ shock", ylabel=L"initial $a$")
# Consumption
plot(param.agrid,consumption,title="Consumption - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)
# K_prime
plot(param.agrid,a_prime,title="k' - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright) 
# Invariant
plot(param.agrid,λ,title="Invariant distribution - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:topright)
surface(exp.(param.Tauchen[1]),param.agrid,λ,title="Invariant 3D - Huggett", xlabel=L"$z$ shock", ylabel=L"initial $a$")
# Distribution of assets
plot(param.agrid,pdf_a,title="Distribution of assets - Huggett",xlabel=L"initial $a$")
# EEE
plot(param.agrid,eee,title="Euler Errors - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)

# LETTER F
# This tryout took 158 sec to run (2m 38s) for agrid_N = 500
(HHChoice, λ, r, Ea, pdf_a) = @time HuggettModel(ρ=0.9, γ=1.001, σ=0.05, r_max=(1/param.β) - 1)
r_f = r # 0.0408862304.... # is the equilibrium interest rate

value=copy(HHChoice[1])
policy=copy(HHChoice[2])
consumption=copy(HHChoice[3]) 
a_prime=copy(HHChoice[4])
eee=copy(HHChoice[5])

# Value function (2d and 3d)
plot(param.agrid,value,title="Value - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)
surface(exp.(param.Tauchen[1]),param.agrid,value,title="Value 3D - Huggett", xlabel=L"$z$ shock", ylabel=L"initial $a$")
# Consumption
plot(param.agrid,consumption,title="Consumption - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)
# K_prime
plot(param.agrid,a_prime,title="k' - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright) 
# Invariant
plot(param.agrid,λ,title="Invariant distribution - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:topright)
surface(exp.(param.Tauchen[1]),param.agrid,λ,title="Invariant 3D - Huggett", xlabel=L"$z$ shock", ylabel=L"initial $a$")
# Distribution of assets
plot(param.agrid,pdf_a,title="Distribution of assets - Huggett",xlabel=L"initial $a$")
# EEE
plot(param.agrid,eee,title="Euler Errors - Huggett",xlabel=L"initial $a$",label=[L"$z_1$" L"$z_2$" L"$z_3$" L"$z_4$" L"$z_5$" L"$z_6$" L"$z_7$" L"$z_8$" L"$z_9$"],legend=:bottomright)