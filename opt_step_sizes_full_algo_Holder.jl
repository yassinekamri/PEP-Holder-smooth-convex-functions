# ============================================================================
# Full first-order method for Hölder smooth convex functions — Step-Size Optimization via Linearization method
# objective : f(x_N) - f(x_*)
# ----------------------------------------------------------------------------
# Julia code to optimize the step-sizes of full gradient methods over Hölder smooth convex functions  using the
# linearization method described in:
#   Y. Kamri, J. M. Hendrickx, and F. Glineur.
#   "Numerical Design of Optimized First-Order Algorithms." arXiv, 2025.
#   Link: https://arxiv.org/abs/2507.20773
#
#
# Dependencies:
#   - JuMP, MosekTools, Mosek
#   - LinearAlgebra
#   - ProgressBars (optional)
#   - JLD2 (optional)
# ============================================================================

# Imports
using JuMP, MosekTools, Mosek
using LinearAlgebra
using ProgressBars
using JLD2

# -------------------------------------------------------------------------------------------------
# Compute xbar, gbarn dbar for full gradient methods
# -------------------------------------------------------------------------------------------------
# Populates the vectors xbar, gbar which represent respectively iterates and
# the associated gradients
function computations_x_g!(N,gamma,xbar,gbar)
    fill!(xbar,0.0)
    fill!(gbar,0.0)
   
    dimG = N + 2
    dimF = N + 1

    for i = 1:dimF
        gbar[i + 1, i] = 1
    end
    
    for i in 1:N+2
        U = zeros(dimG, 1)
        U[1, 1] = 1
        if i == 1
            x = vec(U)
        elseif i == N+2
            x = vec(zeros(dimG, 1))
        else
            iter = U
            for j = 1:i-1
                iter = iter .- gamma[i-1,j]*gbar[:,j]
            end
            x = vec(iter)
        end
        xbar[:,i] = x
    end
end
# ---------------------------------------------------------------------------
# Dual PEP formulation
# ---------------------------------------------------------------------------
# Constructs and solves the dual PEP for full gradient method over Hölder smooth convex functions
function pep_dual_memoryless_holder(gamma,xbar,gbar,fbar, beta)
    K = size(gamma,1)
    dimG = K + 2
    dimF = K + 1
    exp = 0.5*(1 + beta) / beta
    coeff = beta/(1 + beta)

    model = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => 1e-7))
    set_silent(model)
    #set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", 60.0)
    @variable(model, tau >= 0) 
    @variable(model, lb[1:K+2,1:K+2] >= 0) # dual variables associated to the interpolation conditions
    @variable(model, mu[1:K+2, 1:K+2]) #  variables associated to the dual power cone constraints
    @variable(model, delta[1:K+2, 1:K+2]) #  variables associated to the dual power cone constraints 
    @variable(model, s[1:K+2, 1:K+2]) #  variables associated to the dual power cone constraints
    cond = fbar[:,K+1]
    mat = tau * (xbar[:,1] * xbar[:,1]') #  initilization of the gram matrix
    obj = 0 
    for i in 1:K+2
        for j in 1:K+2
            if i != j
                xi = xbar[:,i]
                xj = xbar[:,j]
                gi = gbar[:,i]
                gj = gbar[:,j]
                fi = fbar[:,i]
                fj = fbar[:,j]
                AA = (fj - fi)
                cond -= lb[i,j]*AA
                A = gj * (xi -xj)' + (xi - xj) * gj'
                AAA = (gi - gj) * (gi - gj)'
                exp1 = 1/exp
                @constraint(model, [delta[i,j], s[i,j] , mu[i,j]] in MOI.DualPowerCone(exp1) ) #  dual constraints linked to the interpolation conditions for this class of functions 
                mat += 0.5 * lb[i,j] * A
                mat  +=  mu[i,j] * AAA
                @constraint(model, lb[i,j]*coeff == delta[i,j])
                obj += s[i,j]
            end
        end
    end
     # resulotion of the conic program
    @objective(model, Min , tau + obj )
    @constraint(model, mat in PSDCone())
    @constraint(model, cond .== 0)
    optimize!(model)

    # Access the results
    objective = objective_value(model)
    tau_val = JuMP.value(tau)
    lb_val = JuMP.value.(lb)
    mat_val = JuMP.value.(mat)
    mu_val = JuMP.value.(mu)
    s_val = JuMP.value.(s)
    delta_val = JuMP.value.(delta)
    return objective,tau_val ,lb_val, mat_val,mu_val,s_val, delta_val
end

# ---------------------------------------------------------------------------------------------
# Derivatives of the SDP matrices of the dual PEP with respect to dual variables and step-sizes
# ---------------------------------------------------------------------------------------------
# Computes the derivatives of the SDP matrix in the dual PEP w.r.t. dual variables lambda
function derivative_lambda(i,j,xbar,gbar)
    L = 1
    xi = xbar[:, i]
    xj = xbar[:, j]
    gi = gbar[:, i]
    gj = gbar[:, j]
    return 0.5 * ( gj * (xi - xj)' + (xi - xj) * gj'  )
end

# Computes the derivative of the i-th iterate with respect to the step-size gamma_{j,k}
function dxi_dgammajk(N,i,j,k,gbar)
    dimG =  N + 2
    if j == i-1 && k <= i-1
        return - gbar[:, k]
    else
        return zeros(dimG,1)
    end
end

# Computes the derivatives of the SDP matrices in the dual PEP w.r.t. to the step sizes gamma
function derivative_gamma!(N,t,k,gbar,lb,mat_gamma)
    fill!(mat_gamma, 0.0)
    for i in 1:(N + 2)
        for j in 1:(N + 2)
            if i != j
                gi = gbar[:, i]
                gj = gbar[:, j]
                dxi = dxi_dgammajk(N,i,t,k,gbar)
                dxj = dxi_dgammajk(N,j,t,k,gbar)
                mat_gamma .+= 0.5 * lb[i,j] * (  gj * (dxi - dxj)' + (dxi - dxj) * gj' )
            end
        end
    end
end


# ---------------------------------------------------------------------------
# Linearized subproblem for step-size optimization
# ---------------------------------------------------------------------------
# Constructs and solves the linearized PEP subproblem to compute an update
# for the step-sizes gamma. For details, see:
#   https://arxiv.org/abs/2507.20773

function linearized_pep_holder(gamma_init,beta ,xbar,gbar,fbar,mat_gamma,derivative_tau,D_r)
    K = size(gamma_init,1)
    L = 1
    m = 0
    exp = 0.5*(1 + beta) / beta
    exp1 = 1/exp
    coeff = beta/(1 + beta)
    computations_x_g!(K,gamma_init,xbar,gbar)
    objective,tau_val ,lb_val, mat_val,mu_val,s_val, delta_val = pep_dual_memoryless_holder(gamma_init,xbar,gbar,fbar, beta)

 
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => 1e-10))
    #set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", 60.0)
    set_silent(model)
    # Variables
    @variable(model, tau) 
    @variable(model, lb[1:K+2,1:K+2]) 
    @variable(model, mu[1:K+2, 1:K+2])
    @variable(model, delta[1:K+2, 1:K+2])
    @variable(model, s[1:K+2, 1:K+2])
    @variable(model, gamma[1:K,1:K])
    @variable(model, n)

    mat = mat_val + tau * derivative_tau
    @constraint(model, tau_val + tau >= 0)

    for i = 1:K+2
        for j = 1:K+2
            mat += lb[i,j] * derivative_lambda(i,j,xbar,gbar)
        end
    end

    for i = 1:K
        for j in 1:K
            derivative_gamma!(K,i,j,gbar,lb_val,mat_gamma)
            mat = mat + gamma[i,j]*mat_gamma
        end
    end

    acc_s = 0 
    cond = fbar[:,K+2]
    for i in 1:(K + 2)
        for j in 1:(K + 2)
            if i != j
                fi = fbar[:, i]
                fj = fbar[:, j]
                gi = gbar[:,i]
                gj = gbar[:,j]
                AA = (fj - fi)
                cond -= (lb[i,j] ) * AA
                AAA = (gi - gj) * (gi - gj)'
                mat += mu[i,j] * AAA
                acc_s += s[i,j]
                @constraint(model, lb[i,j] + lb_val[i,j] >= 0)
                @constraint(model, [delta[i,j] + delta_val[i,j], s[i,j] + s_val[i,j] , mu[i,j] + mu_val[i,j]] in MOI.DualPowerCone(exp1) )
                @constraint(model, lb[i,j]*coeff == delta[i,j])

            end
        end
    end

    x = vcat(vec(gamma),vec(lb))
    obj = tau + acc_s
    

    @constraint(model, [n ; x] in SecondOrderCone())
    @constraint(model, n <= D_r)
    @constraint(model, mat in PSDCone())
    @constraint(model, cond .== 0)
    @objective(model, Min, obj)

    optimize!(model)
    obj_val = objective_value(model)
    tau_val = value.(tau)
    dual_vars = value.(lb)
    gamma_val = value.(gamma)
    return obj_val,tau_val,dual_vars,gamma_val
    
end

# ---------------------------------------------------------------------------
# Full optimization routine for the step-sizes
# ---------------------------------------------------------------------------
#solves the linearized PEP subproblem to compute an update, perform a trust region method to ensure
# valid updates and inforce the stoping criterion when either max_iters reached of the norm of the step-size updates 
#smaller than tol = 10e-4
#   https://arxiv.org/abs/2507.20773

function inner_iteration(gamma,beta,D_r, max_iters = 1000, tol = 1e-4)
    K = size(gamma,1)
    xbar = zeros(K+2, K+2)
    gbar = zeros(K+2, K+2)
    fbar = Matrix(I, K+1, K+1) 
    fbar = hcat(fbar, zeros(K+1, 1)) 
    mat_gamma = zeros(K+2, K+2)
    derivative_tau = zeros(K+2, K+2)
    derivative_tau[1,1] = 1
    for i in 1:max_iters
        obj_val,tau_val, dual_vars, gamma_val = linearized_pep_holder(gamma,beta ,xbar,gbar,fbar,mat_gamma,derivative_tau,D_r)
    

        if norm(gamma_val) <= tol
            break
        end

        # Use in-place computation to avoid memory reallocation
        computations_x_g!(K,gamma,xbar,gbar)
        obj,tau_val ,lb_val, mat_val,mu_val,s_val, delta_val = pep_dual_memoryless_holder(gamma,xbar,gbar,fbar, beta)

        computations_x_g!(K,gamma + gamma_val,xbar,gbar)
        obj1,tau_val ,lb_val, mat_val,mu_val,s_val, delta_val = pep_dual_memoryless_holder(gamma,xbar,gbar,fbar, beta)
        ratio = (obj1 - obj) / obj_val
        if (obj1 - obj) >= 0
            D_r *= 0.5
        elseif (obj1 - obj) < 0 && ratio > 0.9
            D_r *= 2
            gamma .+= gamma_val
        elseif (obj1 - obj) < 0 && ratio < 0.1
            D_r *= 0.5
        else
            gamma .+= gamma_val
        end


    end
    computations_x_g!(K,gamma,xbar,gbar)
    obj,tau_val ,lb_val, mat_val,mu_val,s_val, delta_val = pep_dual_memoryless_holder(gamma,xbar,gbar,fbar, beta)
    return obj, gamma, D_r
end

#--------------- test -----------------------

K = 3 # total number of step sizes to optimize
beta = 0.1 # level of Hölder smoothness
gamma = ones(K,K) # initial step sizes
D = 100 # initial trust region diameter
obj, gamma, D_r =  inner_iteration(gamma,beta,D) # results: obj: worst-case upper bound, gamma: optimized step sizes


#---------------- Results ---------------------------------------------

# Upper bound values on the worst-case for our optimized methods for varying number of step-sizes K from 1 to 20

# For a level of smoothness beta = 0.1

WC_upper_bound_01 =  [0.45241515373527896, 0.3466858793110377, 0.2849995677673356, 0.24568896044533567, 0.21765003530235802, 0.19655600198904719, 0.17997860188034517, 0.16654724272544544, 0.15539993968927107, 0.14597112387740874, 0.13787136887135495, 0.13082335880178309, 0.12462367196010496, 0.11911941077321044, 0.11419323093140855, 0.10975341908486888, 0.10572719917240764, 0.1020559825295533, 0.09869203603793436, 0.0955960325363939]


# For a level of smoothness beta = 0.5

WC_upper_bound_05 = [0.2541204156257195, 0.15720571940751604, 0.11202334938859701, 0.08611645480945851, 0.0694225874749796, 0.05782690640016247, 0.04933675292749287, 0.042872349390926164, 0.03779911106733487, 0.03372039683084717, 0.03037598200663344, 0.027588303804727123, 0.025232214406694754, 0.02321706669348038, 0.02147566567840444, 0.01995718019426974, 0.018622487061424373, 0.01744100133640226, 0.016388480468013944, 0.015445477754950792]


# For a level of smoothness beta = 0.7


WC_upper_bound_07 =  [0.18973801601186105, 0.1071369048490367, 0.07157644894570593, 0.05228011421415164, 0.040384514814556474, 0.03242475341705054, 0.026783138755496854, 0.02260981677328293, 0.019418692151989776, 0.01691322453345442, 0.014903112080482867, 0.013261083742637958, 0.011899132035934403, 0.010754599461844032, 0.009781785299357346, 0.008946669470203297, 0.00822342972196658, 0.007592154597156448, 0.00703728034414278, 0.0065464699077943484]
