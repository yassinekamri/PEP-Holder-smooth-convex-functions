# ============================================================================
# Memoryless first-order method for Hölder smooth convex functions — Step-Size Optimization via Linearization method
# objective : f(x_N) - f(x_*)
# ----------------------------------------------------------------------------
# Julia code to optimize the step-sizes of memoryless gradient methods over Hölder smooth convex functions  using the
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
# Compute xbar, gbarn dbar for memoryless gradient descent
# -------------------------------------------------------------------------------------------------
# Populates the vectors xbar, gbar which represent respectively iterates and
# the associated gradients
function computations_x_g!(gamma,xbar,gbar)

    fill!(xbar, 0.0)
    fill!(gbar, 0.0)

    K = size(gamma,1)
    dimG = K + 2
    dimF = K + 1

    for i in 1:dimF
      xbar[1, i] = 1
      gbar[i + 1, i] = 1
    end

    for i in 1:K
        xbar[:, i + 1] = xbar[:, i] - gamma[i] * gbar[:, i]
    end
end
# ---------------------------------------------------------------------------
# Dual PEP formulation
# ---------------------------------------------------------------------------
# Constructs and solves the dual PEP for Memoryless gradient method over Hölder smooth convex functions
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
    @variable(model, lb[1:K+2,1:K+2] >= 0)   # dual variables associated to the interpolation conditions
    @variable(model, mu[1:K+2, 1:K+2])  #  variables associated to the dual power cone constraints
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

# Computes the derivative of the i-th iterate with respect to the step-size gamma_{j}
function dxi_dgammaj(N,i,j,gbar)
    dimG = N + 2
    if i <= j
        grad = zeros(dimG,1)
    elseif i == N + 2
        grad = zeros(dimG,1)
    else
        grad = - gbar[:,j]
    end
    return grad
end
# Computes the derivatives of the SDP matrices in the dual PEP w.r.t. to the step sizes gamma
function derivative_gamma!(N,t,gbar,lb,mat_gamma)
    fill!(mat_gamma, 0.0)
    for i in 1:(N + 2)
        for j in 1:(N + 2)
            if i != j
                gi = gbar[:, i]
                gj = gbar[:, j]
                dxi = dxi_dgammaj(N,i,t,gbar)
                dxj = dxi_dgammaj(N,j,t,gbar)
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
    computations_x_g!(gamma_init,xbar,gbar)
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
    @variable(model, gamma[1:K])
    @variable(model, n)

    mat = mat_val + tau * derivative_tau
    @constraint(model, tau_val + tau >= 0)

    for i = 1:K+2
        for j = 1:K+2
            mat += lb[i,j] * derivative_lambda(i,j,xbar,gbar)
        end
    end

    for i = 1:K
        derivative_gamma!(K,i,gbar,lb_val,mat_gamma)
        mat = mat + gamma[i]*mat_gamma
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

    x = vcat(gamma,vec(lb))
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

        computations_x_g!(gamma,xbar,gbar)
        obj,tau_val ,lb_val, mat_val,mu_val,s_val, delta_val = pep_dual_memoryless_holder(gamma,xbar,gbar,fbar, beta)

        computations_x_g!(gamma + gamma_val,xbar,gbar)
        obj1,tau_val ,lb_val, mat_val,mu_val,s_val, delta_val = pep_dual_memoryless_holder(gamma,xbar,gbar,fbar, beta)

        ratio = (obj1 - obj) / obj_val
        if (obj1 - obj) > 0
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
    computations_x_g!(gamma,xbar,gbar)
    obj,tau_val ,lb_val, mat_val,mu_val,s_val, delta_val = pep_dual_memoryless_holder(gamma,xbar,gbar,fbar, beta)
    return obj, gamma, D_r


end

#--------------- test -----------------------

K = 2 # total number of step sizes to optimize
beta = 0.1 # level of Hölder smoothness
gamma = ones(K) # initial step sizes
D = 100 # initial trust region diameter
obj, gamma, D_r =  inner_iteration(gamma,beta,D) # results: obj: worst-case upper bound, gamma: optimized step sizes


#---------------- Results ---------------------------------------------

# Upper bound values on the worst-case for our optimized methods for varying number of step-sizes K from 1 to 30

# For a level of smoothness beta = 0.1

WC_upper_bound_01 = [0.45241515373507646, 0.34696994968079337, 0.2879398937921216, 0.25096332733062665, 0.22477739647334852, 0.20510017489585516, 0.18961436396199555, 0.17703353986899137, 0.16655549486632842, 0.15765851183081578, 0.14998448176063606, 0.14327927244285688, 0.1373567006529335, 0.1320770631790752, 0.12733294042122825, 0.12304036363229216, 0.11913264940769193, 0.1155561174619321, 0.11226699999335935, 0.10922912103212862, 0.10641233762965568, 0.10379137652766178, 0.10134479367224121, 0.09905426217266379, 0.09690403622967028, 0.09488052212889792, 0.09297186518888201, 0.0911677091042978, 0.08945896442301066, 0.08783758905191649]

# For a level of smoothness beta = 0.5

WC_upper_bound_05 = [0.25412041562584325, 0.16278540120952348, 0.1219809673017124, 0.09879956365786524, 0.08327901924723546, 0.07257379926678373, 0.06461601839062148, 0.058245209182444295, 0.053275387179783466, 0.04920247765508255, 0.045947130430094074, 0.043016817870762294, 0.040385893087096836, 0.03818044774271848, 0.03624438522478856, 0.03451632728567342, 0.03289730563482768, 0.03145221456253766, 0.030189215668930368, 0.029185107797663583, 0.028024706853716407, 0.027017486859632005, 0.026125521530736873, 0.025266935208978797, 0.02450048811826134, 0.02380748220508805, 0.02314135637092623, 0.0224887518410331, 0.021902826547166326, 0.02132734188493117]


# For a level of smoothness beta = 0.7


WC_upper_bound_07 = [0.18973801601186105, 0.1129658091251963, 0.08011699453728884, 0.06234861821856661, 0.05142789229836126, 0.043963701460042096, 0.038174372745651164, 0.034015403368731566, 0.030259342262578818, 0.027627778260164677, 0.025425061814110684, 0.0233199505562238, 0.02160555140336227, 0.020227666707614972, 0.019050317565075584, 0.017986283190650112, 0.016971498539763232, 0.016229753922146057, 0.015381616794630036, 0.014700696210594295, 0.014016538326010273, 0.013399258885597995, 0.012865705591425096, 0.012433303383329599, 0.011943343879008887, 0.011604706801403679, 0.011175844237596754, 0.01088304641482173, 0.01051411924324061, 0.010212238189858872]

