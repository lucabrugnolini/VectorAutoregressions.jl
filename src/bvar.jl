using Parameters, GrowableArrays

#TODO: change bVAR --> VAR
# function bVAR(y::Array,p::Int64,i::Bool,reps::Int64,burn::Int64,max_try::Int64,prior::Hyperparameter)
#     i == false ? ((Y,X,β,ϵ,Σ,p) = fit_bvar(y,prior)) : ((Y,X,β,ϵ,Σ,p) = fit_bvar(y,Intercept()))
#     return VAR(Y,X,β,ϵ,Σ,p,Intercept())
# end

@with_kw type Hyperparameter
    λ::Range{Float64} = 0.1:0.1:1
    τ::Range{Float64} = 10*(0.1:0.1:1)
    ε::Float64 = 0.0001
    p = 4
    H::Int64 = 10
    reps::Int64 = 100
    burnin::Int64 = 10
    max_try::Int64 = 100
    update::Int64 = 10
end

function get_prior(y::Array,p::Int64 = 1)
    T,K = size(y) 
    μ = mean(y,1)'
    σ = Array{Float64}(0)
    δ = Array{Float64}(0)
    e0 = Array{Float64}(T-p,0)
    for i = 1:K
        ytemp = y[:,i]
        xtemp = [lagmatrix(ytemp,p) ones(length(ytemp)-p)]
        ytemp = ytemp[1+p:end,:]
        btemp=xtemp\ytemp
        etemp=ytemp-xtemp*btemp
        stemp=etemp'*etemp/length(ytemp)
        if abs(btemp[1]) > 1
            btemp[1] = 1
        end
        σ = vcat(σ,stemp)
        δ = vcat(δ,btemp[1])
        e0 = hcat(e0, etemp)
    end
    return μ, σ, δ, e0 
end

function lagmatrix{F}(x::Array{F},p::Int64,inter::Intercept)
    sk = 1
    T, K = size(x)
    k    = K*p+1
    idx  = repmat(1:K, p)
    X    = Array{F}(T-p, k)
    # building X (t-1:t-p) allocating data from D matrix - avoid checking bounds
    for j = 1+sk:(sk+K*p)
        for i = 1:(T-p)
            lg = round(Int, ceil((j-sk)/K)) - 1 # create index [0 0 1 1 2 2 ...etc]
            @inbounds X[i, j] = x[i+p-1-lg, idx[j-sk]]
        end
    end
    for j=1:T-p
        @inbounds X[j,1] = 1.0
    end
    return X
end

function lagmatrix{F}(x::Array{F},p::Int64)
    sk = 1
    T, K = size(x)
    k    = K*p+1
    idx  = repmat(1:K, p)
    X    = Array{F}(T-p, k)
    # building X (t-1:t-p) allocating data from D matrix - avoid checking bounds
    for j = 1+sk:(sk+K*p)
        for i = 1:(T-p)
            lg = round(Int, ceil((j-sk)/K)) - 1 # create index [0 0 1 1 2 2 ...etc]
            @inbounds X[i, j] = x[i+p-1-lg, idx[j-sk]]
        end
    end
    return X[:,2:end]
end

function lagmatrix{F}(x::Vector{F},p::Int64)
    sk = 1
    T = length(x)
    K = 1
    k    = K*p+1
    idx  = repmat(1:K, p)
    X    = Array{F}(T-p, k)
    # building X (t-1:t-p) allocating data from D matrix - avoid checking bounds
    for j = 1+sk:(sk+K*p)
        for i = 1:(T-p)
            lg = round(Int, ceil((j-sk)/K)) - 1 # create index [0 0 1 1 2 2 ...etc]
            @inbounds X[i, j] = x[i+p-1-lg, idx[j-sk]]
        end
    end
    return X[:,2:end]
end

function create_dummies(λ::Float64,τ::Float64,δ::Array,ε::Float64,p::Int64,μ::Array,σ::Array,K::Int64)
    if typeof(σ) == Array{Float64,2}
        σ = σ[:,1]
    end
    if typeof(μ) == Array{Float64,2}
        μ = μ[:,1]
    end
    if λ > 0
        if ε > 0
            yd1 = [diagm(σ.*δ)./λ;
            zeros(K*(p-1),K);
            diagm(σ);
            zeros(1,K)]
            
            jp = diagm(1:p)
            xd1 = [hcat(kron(jp,diagm(σ)./λ), zeros((K*p),1));
            zeros(K,(K*p)+1);
            hcat(zeros(1,K*p), ε)]
        else
            yd1 = [diagm(σ.*δ)./λ;
            zeros(K*(p-1),K);
            diagm(σ)]
            
            jp = diagm(1:p)
            xd1 = [kron(jp,diagm(σ)./λ);
            zeros(K,(K*p))] 
        end
    end
    if τ > 0
        if ε > 0
            yd2 = diagm(δ.*μ)./τ
            xd2 = [kron(ones(1,p),yd2) zeros(K,1)]
        else
            yd2 = diagm(δ.*μ)./τ
            xd2 = [kron(ones(1,p),yd2)]  
        end
    end
    y = vcat(yd1,yd2)
    x = vcat(xd1,xd2)
    return y,x
end

function sum_loggamma(K::Int64,v::Int64)
    out = 0
    for i = 1:K
        out = out + lgamma((v+1-i)/2)
    end
    return out
end

function mgamln(K::Int64,v::Int64)
    constant = (K*(K-1)/4)*log(pi)
    term2 = sum_loggamma(K,v)
    out = constant + term2
    return out
end

function max_lik_var(y::Array,x::Array,yd::Array,xd::Array)
    T,K = size(y)
    v = size(yd,1)
    y1 = vcat(y,yd)
    x1 = vcat(x,xd)
    #prior moments
    xx0 = xd'*xd
    invxx0 = pinv(xx0) 
    b0 = invxx0*xd'*yd
    v0 = size(yd,1)
    e0 = yd-xd*b0
    sigma0 = e0'*e0
    #posterior moments
    xx1 = x1'*x1
    invxx = pinv(xx1)
    b = invxx*x1'*y1
    v1 = v0+T 
    e = y1-x1*b 
    sigma1 = e'*e
    
    PP = inv(eye(T)+x*invxx0*x') 
    QQ = sigma0
    
    lngam_ratio = mgamln(K,v0)-mgamln(K,v1)
    py = -(lngam_ratio+(T*K/2)*log(pi))+0.5*K*log(det(PP))+(v0/2)*log(det(QQ))-(v1/2)*log(det(QQ+(y-x*b0)'*PP*(y-x*b0)))
    out = py
    return out
end

function get_opt_lag_prior(y::Array,p::Int64,λ::Range{Float64},τ::Range{Float64},δ::Array,ε::Float64,μ::Array,σ::Array)
    K  =  size(y,2)
    outmlik = zeros(length(p),length(λ))
    tableL = zeros(length(p),length(λ))
    tableP = copy(tableL)
    tableD = copy(tableL)
    for i = 1:length(p)
        L = p[i]
        Y = copy(y)
        X = lagmatrix(y,L)
        X = hcat(X, ones(size(X,1)))
        Y = Y[L+1:end,:]
        for j = 1:length(λ)
            yd,xd  =  create_dummies(λ[j],τ[j],δ,ε,L,μ,σ,K)
            # get marginal likelihood
            mlik = max_lik_var(Y,X,yd,xd)
            outmlik[i,j] = mlik
            tableL[i,j] = L
            tableP[i,j] = λ[j]
            tableD[i,j] = τ[j]
        end
    end
    id = outmlik .== maximum(maximum(outmlik))
    optL = tableL[id]
    optP = tableP[id]
    optD = tableD[id]
    return optL, optP, optD
end

function get_coef(β::Array,sigma::Array,ixx::Array,max_try::Int64,K::Int64,p::Int64)
    problem = 0
    vstar = kron(sigma,ixx)
    check = -1
    num_try = 1
    control_stability = 0
    mβ = Array{}
    while check < 0 && num_try < max_try
        mβ = β + (randn(1,K*(K*p+1))*chol(Hermitian(vstar)))'
        
        control_stability = check_stability(mβ,K,p)
        if control_stability == 0
            check = 10
        else
            num_try += 1
        end
    end
    if control_stability > 0
        problem = 1
    end
    return mβ,problem
end

function check_stability(β::Array,K::Int64,p::Int64)
    Kp = K*p 
    K_p1 = K*(p-1)
    mCoef = zeros(Kp,Kp)
    mCoef[K+1:Kp,1:K_p1] = eye(K_p1,K_p1)
    
    temp = reshape(β,Kp+1,K)
    temp = view(temp,1:Kp,1:K)'
    mCoef[1:K,1:Kp] = temp
    ee = maximum(abs.(eigvals(mCoef)))
    is_stable = ee >= 1
    return is_stable
end

function iwpq(v::Int64,ixpx::Array)
    k = size(ixpx,1)
    z = zeros(v,k)
    mu = zeros(k,1)
    for i = 1:v
        z[i,:]=(chol(Hermitian(ixpx))'*randn(k,1))'
    end
    out = inv(z'*z)
    return out
end    

function get_forecast(K::Int64, H::Int64, T::Int64, p::Int64, Y::SubArray, β::Array, sigma::Array) 
    # -------------------------------------------------------------------------
    # get_paths:
    # generates a matrix of simulated paths for Y for given parameters and
    # general set-up (lags, horizon, etc.)
    # -------------------------------------------------------------------------
    # Draw K(0,1) innovations for variance and mean equation:
    csigma = chol(Hermitian(sigma))
    uu = randn(H+p,K)
    # Note we only need H*K innovations, but adding an extra p draws makes 
    # the indexing in the loop below a bit cleaner.
    #compute forecast
    yhat = zeros(H+p,K)
    yhat[1:p,:] = Y[T-p+1:T,:]
    for fi = p+1:H+p
        xhat= Array{Float64}(0) 
        for ji = 1:p
            xhat = vcat(xhat,yhat[fi-ji,:])
        end
        xhat = vcat(xhat,1)
        yhat[fi,:] = xhat'*reshape(β,K*p+1,K) + uu[fi,:]'*csigma
    end
    return yhat
end

function gibbs!(Y::Array,yd::Array,xd::Array,p::Int64,prior::Hyperparameter,mForecast::Array)
    @unpack λ ,τ ,ε ,H ,reps ,burnin ,max_try ,update = prior 
    x = lagmatrix(Y,p)
    T = size(x,1)
    K = size(Y,2)
    iT = size(yd,1)
    x = hcat(x, ones(T))
    y = view(Y,p+1:T+p,:)
    
    y0 = [y; yd]
    x0 = [x; xd]
    # conditional mean of the VAR coefficients
    β0_fix = vec(x0\y0) # ols on the appended data
    β0 = vec(x0\y0) # ols on the appended data
    xx = x0'*x0
    ixx = xx\eye(size(xx,2))  # inv(x0'x0) to be used later in the Gibbs sampling lgorithm
    σ = eye(K) # starting value for sigma
    igibbs = 1
    jgibbs = 0
    
    while jgibbs < reps-burnin
        # Display progress:
        if mod(igibbs,update)==0 
            println("Replication $igibbs of $reps. Lag $p Prior tightness $λ")
        end
        # #step 1: Sample VAR coefficients
        β1, problem = get_coef(β0_fix,σ,ixx,max_try,K,p)
        if problem == 1
            β1 = β0
        else
            β0 = β1
        end
        #draw covariance
        e = y0-x0*reshape(β1,K*p+1,K)
        scale = e'*e
        v = T+iT
        inv_scale = inv(scale)
        σ = iwpq(v,inv_scale)
        if igibbs > burnin && problem == 0
            jgibbs += 1
            yhat = get_forecast(K, H+1, T, p, y, β1,σ)
            mForecast[jgibbs,:,:] = view(yhat,p+1:H+p,:)
        end
        igibbs += 1
    end
    return mForecast
end

function fit_bvar(y::Array,prior::Hyperparameter)
    @unpack λ ,τ ,ε ,p ,H ,reps ,burnin ,max_try ,update = prior 
    T,K = size(y)
    μ, σ, δ, e0  = get_prior(y)
    p_star,λ_star,τ_star = get_opt_lag_prior(y,p,λ,τ,δ,ε,μ,σ)
    yd,xd = create_dummies(λ_star[1],τ_star[1],δ,ε,Int(p_star[1]),μ,σ,K)
    mForecast = zeros(reps-burnin,H,K) 
    gibbs!(y,yd,xd,Int(p_star[1]),prior,mForecast)
    return mForecast
end

