# This package is a work in progress for the estimation and identification of Vector Autoregressive (VAR) models.
# Credits:
# Kilian and Kim 2011, Cremfi codes

module VectorAutoregressions
using Parameters, GrowableArrays

type Intercept end

type VAR
    Y::Array
    X::Array
    β::Array
    ϵ::Array
    Σ::Array
    p::Int64
    inter::Intercept
    VAR(Y,X,β,ϵ,Σ,p,inter) = p <= 0 ? error("Lag-length error: 'p' must be strictly positive") : new(Y,X,β,ϵ,Σ,p,inter)
end

function VAR(y::Array,p::Int64,i::Bool)
    i == false ? ((Y,X,β,ϵ,Σ,p) = fit(y,p)) : ((Y,X,β,ϵ,Σ,p) = fit(y,p,Intercept()))
    return VAR(Y,X,β,ϵ,Σ,p,Intercept())
end

function bVAR(y::Array,p::Int64,i::Bool,reps::Int64,burn::Int64,max_try::Int64,prior::Hyperparameter)
    i == false ? ((Y,X,β,ϵ,Σ,p) = fit_bvar(y,prior)) : ((Y,X,β,ϵ,Σ,p) = fit_bvar(y,Intercept()))
    return VAR(Y,X,β,ϵ,Σ,p,Intercept())
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

abstract type CI end

type IRFs
    IRF::Array
    CI::CI
end


type CI_asy <: CI
    CIl::Array
    CIh::Array
end

type CI_boot <: CI
    CIl::Array
    CIh::Array
end

function IRFs_a(V::VAR,H::Int64,i::Bool)
    if i == true
        mVar1 = get_VAR1_rep(V,V.inter)
        mIRF = irf_chol(V, mVar1, H)
        mStd = irf_ci_asymptotic(V, H, V.inter)
    else
        mVar1 = get_VAR1_rep(V)
        mIRF = irf_chol(V, mVar1, H)
        mStd = irf_ci_asymptotic(V, H)
    end
    mCIl = mIRF - 1.96.*mStd
    mCIh = mIRF + 1.96.*mStd
    return IRFs(mIRF,CI_asy(mCIl,mCIh))
end

function IRFs_b(V::VAR,H::Int64,nrep::Int64,i::Bool)
    if i == true
        mVar1 = get_VAR1_rep(V,V.inter)
        mIRF = irf_chol(V, mVar1, H)
        CI = irf_ci_bootstrap(V, H, nrep,V.inter)
    else
        mVar1 = get_VAR1_rep(V)
        mIRF = irf_chol(V, mVar1, H)
        CI = irf_ci_bootstrap(V, H, nrep)
    end
    return IRFs(mIRF,CI)
end

function fit(y::Array,p::Int64)
    (T,K) = size(y)
    T < K && error("error: there are more covariates than observation")
    X = y
    y = transpose(y)
    Y = y[:,p+1:T]
    X = lagmatrix(X,p)'
    β = (Y*X')/(X*X')
    ϵ = Y - β*X
    Σ = ϵ*ϵ'/(T-p-p*K-1)
    return Y,X,β,ϵ,Σ,p
end

function fit(y::Array,p::Int64,inter::Intercept)
    (T,K) = size(y)
    T < K && error("error: there are more covariates than observation")
    X = y
    y = transpose(y)
    Y = y[:,p+1:T]
    X = lagmatrix(X,p,inter)'
    β = (Y*X')/(X*X')
    ϵ = Y - β*X
    Σ = ϵ*ϵ'/(T-p-p*K-1)
    return Y,X,β,ϵ,Σ,p
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

function commutation(n::Int64, m::Int64)
    # returns Magnus and Neudecker's commutation matrix of dimensions n by m
    k = reshape(kron(vec(eye(n)), eye(m)), n*m, n*m)
    return k
end

function duplication(n::Int64)
    # Returns Magnus and Neudecker's duplication matrix of size n
    # VERY AMBIGUOUS FUNC
    a = tril(ones(n,n))::Array{Float64}
    i = find(a)::Vector{Int64}
    a[i] = 1:size(i,1)
    a = a + tril(a,-1)'
    j = convert(Vector{Int64}, vec(a))::Vector{Int64}
    m = trunc.(Int,(n*(n+1)/2))::Int64
    d = zeros(n*n,m)
    for r = 1:size(d,1)
        d[r, j[r]] = 1.0
    end
    return d
end

function elimat(m::Int64)
    # elimat(m) returns the elimination matrix Lm
    # The elimination matrix Lm is for any matrix F, Vech(F)=Lm Vec(F)
    A = eye(m^2)
    L = A[1:m,:]
    for n in 2:m
        S = A[m*(n-1)+1:n*m,:]
        S = S[n:end,:]
        L = [L;S]
    end
    return L
end

function get_lag_length(D::Array, pbar::Integer)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        V = VAR(D,p,false)
        n,m = size(V.X)
        t = n-p
        sig = V.Σ/t
        if ic == "aic"
            IC[p] = log(det(sig))+2*p*m^2/t
        elseif ic == "bic"
            IC[p] = log(det(sig))+(m^2*p)*log(t)/t
        elseif ic == "hqc"
            IC[p] = log(det(sig))+2*log(log(t))*m^2*p/t
        elseif ic == "aicc"
            b = t/(t-(p*m+m+1))
            IC[p] = t*(log(det(sig))+m)+2*b*(m^2*p+m*(m+1)/2)
        elseif error("'ic' must be aic, bic, hqc or aicc")
        end
    end
    length_ic = indmin(IC)
    println("Using $ic the best lag-length is $length_ic")
    return length_ic
end

function get_lag_length_aic(D::Array, pbar::Integer, inter::Intercept)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        V = VAR(D,p,true)
        n,m = size(V.X)::Tuple{Int64,Int64}
        t = convert(Float64,n-p)
        sig = (V.Σ./t)::Array{Float64}
        IC[p] = log(det(sig))+2*p*m^2/t
    end
    length_ic = indmin(IC)
    println("The best lag-length is $length_ic")
    return length_ic
end

function get_lag_length_aic(D::Array, pbar::Integer)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        V = VAR(D,p,false)
        n,m = size(V.X)::Tuple{Int64,Int64}
        t = convert(Float64,n-p)
        sig = (V.Σ./t)::Array{Float64}
        IC[p] = log(det(sig))+2*p*m^2/t
    end
    length_ic = indmin(IC)
    println("The best lag-length is $length_ic")
    return length_ic
end

function get_lag_length_bic(D::Array, pbar::Integer, inter::Intercept)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        V = VAR(D,p,true)
        n,m = size(V.X)::Tuple{Int64,Int64}
        t = convert(Float64,n-p)
        sig = (V.Σ./t)::Array{Float64}
        IC[p] = log(det(sig))+(m^2*p)*log(t)/t
    end
    length_ic = indmin(IC)
    println("The best lag-length is $length_ic")
    return length_ic
end

function get_lag_length_bic(D::Array, pbar::Integer)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        V = VAR(D,p,false)
        n,m = size(V.X)::Tuple{Int64,Int64}
        t = convert(Float64,n-p)
        sig = (V.Σ./t)::Array{Float64}
        IC[p] = log(det(sig))+(m^2*p)*log(t)/t
    end
    length_ic = indmin(IC)
    println("The best lag-length is $length_ic")
    return length_ic
end

function get_lag_length_hqc(D::Array, pbar::Integer, inter::Intercept)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        V = VAR(D,p,true)
        n,m = size(V.X)::Tuple{Int64,Int64}
        t = convert(Float64,n-p)
        sig = (V.Σ./t)::Array{Float64}
        IC[p] = log(det(sig))+2*log(log(t))*m^2*p/t
    end
    length_ic = indmin(IC)
    println("The best lag-length is $length_ic")
    return length_ic
end

function get_lag_length_hqc(D::Array, pbar::Integer)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        V = VAR(D,p,false)
        n,m = size(V.X)::Tuple{Int64,Int64}
        t = convert(Float64,n-p)
        sig = (V.Σ./t)::Array{Float64}
        IC[p] = log(det(sig))+2*log(log(t))*m^2*p/t
    end
    length_ic = indmin(IC)
    println("The best lag-length is $length_ic")
    return length_ic
end

function get_lag_length_aicc(D::Array, pbar::Integer, inter::Intercept)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        V = VAR(D,p,true)
        n,m = size(V.X)::Tuple{Int64,Int64}
        t = convert(Float64,n-p)
        sig = (V.Σ./t)::Array{Float64}
        b = t/(t-(p*m+m+1))
        IC[p] = t*(log(det(sig))+m)+2*b*(m^2*p+m*(m+1)/2)
    end
    length_ic = indmin(IC)
    println("The best lag-length is $length_ic")
    return length_ic
end

function get_lag_length_aicc(D::Array, pbar::Integer)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        V = VAR(D,p,false)
        n,m = size(V.X)::Tuple{Int64,Int64}
        t = convert(Float64,n-p)
        sig = (V.Σ./t)::Array{Float64}
        b = t/(t-(p*m+m+1))
        IC[p] = t*(log(det(sig))+m)+2*b*(m^2*p+m*(m+1)/2)
    end
    length_ic = indmin(IC)
    println("The best lag-length is $length_ic")
    return length_ic
end

function get_VAR1_rep(V::VAR)
    K = size(V.Σ,1)
    B = vcat(V.β, hcat(eye(K*(V.p-1)), zeros(K*(V.p-1),K))::Array{Float64,2})::Array{Float64,2}
    B = convert(Array{Float64,2},B)
end

function get_VAR1_rep(V::VAR,inter::Intercept)
    K = size(V.Σ,1)
    # v = [V.β[:,1]; zeros(K*(V.p-1),1)]
    B = vcat(V.β[:,2:end], hcat(eye(K*(V.p-1)), zeros(K*(V.p-1),K)))::Array{Float64,2}
    B = convert(Array{Float64,2},B)
end

function irf_ci_asymptotic(V::VAR, H::Int64)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    SIGa = kron(inv(V.X*V.X'/(T-V.p)),V.Σ)
    # Calculation of stdev follows Lutkepohl(2005) p.111,93
    A = get_VAR1_rep(V)
    A0inv = cholfact(V.Σ)[:L]
    STD   = zeros(K^2,H+1)
    COV2   = zeros(K^2,H+1)
    J = [eye(K) zeros(K,K*(V.p-1))]
    L = elimat(K)
    Kk = commutation(K,K)
    Hk = L'/(L*(eye(K^2)+Kk)*kron(A0inv,eye(K))*L')
    D = duplication(K)
    Dplus = (D'*D)\D'
    SIGsig = 2*Dplus*kron(V.Σ,V.Σ)*Dplus'
    Cbar0 = kron(eye(K),J*eye(K*V.p)*J')*Hk
    STD[:,1] = vec((reshape(diag(real(sqrt.(complex(Cbar0*SIGsig*Cbar0'/(T-V.p))))),K,K))')
    COV2[:,1] = vec((reshape(diag((Cbar0*SIGsig*Cbar0'/(T-V.p))),K,K))')
    for h=1:H
        Gi = zeros(K^2,K^2*V.p)
        for m=0:(h-1)
            Gi += kron(J*(A')^(h-1-m),J*(A^m)*J')
        end
        C = kron(A0inv',eye(K))*Gi
        Cbar = kron(eye(K),J*A^h*J')*Hk
        STD[:,h+1] = vec((reshape(diag(real(sqrt.(complex(C*SIGa*C'+Cbar*SIGsig*Cbar')/(T-V.p)))),K,K))')
        COV2[:,h+1] = vec((reshape(diag(((Cbar*SIGsig*Cbar')/(T-V.p))),K,K))')
    end
    return STD
end

function irf_ci_asymptotic(V::VAR, H::Int64, inter::Intercept)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    SIGa = kron(inv(V.X*V.X'/(T-V.p)),V.Σ)
    SIGa = SIGa[K+1:end,K+1:end]
    # Calculation of stdev follows Lutkepohl(2005) p.111,93
    A = get_VAR1_rep(V,V.inter)
    A0inv = cholfact(V.Σ)[:L]
    STD   = zeros(K^2,H+1)
    COV2   = zeros(K^2,H+1)
    J = [eye(K) zeros(K,K*(V.p-1))]
    L = elimat(K)
    Kk = commutation(K,K)
    Hk = L'/(L*(eye(K^2)+Kk)*kron(A0inv,eye(K))*L')
    D = duplication(K)
    Dplus = (D'*D)\D'
    SIGsig = 2*Dplus*kron(V.Σ,V.Σ)*Dplus'
    Cbar0 = kron(eye(K),J*eye(K*V.p)*J')*Hk
    STD[:,1] = vec((reshape(diag(real(sqrt.(complex(Cbar0*SIGsig*Cbar0'/(T-V.p))))),K,K))')
    COV2[:,1] = vec((reshape(diag((Cbar0*SIGsig*Cbar0'/(T-V.p))),K,K))')
    for h=1:H
        Gi = zeros(K^2,K^2*V.p)
        for m=0:(h-1)
            Gi += kron(J*(A')^(h-1-m),J*(A^m)*J')
        end
        C = kron(A0inv',eye(K))*Gi
        Cbar = kron(eye(K),J*A^h*J')*Hk
        STD[:,h+1] = vec((reshape(diag(real(sqrt.(complex(C*SIGa*C'+Cbar*SIGsig*Cbar')/(T-V.p)))),K,K))')
        COV2[:,h+1] = vec((reshape(diag(((Cbar*SIGsig*Cbar')/(T-V.p))),K,K))')
    end
    return STD
end

get_boot_init_int_draw(T::Int64,p::Int64) = Int64(trunc.((T-p+1)*rand()+1))
get_boot_init_vector_draw(T::Int64,p::Int64) = Array{Int64}(trunc.((T-p)*rand(T-p)+1))

function build_sample(V::VAR)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    # Draw block of initial pre-sample values
    y = zeros(K,T)                                       # bootstrap data
    u = zeros(K,T)                             # bootstrap innovations
    iDraw = get_boot_init_int_draw(T,V.p)            # position of initial draw
    y[:,1:V.p] = V.Y[:,iDraw:iDraw+V.p-1]                   # drawing pre-sample obs
    # Draw innovations
    vDraw = get_boot_init_vector_draw(T,V.p)    # index for innovation draws
    u[:, V.p+1:T] = V.ϵ[:,vDraw]                  # drawing innovations
    @inbounds for i = V.p+1:T
        y[:,i] = u[:,i]
        for j =  1:V.p
            y[:,i] += V.β[:,(j-1)*K + 1:j*K]*y[:,i-j]
        end
    end
    return y
end

function build_sample(V::VAR,inter::Intercept)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    # Draw block of initial pre-sample values
    y = zeros(K,T)                                       # bootstrap data
    u = zeros(K,T)                             # bootstrap innovations
    iDraw = get_boot_init_int_draw(T,V.p)            # position of initial draw
    y[:,1:V.p] = V.Y[:,iDraw:iDraw+V.p-1]                   # drawing pre-sample obs
    # Draw innovations
    vDraw = get_boot_init_vector_draw(T,V.p)    # index for innovation draws
    u[:, V.p+1:T] = V.ϵ[:,vDraw]                  # drawing innovations
    @inbounds for i = V.p+1:T
        y[:,i] = V.β[:,1] + u[:,i]
        for j =  1:V.p
            y[:,i] += V.β[:,(j-1)*K + 2:j*K+1]*y[:,i-j]
        end
    end
    return y
end

col_mean(x::Array) = mean(x,2)
test_bias_correction(x::Array) =  any(abs.(eigvals(x)).<1)

function get_boot_ci(V::VAR,H::Int64,nrep::Int64, bDo_bias_corr::Bool,inter::Intercept)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    mIRFbc = zeros(nrep, K^2*(H+1))
    @inbounds for j = 1:nrep
        # Recursively construct sample
        yr = build_sample(V,V.inter)
        yr = (yr .- col_mean(yr))'  # demean yr bootstrap data
        #pr = V.p # also using lag length selection
        Vr = VAR(yr,V.p,true)
        # Bias correction: if the largest root of the companion matrix
        # is less than 1, do BIAS correction
        mVar1 = get_VAR1_rep(Vr,V.inter)::Array{Float64,2}
        bBias_corr_test = test_bias_correction(mVar1)
        if all([bDo_bias_corr, bBias_corr_test])
            mVar1 = bias_correction(Vr,mVar1)
        end
        mIRF = irf_chol(Vr,mVar1,H)
        mIRFbc[j,:] = vec(mIRF')'
    end                     # end bootstrap
    return mIRFbc
end

function get_boot_ci(V::VAR,H::Int64,nrep::Int64, bDo_bias_corr::Bool)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    mIRFbc = zeros(nrep, K^2*(H+1))
    @inbounds for j = 1:nrep
        # Recursively construct sample
        yr = build_sample(V)
        yr = (yr .- col_mean(yr))'  # demean yr bootstrap data
        #pr = V.p # also using lag length selection
        Vr = VAR(yr,V.p,false)
        # Bias correction: if the largest root of the companion matrix
        # is less than 1, do BIAS correction
        mVar1 = get_VAR1_rep(Vr)::Array{Float64,2}
        bBias_corr_test = test_bias_correction(mVar1)
        if all([bDo_bias_corr, bBias_corr_test])
            mVar1 = bias_correction(Vr,mVar1)
        end
        mIRF = irf_chol(Vr,mVar1,H)
        mIRFbc[j,:] = vec(mIRF')'
    end                     # end bootstrap
    return mIRFbc
end

function get_companion_vcv(V::VAR)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    mSigma = vcat(hcat(V.Σ, zeros(K,K*V.p-K)), zeros(K*V.p-K,K*V.p))::Array{Float64,2}
    mSigma = convert(Array{Float64,2},mSigma)
end

function get_sigma_y(V::VAR, mVar1::Array, mSigma::Array)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    vSigma = (eye((K*V.p)^2)::Array{Float64,2}-kron(mVar1,mVar1)::Array{Float64,2})\vec(mSigma)::Vector{Float64}    # Lutkepohl p.29 (2.1.39)
    mSigma_y = reshape(vSigma, K*V.p::Int64, K*V.p::Int64)::Array{Float64,2}
    return convert(Array{Float64,2},mSigma_y)
end

function get_bias(mSigma::Array,mSigma_y::Array,B::Array,I::Array,mSum_eigen::Array)
    return mBias = mSigma*(inv(I - B) + B/(I-B^2) + mSum_eigen)/(mSigma_y)
end

function bias_correction(V::VAR,mVar1::Array)
    # Bias-correction Pope (1990)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    mSigma = get_companion_vcv(V)
    mSigma_y = get_sigma_y(V,mVar1,mSigma)
    I = eye(K*V.p, K*V.p)
    B = mVar1'
    vEigen = eigvals(mVar1)
    mSum_eigen = zeros(K*V.p,K*V.p)
    for h = 1:K*V.p
        mSum_eigen += vEigen[h].\(I - vEigen[h]*B)
    end
    mBias = get_bias(mSigma,mSigma_y,B,I,mSum_eigen)
    mAbias = -mBias/T
    return mBcA = get_proportional_bias_corr(mVar1,mAbias)
end

function get_proportional_bias_corr(mVar1::Array,Abias::Array;iBc_stab::Int64 = 9, iδ::Int64 = 1)
    mBcA = zeros(size(mVar1))
    while iBc_stab >= 1
        # Adjust bias-correction proportionately
        mBcA = real(mVar1-iδ*Abias)
        bcmod = abs.(eigvals(mBcA))
        any(bcmod.>= 1) ?  iBc_stab = 1 : iBc_stab = 0
        iδ += -0.01
        iδ <= 0 && (iBc_stab = 0)
    end
    return mBcA
end

function get_boot_conf_interval(mIRFbc::Array,H::Int64,K::Int64)
    N = size(mIRFbc,2)
    mCILv = zeros(1,N)
    mCIHv = zeros(1,N)
    for i = 1:N
        mCILv[:,i] = quantile(vec(mIRFbc[:,i]),0.025)
        mCIHv[:,i] = quantile(vec(mIRFbc[:,i]),0.975)
    end
    mCIL  = reshape(mCILv',H+1,K^2)'
    mCIH  = reshape(mCIHv',H+1,K^2)'
    return mCIL, mCIH
end

function irf_ci_bootstrap(V::VAR, H::Int64, nrep::Int64; bDo_bias_corr::Bool=true)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    iScale_ϵ = sqrt((T-V.p)/(T-V.p-K*V.p-1))
    u = V.ϵ*iScale_ϵ   # rescaling residual (Stine, JASA 1987)
    mIRFbc = get_boot_ci(V,H,nrep,bDo_bias_corr)
    # Calculate 95 perccent interval endpoints
    mCIL, mCIH = get_boot_conf_interval(mIRFbc::Array,H::Int64,K::Int64)
    return CI_boot(mCIL,mCIH)
end

function irf_ci_bootstrap(V::VAR, H::Int64, nrep::Int64, inter::Intercept; bDo_bias_corr::Bool=true)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    iScale_ϵ = sqrt((T-V.p)/(T-V.p-K*V.p-1))
    u = V.ϵ*iScale_ϵ   # rescaling residual (Stine, JASA 1987)
    mIRFbc = get_boot_ci(V,H,nrep,bDo_bias_corr,V.inter)
    # Calculate 95 perccent interval endpoints
    mCIL, mCIH = get_boot_conf_interval(mIRFbc::Array,H::Int64,K::Int64)
    return CI_boot(mCIL,mCIH)
end

function irf_chol(V::VAR, mVar1::Array, H::Int64)
    K = size(V.Σ,1)
    mSigma = cholfact(V.Σ)[:L]::LowerTriangular  # Cholesky or reduced form
    J = [eye(K,K) zeros(K,K*(V.p-1))]
    mIRF = zeros(K^2,H+1)
    mIRF[:,1] = reshape((J*mVar1^0*J'*mSigma)',K^2,1)::Array
    for i = 1:H
        mIRF[:,i+1] = reshape((J*mVar1^i*J'*mSigma)',K^2,1)::Array
    end
    return mIRF
end

function irf_reduce_form(V::VAR, mVar1::Array, H::Int64)
    K = size(V.Σ,1)
    mSigma = eye(K,K)
    J = [eye(K,K) zeros(K,K*(V.p-1))]
    mIRF = zeros(K^2,H+1)
    mIRF[:,1] = reshape((J*mVar1^0*J'*mSigma)',K^2,1)::Array
    for i = 1:H
        mIRF[:,i+1] = reshape((J*mVar1^i*J'*mSigma)',K^2,1)::Array
    end
    return mIRF
end

function t_test(V::VAR)
    K = size(V.Σ,1)
    Kp = size(V.X,1)
    H = kron(inv(V.X*V.X'),V.Σ)
    SE = sqrt.(diag(H))
    T = reshape(vec(V.β)./SE,K,Kp)
    return T
end

function gen_var1_data!(y::Array,mR::Array,mP,burnin::Int64)
    T,K = size(y)
     for j = 2:T                          
         y[j,:] = mR*y[j-1,:] + mP*randn(K,1)   
     end
     y = y[burnin+1:end,:]                      
     return y .- mean(y,1)
 end

export VAR, IRFs_a, IRFs_b, gen_var1_data!
end # end of the module

