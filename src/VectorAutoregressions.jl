# This package is a work in progress for the estimation and identification of Vector Autoregressive (VAR) models.
# Credits:
# Kilian and Kim 2011, Cremfi codes, Geertler and Karadi 2015

module VectorAutoregressions
using Parameters, GrowableArrays, LinearAlgebra, Statistics

using Statistics: mean, std, quantile
using LinearAlgebra: I, cholesky, LowerTriangular, diag, eigvals, det
eye(n) = float.(I(n))
function eye(n, m)
    out = zeros(n, m)
    n = min(n, m)
    out[1:n, 1:n] .= eye(n)
    out
end

struct Intercept end

struct VAR
    mData::AbstractArray
    Y::AbstractArray
    X::AbstractArray
    β::AbstractArray
    ϵ::AbstractArray
    Σ::AbstractArray
    p::Int64
    inter::Intercept
    VAR(mData,Y,X,β,ϵ,Σ,p,inter) = p <= 0 ? error("Lag-length error: 'p' must be strictly positive") : new(mData,Y,X,β,ϵ,Σ,p,inter)
end

function VAR(y::AbstractArray,p::Int64,i::Bool)
    i == false ? ((mData,Y,X,β,ϵ,Σ,p) = fit(y,p)) : ((mData,Y,X,β,ϵ,Σ,p) = fit(y,p,Intercept()))
    return VAR(mData,Y,X,β,ϵ,Σ,p,Intercept())
end

abstract type  CIs end

struct IRFs
    IRF::AbstractArray
    CI::CIs
end

struct CIs_asy <: CIs
    CIl::AbstractArray
    CIh::AbstractArray
end

struct CIs_boot <: CIs
    CIl::AbstractArray
    CIh::AbstractArray
end

function IRFs_a(V::VAR,H::Int64,i::Bool)
    if i == true
        mVar1 = get_VAR1_rep(V,V.inter)
        mIRF = irf_chol(V, mVar1, H)
        mStd,mCov_Σ = irf_ci_asymptotic(V, H, V.inter)
    else
        mVar1 = get_VAR1_rep(V)
        mIRF = irf_chol(V, mVar1, H)
        mStd,mCov_Σ = irf_ci_asymptotic(V, H)
    end
    mCIl = mIRF - 1.96.*mStd
    mCIh = mIRF + 1.96.*mStd
    return IRFs(mIRF,CIs_asy(mCIl,mCIh))
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

function IRFs_ext_instrument(V::VAR,Z::AbstractArray,H::Int64,nrep::Int64, α::AbstractArray, intercept::Bool)
    mIRF = irf_ext_instrument(V, Z, H, intercept)
    CI = irf_ci_wild_bootstrap(V, Z, H, nrep, α, intercept)
    return IRFs(mIRF,CI)
end

function IRFs_localprojection(z::AbstractArray{Float64}, p::AbstractArray{Int64}, H::Int64, A0inv::AbstractArray{Float64},cov_Σ::AbstractArray{Float64})
    T,K = size(z)
    vA0inv = vec(A0inv')
    mIRF = GrowableArray(copy(vA0inv))      # IRF for Horizon 0 --> use auxiliary model for identification
    cov_mIRF = zeros(K^2,1)   
    for h = 1:H                          # IRF for Horizon 1~H
        ph = p[h]                     # lag-order for horizon h
        ys = z[ph+h:T,:]   
        yt = z[ph:T-h,:]                 # RHS variable of interest: y(t)
        x = ones(T-ph-h+1,1)             # constant term
        for i = 2:ph
            x = [x z[ph+1-i:T-i-h+1,:]]  # other RHS lags: y(t-1)~y(t-p+1)
        end
        Mx = get_annhilation_matrix(x)    # annhilation matrix **
        β  = get_lp_beta(ys,yt,Mx)        # IRF by Local Projection **
        push!(mIRF, vec(A0inv'*β))      # structural IRF
        Σ_u = newey_west(ys,yt,Mx,β,h)
        invytMxyt  = inv(yt'*Mx*yt)
        Σ_β = kron(Σ_u, invytMxyt)                                 # var(vec(β))
        Σ_mIRF = kron(Matrix(1.0I,K,K) , A0inv')*Σ_β*kron(Matrix(1.0I,K,K) , A0inv')'    # var(vec(A0inv*β))
        cov_mIRF = [cov_mIRF reshape(diag(Σ_mIRF), K^2, 1)]
    end
    mIRF = Array(mIRF')
    mStd = sqrt.(cov_mIRF+cov_Σ)    
    mCIl = mIRF - 1.96.*mStd
    mCIh = mIRF + 1.96.*mStd
    return IRFs(mIRF,CIs_asy(mCIl, mCIh))
end

function IRFs_localprojection(z::AbstractArray{Float64}, p::AbstractArray{Int64}, H::Int64)
    T,K = size(z)
    vA0inv = vec(eye(K))
    mIRF = GrowableArray(copy(vA0inv))      # IRF for Horizon 0 --> reduce form
    cov_mIRF = zeros(K^2,1)   
    for h = 1:H      # IRF for Horizon 1~H
        ph = p[h]                     # lag-order for horizon h
        ys = z[ph+h:T,:]   
        yt = z[ph:T-h,:]                 # RHS variable of interest: y(t)
        x = ones(T-ph-h+1,1)             # constant term
        for i = 2:ph
            x = [x z[ph+1-i:T-i-h+1,:]]  # other RHS lags: y(t-1)~y(t-p+1)
        end
        Mx = get_annhilation_matrix(x)    # annhilation matrix **
        β  = get_lp_beta(ys,yt,Mx)        # IRF by Local Projection **
        push!(mIRF, vec(β))                      # reduced form IRF
        Σ_u = newey_west(ys,yt,Mx,β,h)
        invytMxyt  = inv(yt'*Mx*yt)
        Σ_β = kron(Σ_u, invytMxyt)                               # var(vec(β))
        cov_mIRF = [cov_mIRF reshape(diag(Σ_β), K^2, 1)]
    end
    mIRF = mIRF'
    mStd = sqrt.(cov_mIRF)    
    mCIl = mIRF - 1.96.*mStd
    mCIh = mIRF + 1.96.*mStd
    return IRFs(mIRF,CIs_asy(mCIl, mCIh))
end

function IRFs_localprojection(z::AbstractArray{Float64}, p::Int64, H::Int64, A0inv::AbstractArray{Float64},cov_Σ::AbstractArray{Float64})
    T,K = size(z)
    vA0inv = vec(A0inv')
    mIRF = GrowableArray(copy(vA0inv))      # IRF for Horizon 0 --> use auxiliary model for identification
    cov_mIRF = zeros(K^2,1)   
    for h = 1:H           # IRF for Horizon 1~H
        ph = p                     # lag-order for horizon h
        ys = z[ph+h:T,:]   
        yt = z[ph:T-h,:]                 # RHS variable of interest: y(t)
        x = ones(T-ph-h+1,1)             # constant term
        for i = 2:ph
            x = [x z[ph+1-i:T-i-h+1,:]]  # other RHS lags: y(t-1)~y(t-p+1)
        end
        Mx = get_annhilation_matrix(x)    # annhilation matrix **
        β  = get_lp_beta(ys,yt,Mx)        # IRF by Local Projection **
        push!(mIRF, vec(A0inv'*β))      # structural IRF
        Σ_u = newey_west(ys,yt,Mx,β,h)
        invytMxyt  = inv(yt'*Mx*yt)
        Σ_β = kron(Σ_u, invytMxyt)                               # var(vec(β))
        Σ_mIRF = kron(Matrix(1.0I,K,K) , A0inv')*Σ_β*kron(Matrix(1.0I,K,K) , A0inv')'    # var(vec(A0inv*β))
        cov_mIRF = [cov_mIRF reshape(diag(Σ_mIRF), K^2, 1)]
    end
    mIRF = mIRF'
    mStd = sqrt.(cov_mIRF+cov_Σ)    
    mCIl = mIRF - 1.96.*mStd
    mCIh = mIRF + 1.96.*mStd
    return IRFs(mIRF,CIs_asy(mCIl, mCIh)), mStd
end

function IRFs_localprojection(z::AbstractArray{Float64}, p::Int64, H::Int64)
    T,K = size(z)
    vA0inv = vec(eye(K))
    mIRF = GrowableArray(copy(vA0inv))      # IRF for Horizon 0 --> reduce form
    cov_mIRF = zeros(K^2,1)                            
    for h = 1:H     # IRF for Horizon 1~H
        ph = p                     # lag-order for horizon h
        ys = z[ph+h:T,:]   
        yt = z[ph:T-h,:]                 # RHS variable of interest: y(t)
        x = ones(T-ph-h+1,1)             # constant term
        for i = 2:ph
            x = [x z[ph+1-i:T-i-h+1,:]]  # other RHS lags: y(t-1)~y(t-p+1)
        end
        Mx = get_annhilation_matrix(x)    # annhilation matrix **
        β  = get_lp_beta(ys,yt,Mx)        # IRF by Local Projection **
        push!(mIRF, vec(β))                      # reduced form IRF
        Σ_u = newey_west(ys,yt,Mx,β,h)
        invytMxyt  = inv(yt'*Mx*yt)
        Σ_β = kron(Σ_u, invytMxyt)                               # var(vec(β))
        cov_mIRF = [cov_mIRF reshape(diag(Σ_β), K^2, 1)]
    end
    mIRF = mIRF'
    mStd = sqrt.(cov_mIRF)    
    mCIl = mIRF - 1.96.*mStd
    mCIh = mIRF + 1.96.*mStd
    return IRFs(mIRF,CIs_asy(mCIl, mCIh))
end

function newey_west(ys::AbstractArray,yt::AbstractArray,Mx::AbstractArray,β::AbstractArray,h::Int64)
    T,K = size(yt)
    u = Mx*ys - Mx*yt*β    # residual from LPs
    μ_u = zeros(1,K) 
    for i = 1:K 
        μ_u[:,i] .= mean(u[:,i])
    end
    iu = size(u,1)
    u0 = u-kron(ones(iu,1),μ_u)
    ρ0 = (u0'*u0)./iu        # sample cov(u)
    Σ_u = ρ0 
    M = h    # truncation point
    for j = 1:M
        Rj = (u0[1:end-j,:]'*u0[j+1:end,:])./(size(u,1))   # R(j)
        Σ_u = Σ_u + (1-j/(M+1))*(Rj+Rj')                 # varcov(u)
    end
    return Σ_u
end

function get_lp_component(z::AbstractArray,p::Int64,H::Int64)
    T,K = size(z)
    ys = z[p+1:T,:]
    yt = z[p:T-1,:]                  # RHS variable of interest: y(t)
    x = ones(T-p,1)                  # constant term
    for j=2:p   
        x = [x z[p+1-j:T-j,:]]       # other RHS variables: y(t-1)~y(t-p+1)
    end
    return ys,yt,x
end

function lp_estimator(ys::AbstractArray,yt::AbstractArray,x::AbstractArray,t::Int64)
    Mx = get_annhilation_matrix(x)    # annhilation matrix **
    β  = get_lp_beta(ys,yt,Mx)        # IRF by Local Projection **
    u  = get_lp_residual(ys,yt,Mx,β)  # residual from LPs
    Σ  = get_variance_estimator(u,t)  # variance of errors
    return β, u, Σ
end

get_variance_estimator(u::AbstractArray,t::Int64) = u'u/t
get_annhilation_matrix(x) = size(x,1) |> λ -> eye(λ) - x/(x'*x)*x'
get_lp_beta(ys::AbstractArray,yt::AbstractArray,Mx::AbstractArray) = (yt'*Mx*yt)\(yt'*Mx*ys)
get_lp_residual(ys::AbstractArray,yt::AbstractArray,Mx::AbstractArray,β::AbstractArray) = Mx*ys - Mx*yt*β

function lp_lagorder(z::AbstractArray,pbar::Int64,H::Int64,ic::String)
    T,K = size(z)
    t     = T-pbar
    vIC  = Array{Int64}(undef, H)
    Ys,Yt,X = get_lp_component(z,pbar,H)
    for j = 1:H                                  # loop for horizon h of IRF
        IC = zeros(pbar,1)                       # the vector of AIC(p)
        ys = Ys[j:end,:]                         # dependent variable: yt+s
        yt = Yt[1:end-j+1,:]                     # regressor of interest: yt
        for m = 1:pbar                           # loop for lag order selection
            x = X[1:end-j+1,1:(m-1)*K+1]         # other independent variables
            β, u, Σ = lp_estimator(ys,yt,x,t)
            if ic == "aic"
                IC[m]  = log(det(Σ)) + 2*(K^2*m)/t                      # AIC statistic
            elseif ic == "bic"
                IC[m]  = log(det(Σ)) + (K^2*m)*log(t)/t                 # SIC statistic
            elseif ic == "aicc"
                b = t/(t-(m*K+K+1))
                IC[m] = t*(log(det(Σ))+K) + 2*b*(K^2*m+K*(K+1)/2)       # HURVICH AND TSAI
            elseif ic == "hqc"
                IC[m]  = log(det(Σ)) + 2*log(log(t))*K^2*m/t            # HQ
            elseif error("ic must be aic, bic, aicc or hqc")
            end
        end
        _, ind = findmin(vec(IC))
        vIC[j] = Int(ind)
    end
    return vIC
end

function fit(y::AbstractArray,p::Int64)
    (T,K) = size(y)
    T < K && error("error: there are more covariates than observation")
    X = y
    y = transpose(y)
    Y = y[:,p+1:T]
    X = lagmatrix(X,p)'
    β = (Y*X')/(X*X')
    ϵ = Y - β*X
    Σ = ϵ*ϵ'/(T-p-p*K-1)
    return y',Y,X,β,ϵ,Σ,p
end

function fit(y::AbstractArray,p::Int64,inter::Intercept)
    (T,K) = size(y)
    T < K && error("error: there are more covariates than observation")
    X = y
    y = transpose(y)
    Y = y[:,p+1:T]
    X = lagmatrix(X,p,inter)'
    β = (Y*X')/(X*X')
    ϵ = Y - β*X
    Σ = ϵ*ϵ'/(T-p-p*K-1)
    return y',Y,X,β,ϵ,Σ,p
end

function lagmatrix(x::AbstractArray,p::Int64,inter::Intercept)
    sk = 1
    T, K = size(x)
    k    = K*p+1
    idx  = repeat(1:K, p)
    X    = Array{eltype(x)}(undef, (T-p, k))
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

function lagmatrix(x::AbstractArray,p::Int64)
    sk = 1
    T, K = size(x)
    k    = K*p+1
    idx  = repeat(1:K, p)
    X    = Array{eltype(x)}(undef, (T-p, k))
    # building X (t-1:t-p) allocating data from D matrix - avoid checking bounds
    for j = 1+sk:(sk+K*p)
        for i = 1:(T-p)
            lg = round(Int, ceil((j-sk)/K)) - 1 # create index [0 0 1 1 2 2 ...etc]
            @inbounds X[i, j] = x[i+p-1-lg, idx[j-sk]]
        end
    end
    return X[:,2:end]
end

function lagmatrix(x::AbstractVector,p::Int64)
    sk = 1
    T = length(x)
    K = 1
    k    = K*p+1
    idx  = repeat(1:K, p)
    X    = Array{eltype(x)}(undef, (T-p, k))
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
    k = reshape(kron(vec(Matrix(1.0I,n,n)), Matrix(1.0I,m,m)), n*m, n*m)
    return k
end

function duplication(n::Int64)
    # Returns Magnus and Neudecker's duplication matrix of size n
    # VERY AMBIGUOUS FUNC
    a = LowerTriangular(ones(n, n))
    inds = (a .== 1) # find(a)::Vector{Int64}
    a[inds] .= 1:Int(sum(inds))
    aT = transpose(a) - I
    for i in 1:n
        aT[i, i] = 0
    end
    a = a + aT
    j = Int.(vec(a)) #convert(Vector{Int64}, vec(a))::Vector{Int64}
    m = trunc.(Int,(n*(n+1)/2))::Int64
    d = zeros(n*n,m)
    for r = 1:size(d,1)
        r, j[r]
        d[r, j[r]] = 1.0
    end
    return d
end

function elimat(m::Int64)
    # elimat(m) returns the elimination matrix Lm
    # The elimination matrix Lm is for any matrix F, Vech(F)=Lm Vec(F)
    A = Matrix(1.0I,m^2,m^2) 
    L = A[1:m,:]
    for n in 2:m
        S = A[m*(n-1)+1:n*m,:]
        S = S[n:end,:]
        L = [L;S]
    end
    return L
end

function var_lagorder(z::AbstractArray,pbar::Int64,ic::String)
    T,K = size(z)::Tuple{Int64,Int64}
    t   = Int(T-pbar)
    IC  = zeros(pbar,1)
    Y = z[pbar+1:T,:]                           # dependent variable
    for p = 1:pbar
        X = ones(Int(t),1)
        for i = 1:p
            X = [X z[pbar+1-i:T-i,:]]            # construct lagged regressors
        end
        β  = (X'*X)\(X'*Y)                    # estimate by multivariate LS **
        u     = Y-X*β                         # errors
        Σ   = (u'*u/t)::AbstractArray{Float64}
        if ic == "aic"                           # variance of errors
            IC[p]     = log(det(Σ))+2*p*K^2/t                      # AIC statistic
        elseif ic == "bic"
            IC[p]   = log(det(Σ))+(K^2*p)*log(t)/t                # SIC statistic
        elseif ic == "aicc"
            b = t/(t-(p*K+K+1))                    # COEFFICIENT FOR AICC
            IC[p] = t*(log(det(Σ))+K)+2*b*(K^2*p+K*(K+1)/2)     # AICC
        elseif ic == "hqc"
            IC[p]   = log(det(Σ))+2*log(log(t))*K^2*p/t # HQC
        elseif error("ic must be aic, bic, aicc or hqc")
        end
    end
    _, length_ic = findmin(vec(IC))
    println("Using $ic the best lag-length is $length_ic")
    return Int(length_ic)
end

function get_VAR1_rep(V::VAR)
    K = size(V.Σ,1)
    B = vcat(V.β, hcat(eye(K*(V.p-1)), zeros(K*(V.p-1),K))::AbstractArray{Float64,2})::AbstractArray{Float64,2}
    B = convert(Array{Float64,2},B)
end

function get_VAR1_rep(V::VAR,inter::Intercept)
    K = size(V.Σ,1)
    # v = [V.β[:,1]; zeros(K*(V.p-1),1)]
    B = vcat(V.β[:,2:end], hcat(eye(K*(V.p-1)), zeros(K*(V.p-1),K)))::AbstractArray{Float64,2}
    B = convert(Array{Float64,2},B)
end

function irf_ci_asymptotic(V::VAR, H::Int64)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    SIGa = kron(inv(V.X*V.X'/(T-V.p)),V.Σ)
    # Calculation of stdev follows Lutkepohl(2005) p.111,93
    A = get_VAR1_rep(V)
    A0inv = cholesky(V.Σ).L
    STD   = zeros(K^2,H+1)
    COV2   = zeros(K^2,H+1)
    J = [Matrix(1.0I,K,K)  zeros(K,K*(V.p-1))]
    L = elimat(K)
    Kk = commutation(K,K)
    Hk = L'/(L*(Matrix(1.0I,K^2,K^2) +Kk)*kron(A0inv,Matrix(1.0I,K,K) )*L')
    D = duplication(K)
    Dplus = (D'*D)\D'
    SIGsig = 2*Dplus*kron(V.Σ,V.Σ)*Dplus'
    Cbar0 = kron(Matrix(1.0I,K,K) ,J*Matrix(1.0I,K*V.p,K*V.p) *J')*Hk
    STD[:,1] = vec((reshape(diag(real(sqrt.(complex(Cbar0*SIGsig*Cbar0'/(T-V.p))))),K,K))')
    COV2[:,1] = vec((reshape(diag((Cbar0*SIGsig*Cbar0'/(T-V.p))),K,K))')
    for h=1:H
        Gi = zeros(K^2,K^2*V.p)
        for m=0:(h-1)
            Gi += kron(J*(A')^(h-1-m),J*(A^m)*J')
        end
        C = kron(A0inv',Matrix(1.0I,K,K) )*Gi
        Cbar = kron(Matrix(1.0I,K,K) ,J*A^h*J')*Hk
        STD[:,h+1] = vec((reshape(diag(real(sqrt.(complex(C*SIGa*C'+Cbar*SIGsig*Cbar')/(T-V.p)))),K,K))')
        COV2[:,h+1] = vec((reshape(diag(((Cbar*SIGsig*Cbar')/(T-V.p))),K,K))')
    end
    return STD,COV2
end

function irf_ci_asymptotic(V::VAR, H::Int64, inter::Intercept)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    SIGa = kron(inv(V.X*V.X'/(T-V.p)),V.Σ)
    SIGa = SIGa[K+1:end,K+1:end]
    # Calculation of stdev follows Lutkepohl(2005) p.111,93
    A = get_VAR1_rep(V,V.inter)
    A0inv = cholesky(V.Σ).L
    STD   = zeros(K^2,H+1)
    COV2   = zeros(K^2,H+1)
    J = [Matrix(1.0I,K,K)  zeros(K,K*(V.p-1))]
    L = elimat(K)
    Kk = commutation(K,K)
    Hk = L'/(L*(Matrix(1.0I,K^2,K^2) +Kk)*kron(A0inv,Matrix(1.0I,K,K) )*L')
    D = duplication(K)
    Dplus = (D'*D)\D'
    SIGsig = 2*Dplus*kron(V.Σ,V.Σ)*Dplus'
    Cbar0 = kron(Matrix(1.0I,K,K) ,J*Matrix(1.0I,K*V.p,K*V.p)*J')*Hk
    STD[:,1] = vec((reshape(diag(real(sqrt.(complex(Cbar0*SIGsig*Cbar0'/(T-V.p))))),K,K))')
    COV2[:,1] = vec((reshape(diag((Cbar0*SIGsig*Cbar0'/(T-V.p))),K,K))')
    for h=1:H
        Gi = zeros(K^2,K^2*V.p)
        for m=0:(h-1)
            Gi += kron(J*(A')^(h-1-m),J*(A^m)*J')
        end
        C = kron(A0inv',Matrix(1.0I,K,K) )*Gi
        Cbar = kron(Matrix(1.0I,K,K) ,J*A^h*J')*Hk
        STD[:,h+1] = vec((reshape(diag(real(sqrt.(complex(C*SIGa*C'+Cbar*SIGsig*Cbar')/(T-V.p)))),K,K))')
        COV2[:,h+1] = vec((reshape(diag(((Cbar*SIGsig*Cbar')/(T-V.p))),K,K))')
    end
    return STD,COV2
end

get_boot_init_int_draw(T::Int64,p::Int64) = Int64(trunc.((T-p+1)*rand()+1))
get_boot_init_vector_draw(T::Int64,p::Int64) = Array{Int64}(trunc.((T-p)*rand(T-p) .+ 1))

function build_sample(V::VAR)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    # Draw block of initial pre-sample values
    y = zeros(K,T)                             # bootstrap data
    u = zeros(K,T)                             # bootstrap innovations
    iDraw = get_boot_init_int_draw(T,V.p)      # position of initial draw
    y[:,1:V.p] = V.Y[:,iDraw:iDraw+V.p-1]      # drawing pre-sample obs
    # Draw innovations
    vDraw = get_boot_init_vector_draw(T,V.p)   # index for innovation draws
    u[:, V.p+1:T] = V.ϵ[:,vDraw]               # drawing innovations
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
    y = zeros(K,T)                             # bootstrap data
    u = zeros(K,T)                             # bootstrap innovations
    iDraw = get_boot_init_int_draw(T,V.p)      # position of initial draw
    y[:,1:V.p] = V.Y[:,iDraw:iDraw+V.p-1]      # drawing pre-sample obs
    # Draw innovations
    vDraw = get_boot_init_vector_draw(T,V.p)   # index for innovation draws
    u[:, V.p+1:T] = V.ϵ[:,vDraw]               # drawing innovations
    @inbounds for i = V.p+1:T
        y[:,i] = V.β[:,1] + u[:,i]
        for j =  1:V.p
            y[:,i] += V.β[:,(j-1)*K + 2:j*K+1]*y[:,i-j]
        end
    end
    return y
end

col_mean(x::AbstractArray) = mean(x, dims = 2)
test_bias_correction(x::AbstractArray) =  any(abs.(eigvals(x)).<1)

function get_boot_ci(V::VAR,H::Int64,nrep::Int64, bDo_bias_corr::Bool,inter::Intercept)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    mIRFbc = zeros(nrep, K^2*(H+1))
    @inbounds for j = 1:nrep
        # Recursively construct sample
        yr = build_sample(V,V.inter)
        yr = Array((yr .- col_mean(yr))')  # demean yr bootstrap data
        #pr = V.p # also using lag length selection
        Vr = VAR(yr,V.p,true)
        # Bias correction: if the largest root of the companion matrix
        # is less than 1, do BIAS correction
        mVar1 = get_VAR1_rep(Vr,V.inter)::AbstractArray{Float64,2}
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
        yr = Array((yr .- col_mean(yr))')  # demean yr bootstrap data
        #pr = V.p # also using lag length selection
        Vr = VAR(yr,V.p,false)
        # Bias correction: if the largest root of the companion matrix
        # is less than 1, do BIAS correction
        mVar1 = get_VAR1_rep(Vr)::AbstractArray{Float64,2}
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
    mSigma = vcat(hcat(V.Σ, zeros(K,K*V.p-K)), zeros(K*V.p-K,K*V.p))::AbstractArray{Float64,2}
    mSigma = convert(Array{Float64,2},mSigma)
end

function get_sigma_y(V::VAR, mVar1::AbstractArray, mSigma::AbstractArray)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    vSigma = (eye((K*V.p)^2) - kron(mVar1,mVar1)) \ vec(mSigma)    # Lutkepohl p.29 (2.1.39)
    mSigma_y = reshape(vSigma, K*V.p::Int64, K*V.p::Int64)
    return convert(Array{Float64,2},mSigma_y)
end

function get_bias(mSigma::AbstractArray,mSigma_y::AbstractArray,B::AbstractArray,I::AbstractArray,mSum_eigen::AbstractArray)
    return mBias = mSigma*(inv(I - B) + B/(I-B^2) + mSum_eigen)/(mSigma_y)
end

function bias_correction(V::VAR,mVar1::AbstractArray)
    # Bias-correction Pope (1990)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    mSigma = get_companion_vcv(V)
    mSigma_y = get_sigma_y(V,mVar1,mSigma)
    II = Matrix(1.0I,K*V.p, K*V.p)
    B = Array(mVar1')
    vEigen = eigvals(mVar1)
    global mSum_eigen = zeros(K*V.p,K*V.p)
    for h = 1:K*V.p
        mSum_eigen += vEigen[h].\(II - vEigen[h]*B)
    end
    mBias = get_bias(mSigma,mSigma_y,B,II,mSum_eigen)
    mAbias = -mBias/T
    return mBcA = get_proportional_bias_corr(mVar1,mAbias)
end

function get_proportional_bias_corr(mVar1::AbstractArray,Abias::AbstractArray;iBc_stab::Int64 = 9, iδ::Int64 = 1)
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

function get_boot_conf_interval(mIRFbc::AbstractArray,H::Int64,K::Int64)
    N = size(mIRFbc,2)
    mCILv = zeros(1,N)
    mCIHv = zeros(1,N)
    for i = 1:N
        mCILv[:,i] .= quantile(vec(mIRFbc[:,i]),0.025)
        mCIHv[:,i] .= quantile(vec(mIRFbc[:,i]),0.975)
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
    mCIL, mCIH = get_boot_conf_interval(mIRFbc::AbstractArray,H::Int64,K::Int64)
    return CIs_boot(mCIL,mCIH)
end

function irf_ci_bootstrap(V::VAR, H::Int64, nrep::Int64, inter::Intercept; bDo_bias_corr::Bool=true)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    iScale_ϵ = sqrt((T-V.p)/(T-V.p-K*V.p-1))
    u = V.ϵ*iScale_ϵ   # rescaling residual (Stine, JASA 1987)
    mIRFbc = get_boot_ci(V,H,nrep,bDo_bias_corr,V.inter)
    # Calculate 95 perccent interval endpoints
    mCIL, mCIH = get_boot_conf_interval(mIRFbc::AbstractArray,H::Int64,K::Int64)
    return CIs_boot(mCIL,mCIH)
end

function irf_chol(V::VAR, mVar1::AbstractArray, H::Int64)
    K = size(V.Σ,1)
    mSigma = cholesky(V.Σ).L::LowerTriangular  # Cholesky or reduced form
    J = [eye(K,K) zeros(K,K*(V.p-1))]
    mIRF = zeros(K^2,H+1)
    mIRF[:,1] = reshape((J*mVar1^0*J'*mSigma)',K^2,1)
    for i = 1:H
        mIRF[:,i+1] = reshape((J*mVar1^i*J'*mSigma)',K^2,1)
    end
    return mIRF
end

function irf_reduce_form(V::VAR, mVar1::AbstractArray, H::Int64)
    K = size(V.Σ,1)
    mSigma = Matrix(1.0I,K,K)
    J = [Matrix(1.0I,K,K) zeros(K,K*(V.p-1))]
    mIRF = zeros(K^2,H+1)
    mIRF[:,1] = reshape((J*mVar1^0*J'*mSigma)',K^2,1)
    for i = 1:H
        mIRF[:,i+1] = reshape((J*mVar1^i*J'*mSigma)',K^2,1)
    end
    return mIRF
end

function irf_ext_instrument(V::VAR,Z::AbstractArray,H::Int64,intercept::Bool)
    # Version improved by G. Ragusa
    y,B,Σ,U,p = V.Y',V.β,V.Σ,V.ϵ,V.p
    (T,K) = size(y)
    (T_z, K_z) = size(Z)
    ZZ = Z[p+1:end,:]
    ΣηZ = U[:,T-T_z+p+1:end]*ZZ
    Ση₁Z = ΣηZ[1:1,:]
    H1 = ΣηZ*Ση₁Z'./(Ση₁Z*Ση₁Z')
    A0inv = [H1 zeros(K,K-1)]
    A = [B[:,2:end];[Matrix(1.0I,K*(p-1),K*(p-1)) zeros(K*(p-1),K)]]
    J = [Matrix(1.0I,K,K) zeros(K,K*(p-1))]
    IRF = GrowableArray(A0inv[:,1])
    #HD = GrowableArray(zeros(K,K))
    for h in 1:H
        C = J*A^h*J'    
        push!(IRF, (C*A0inv)[:,1])    
    end
    return IRF
end

function irf_ext_instrument(V::VAR,Z::Vector{Float64},H::Int64,intercept::Bool)
    # Version improved by G. Ragusa
    y,B,Σ,U,p = V.Y',V.β,V.Σ,V.ϵ,V.p
    (T,K) = size(y)
    T_z = size(Z)
    ZZ = Z[p+1:end,:]
    ΣηZ = U[:,T-T_z+p+1:end]*ZZ
    Ση₁Z = ΣηZ[1:1,:]
    H1 = ΣηZ*Ση₁Z'./(Ση₁Z*Ση₁Z')
    A0inv = [H1 zeros(K,K-1)]
    A = [B[:,2:end];[Matrix(1.0I,K*(p-1),K*(p-1)) zeros(K*(p-1),K)]]
    J = [Matrix(1.0I,K,K) zeros(K,K*(p-1))]
    IRF = GrowableArray(A0inv[:,1])
    #HD = GrowableArray(zeros(K,K))
    for h in 1:H
        C = J*A^h*J'    
        push!(IRF, (C*A0inv)[:,1])    
    end
    return IRF
end

function irf_ci_wild_bootstrap(V::VAR,Z::AbstractArray,H::Int64,nrep::Int64,α::AbstractArray,intercept::Bool)
    # Wild Bootstrap
    # Version improved by G. Ragusa
    y,Y,A,u,p = V.mData,V.Y',V.β,V.ϵ,V.p
    count = 1
    (T,K) = size(y)
    (T_z, K_z) = size(Z)
    IRFS = GrowableArray(Matrix{Float64}(H+1, K))
    CILv = Array{Float64}(undef, (length(α), size(IRFS,2)))
    CIHv = similar(CILv)
    lower_bound = Array{Float64}(undef, (H+1, length(α)))
    upper_bound = similar(lower_bound)
    res = u' 
    oneK = ones(1,K)
    oneKz = ones(1,K_z)
    varsb = zeros(T,K)
    if intercept == true
        Awc = A[:,2:end]
        Ac  = A[:,1]
    else
        Awc = A
        Ac = zeros(A[:,1])
    end
    rr  = Array{Int16}(undef, T)
    while count < nrep+1
        rand!(rr, [-1, 1])        
        resb = res.*rr[p+1:end]
        #  Deterministic initial values
        for j in 1:K
            for i in 1:p
                @inbounds varsb[i,j] = y[i,j]
            end
        end
        @inbounds for j = p+1:T
            lvars = transpose(varsb[j-1:-1:j-p,:])        
            varsb[j,:] = Awc*vec(lvars) + Ac + resb[j-p,:];
        end
        proxies = Z.*rr[T-T_z+1:end]
        Vr = VAR(varsb,V.p,true)
        IRFr = irf_ext_instrument(Vr,proxies,H,intercept)
        #IRFrmat[count, :, :] = vec(IRFr')'
        push!(IRFS, IRFr)
        count += 1
    end
    CIl = Array{Float64}(undef, (length(α), H+1, K))
    CIh = similar(CIl)
    for i in 1:K
        # FIX THIS POINT--AT THE MOMENT ONLY FIRST VARIABLE CI
        lower = mapslices(u->quantile(u, α./2), IRFS[2:end,:,i],dims=1)'
        upper = mapslices(u->quantile(u, 1 .- α./2), IRFS[2:end,:,i],dims=1)' 
        CIl[:,:,i] = lower'
        CIh[:,:,i] = upper'
    end
    return CIs_boot(CIl, CIh)    
end

function t_test(V::VAR)
    K = size(V.Σ,1)
    Kp = size(V.X,1)
    H = kron(inv(V.X*V.X'),V.Σ)
    SE = sqrt.(diag(H))
    T = reshape(vec(V.β)./SE,K,Kp)
    return T
end

function gen_var1_data!(y::AbstractArray,mR::AbstractArray,mP,burnin::Int64)
    T,K = size(y)
    for j = 2:T                          
        y[j,:] = B*y[j-1,:] + Σ*randn(K,1)   
    end
    y = y[burnin+1:end,:]                      
    return y .- mean(y, dims = 1)
end

export VAR, IRFs_a, IRFs_b, IRFs_ext_instrument, IRFs_localprojection, gen_var1_data!
export lp_lagorder,var_lagorder, irf_ci_asymptotic

end # end of the module