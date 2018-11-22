# This package is a work in progress for the estimation and identification of Vector Autoregressive (VAR) models.
# Credits:
# Kilian and Kim 2011, Cremfi codes, Geertler and Karadi 2015

module VectorAutoregressions
using Parameters, GrowableArrays

type Intercept end

type VAR
    mData::Array
    Y::Array
    X::Array
    β::Array
    ϵ::Array
    Σ::Array
    p::Int64
    inter::Intercept
    VAR(mData,Y,X,β,ϵ,Σ,p,inter) = p <= 0 ? error("Lag-length error: 'p' must be strictly positive") : new(mData,Y,X,β,ϵ,Σ,p,inter)
end

function VAR(y::Array,p::Int64,i::Bool)
    i == false ? ((mData,Y,X,β,ϵ,Σ,p) = fit(y,p)) : ((mData,Y,X,β,ϵ,Σ,p) = fit(y,p,Intercept()))
    return VAR(mData,Y,X,β,ϵ,Σ,p,Intercept())
end

abstract type CIs end

type IRFs
    IRF::Array
    CI::CIs
end


type CIs_asy <: CIs
    CIl::Array
    CIh::Array
end

type CIs_boot <: CIs
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

function IRFs_ext_instrument(V::VAR,Z::Array,H::Int64,nrep::Int64, α::Array, intercept::Bool)
    mIRF = irf_ext_instrument(V, Z, H, intercept)
    CI = irf_ci_wild_bootstrap(V, Z, H, nrep, α, intercept)
    return IRFs(mIRF,CI)
end

function IRFs_localprojection(z::Array{Float64}, paic::Array{Float64}, H::Int64, A0inv::Array{Float64},cov_Σ::Array{Float64})
    T,K = size(z)
    A0 = vec(A0inv')      # IRF for Horizon 0 --> use auxiliary model for identification
    cov_A0 = zeros(K^2,1)   
    for h = 1:H           # IRF for Horizon 1~H
        ph = paic[h]                     # lag-order for horizon h
        ys = z[ph+h:T,:]   
        yt = z[ph:T-h,:]                 # RHS variable of interest: y(t)
        x = ones(T-ph-h+1,1)             # constant term
        for i = 2:ph
            x = [x z[ph+1-i:T-i-h+1,:]]  # other RHS lags: y(t-1)~y(t-p+1)
        end
        Mx = eye(size(x,1))-x/(x'*x)*x'        # annhilation matrix
        β = (yt'*Mx*yt)\(yt'*Mx*ys)            # IRF by Local Projection
        A0 = [A0 vec(β)]                       # reduced form IRF
        
        Σ_u = newey_west(ys,yt,Mx,β)
        
        invytMxyt  = inv(yt'*Mx*yt)
        Σ_β = kron(Σ_u, invytMxyt)                               # var(vec(β))
        Σ_A0 = kron(eye(K), A0inv')*Σ_β*kron(eye(K), A0inv')'    # var(vec(A0inv*β))
        cov_A0 = [cov_A0 reshape(diag(Σ_A0), K^2, 1)]
    end
    sd_A0 = sqrt(cov_A0+cov_Σ)    
    return A0, sd_A0
end

function IRFs_localprojection(z::Array{Float64}, paic::Array{Int64}, H::Int64)
    T,K = size(z)
    A0 = vec(eye(K)) # IRF for Horizon 0 --> reduce form
    cov_A0 = zeros(K^2,1)   
    for h = 1:H      # IRF for Horizon 1~H
        ph = paic[h]                     # lag-order for horizon h
        ys = z[ph+h:T,:]   
        yt = z[ph:T-h,:]                 # RHS variable of interest: y(t)
        x = ones(T-ph-h+1,1)             # constant term
        for i = 2:ph
            x = [x z[ph+1-i:T-i-h+1,:]]  # other RHS lags: y(t-1)~y(t-p+1)
        end
        Mx = eye(size(x,1))-x/(x'*x)*x'        # annhilation matrix
        β = (yt'*Mx*yt)\(yt'*Mx*ys)            # IRF by Local Projection
        A0 = [A0 vec(β)]                       # reduced form IRF
        
        Σ_u = newey_west(ys,yt,Mx,β)
        
        invytMxyt  = inv(yt'*Mx*yt)
        Σ_β = kron(Σ_u, invytMxyt)                               # var(vec(β))
        Σ_A0 = kron(eye(K), A0inv')*Σ_β*kron(eye(K), A0inv')'    # var(vec(A0inv*β))
        cov_A0 = [cov_A0 reshape(diag(Σ_A0), K^2, 1)]
    end
    sd_A0 = sqrt(cov_A0)    
    return A0, sd_A0
end

function IRFs_localprojection(z::Array{Float64}, paic::Int64, H::Int64, A0inv::Array{Float64},cov_Σ::Array{Float64})
    T,K = size(z)
    A0 = vec(A0inv')      # IRF for Horizon 0 --> use auxiliary model for identification
    cov_A0 = zeros(K^2,1)   
    for h = 1:H           # IRF for Horizon 1~H
        ph = paic                     # lag-order for horizon h
        ys = z[ph+h:T,:]   
        yt = z[ph:T-h,:]                 # RHS variable of interest: y(t)
        x = ones(T-ph-h+1,1)             # constant term
        for i = 2:ph
            x = [x z[ph+1-i:T-i-h+1,:]]  # other RHS lags: y(t-1)~y(t-p+1)
        end
        Mx = eye(size(x,1))-x/(x'*x)*x'        # annhilation matrix
        β = (yt'*Mx*yt)\(yt'*Mx*ys)            # IRF by Local Projection
        A0 = [A0 vec(β)]                       # reduced form IRF
        
        Σ_u = newey_west(ys,yt,Mx,β)
        
        invytMxyt  = inv(yt'*Mx*yt)
        Σ_β = kron(Σ_u, invytMxyt)                               # var(vec(β))
        Σ_A0 = kron(eye(K), A0inv')*Σ_β*kron(eye(K), A0inv')'    # var(vec(A0inv*β))
        cov_A0 = [cov_A0 reshape(diag(Σ_A0), K^2, 1)]
    end
    sd_A0 = sqrt(cov_A0+cov_Σ)    
    return A0, sd_A0
end

function IRFs_localprojection(z::Array{Float64}, paic::Int64, H::Int64)
    T,K = size(z)
    A0 = vec(eye(K)) # IRF for Horizon 0 --> reduce form
    cov_A0 = zeros(K^2,1)                            
    for h = 1:H     # IRF for Horizon 1~H
        ph = paic                     # lag-order for horizon h
        ys = z[ph+h:T,:]   
        yt = z[ph:T-h,:]                 # RHS variable of interest: y(t)
        x = ones(T-ph-h+1,1)             # constant term
        for i = 2:ph
            x = [x z[ph+1-i:T-i-h+1,:]]  # other RHS lags: y(t-1)~y(t-p+1)
        end
        Mx = eye(size(x,1))-x/(x'*x)*x'        # annhilation matrix
        β = (yt'*Mx*yt)\(yt'*Mx*ys)            # IRF by Local Projection
        A0 = [A0 vec(β)]                       # reduced form IRF
        
        Σ_u = newey_west(ys,yt,Mx,β)
        
        invytMxyt  = inv(yt'*Mx*yt)
        Σ_β = kron(Σ_u, invytMxyt)                               # var(vec(β))
        cov_A0 = [cov_A0 reshape(diag(Σ_β), K^2, 1)]
    end
    sd_A0 = sqrt(cov_A0)    
    return A0, sd_A0
end

function newey_west(ys,yt,Mx,β)
    T,K = size(yt)
    u = Mx*ys - Mx*yt*β    # residual from LPs
    μ_u = zeros(1,K) 
    for i = 1:K 
        μ_u[:,i] = mean(u[:,i])
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
    return y',Y,X,β,ϵ,Σ,p
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
    return y',Y,X,β,ϵ,Σ,p
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
    return CIs_boot(mCIL,mCIH)
end

function irf_ci_bootstrap(V::VAR, H::Int64, nrep::Int64, inter::Intercept; bDo_bias_corr::Bool=true)
    K,T = size(V.Y)::Tuple{Int64,Int64}
    iScale_ϵ = sqrt((T-V.p)/(T-V.p-K*V.p-1))
    u = V.ϵ*iScale_ϵ   # rescaling residual (Stine, JASA 1987)
    mIRFbc = get_boot_ci(V,H,nrep,bDo_bias_corr,V.inter)
    # Calculate 95 perccent interval endpoints
    mCIL, mCIH = get_boot_conf_interval(mIRFbc::Array,H::Int64,K::Int64)
    return CIs_boot(mCIL,mCIH)
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

function irf_ext_instrument(V::VAR,Z::Array,H::Int64,intercept::Bool)
    # Version improved by G. Ragusa
    y,B,Σ,U,p = V.Y',V.β,V.Σ,V.ϵ,V.p
    (T,K) = size(y)
    (T_z, K_z) = size(Z)
    ZZ = Z[p+1:end,:]
    ΣηZ = U[:,T-T_z+p+1:end]*ZZ
    Ση₁Z = ΣηZ[1:1,:]
    H1 = ΣηZ*Ση₁Z'./(Ση₁Z*Ση₁Z')
    A0inv = [H1 zeros(K,K-1)]
    A = [B[:,2:end];[eye(K*(p-1)) zeros(K*(p-1),K)]]
    J = [eye(K,K) zeros(K,K*(p-1))]
    IRF = GrowableArray(A0inv[:,1])
    #HD = GrowableArray(zeros(K,K))
    for h in 1:H
        C = J*A^h*J'    
        push!(IRF, (C*A0inv)[:,1])    
    end
    return IRF
end

function irf_ci_wild_bootstrap(V::VAR,Z::Array,H::Int64,nrep::Int64,α::Array,intercept::Bool)
    # Wild Bootstrap
    # Version improved by G. Ragusa
    y,Y,A,u,p = V.mData,V.Y',V.β,V.ϵ,V.p
    count = 1
    (T,K) = size(y)
    (T_z, K_z) = size(Z)
    IRFS = GrowableArray(Matrix{Float64}(H+1, K))
    CILv = Array{Float64}(length(α), size(IRFS,2))
    CIHv = similar(CILv)
    lower_bound = Array{Float64}(H+1, length(α))
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
    rr  = Array{Int16}(T)
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
    CIl = Array{Float64}(length(α), H+1,K)
    CIh = similar(CIl)
    for i in 1:K
        # FIX THIS POINT--AT THE MOMENT ONLY FIRST VARIABLE CI
        lower = mapslices(u->quantile(u, α./2), IRFS[2:end,:,i],1)'
        upper = mapslices(u->quantile(u, 1-α./2), IRFS[2:end,:,i],1)' 
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

function gen_var1_data!(y::Array,mR::Array,mP,burnin::Int64)
    T,K = size(y)
    for j = 2:T                          
        y[j,:] = mR*y[j-1,:] + mP*randn(K,1)   
    end
    y = y[burnin+1:end,:]                      
    return y .- mean(y,1)
end

export VAR, IRFs_a, IRFs_b, IRFs_ext_instrument, gen_var1_data!

end # end of the module



(irfv, stdv, Phat, B, SIGMA, U, rfirfv, COVsig) = irfvar(y, pv, H) # IRF by VAR
pl = lplagorder(y,pbar,H,lag_length_crit) #pv*ones(H)
# LAG_LENGTHl[n,:] = pl           # lag-order for LP
println("Fitted LP has length ($pl)")
(irfl, stdl, rfirfl) = irflp(y, pl, H, convert(Array,Phat), COVsig)            # IRF by LP
#---BIAS, MSE of IRF ESTIMATES
BIASv = BIASv + (irfv-irftrue)                    # sum of BIAS
BIASl = BIASl + (irfl-irftrue)
MSEv  = MSEv  + (irfv-irftrue).^2                 # sum of squared error
MSEl  = MSEl  + (irfl-irftrue).^2
#---BIAS-CORRECTED BOOTSTRAP INTERVAL FOR VAR: KILLIAN -- for VAR(12)
# translating estimates into the companion form
A = [B[:,2:end];[eye(K*(pv-1)) zeros(K*(pv-1),K)]]  # slope estimate
V = [B[:,1];zeros(K*(pv-1),1)]                     # intercept
SIGMAc = [SIGMA zeros(K,K*pv-K);zeros(K*pv-K,K*pv)]  # sigma
# Bias correction: if the largest root of the companion matrix
# is less than 1, do BIAS correction
eigv = abs(eigvals(A))
if ~any(eigv.>=1)
    A = asybc(A,SIGMAc,T,K,pv)
end
A = real(A)
# Bias-corrected estimate is the bootstrap DGP
CIv = boot([V[1:K,1] A[1:K,1:K*pv]], U, y, pv, H,nrep)
CILv  = reshape(CIv[1,:]',H+1,K^2)'               # lower bound
CIHv  = reshape(CIv[2,:]',H+1,K^2)'               # upper bound
#---BLOCK BOOTSTRAP INTERVAL FOR LOCAL PROJECTIONS
block = 8;                                         # block size
CIl = bcbootlp(y, rfirfl, SIGMA, pl[:], block, H,nrep)    # bootstrap interval
CILl = reshape(CIl[1,:]',H+1,K^2)'                # lower bound
CIHl = reshape(CIl[2,:]',H+1,K^2)'                # upper bound
#---COVERAGE RATES