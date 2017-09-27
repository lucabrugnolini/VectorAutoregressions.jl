module VARs

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


# returns Magnus and Neudecker's commutation matrix of dimensions n by m
function commutation(n::Int64, m::Int64)
    k = reshape(kron(vec(eye(n)), eye(m)), n*m, n*m)
    return k
end

# Returns Magnus and Neudecker's duplication matrix of size n
# VERY AMBIGUOUS FUNC
function duplication(n::Int64)
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

# elimat(m) returns the elimination matrix Lm
# The elimination matrix Lm is for any matrix F, Vech(F)=Lm Vec(F)
function elimat(m::Int64)
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
    A0inv = full(cholfact(V.Σ,:L))
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
    A0inv = full(cholfact(V.Σ,:L))
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

# Bias-correction Pope (1990)
function bias_correction(V::VAR,mVar1::Array)
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
    mSigma = full(cholfact(V.Σ,:L))::Array  # Cholesky or reduced form
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

export VAR, IRFs_a, IRFs_b
end # end of the module

# Example VAR K=2 T=100
using VARs, Plots

const T,K = 100,2
const H = 24
const nrep = 1000
const p = 2
const intercept = true

y = rand(T,K)
V = VAR(y,p,intercept)
mIRFb = IRFs_b(V,H,nrep,intercept)
mIRFa = IRFs_a(V,H,intercept)

pIRF = plot(layout = grid(K,K))
[plot!(pIRF, [mIRFb.CI.CIl[i,:] mIRFb.IRF[i,:] mIRFb.CI.CIh[i,:]], color = ["red" "black" "red"],
line = [:dot :solid :dot], legend = false, subplot = i) for i in 1:K^2]
gui(pIRF)
















data = rand(100,4)
#p = get_VAR_lag_length(data,12,"aic",true) # select lag-length using aic, bic, hq, aicc
V = VAR(data,2,false) # fit a VAR model to data, true unclude intercept
irf = irf(V,20,true) # get impulse response function, true for cholesky
(CIl, CIh) = irf_ci_bootstrap(V,20,1000, true) # true for bias correction if eig<1
(STD, COV) = irf_ci_asymptotic(V, 20)





# Plot example
using PyPlot
subplot(221); plot(irf[1,:]'); hold(true);
plot(CIl[1,:]', color = "red"); hold(true);plot(CIh[1,:]', color = "red"); hold(true);
plot((irf[1,:]'+1.96*STD[1,:]'), color = "black"); hold(true); plot((irf[1,:]'-1.96*STD[1,:]'), color = "black");
title("shock 1, variable 1")
subplot(222); plot(irf[2,:]'); hold(true);
plot(CIl[2,:]', color = "red"); hold(true); plot(CIh[2,:]', color = "red"); hold(true);
plot((irf[2,:]'+1.96*STD[2,:]'), color = "black"); hold(true); plot((irf[2,:]'-1.96*STD[2,:]'), color = "black");
title("shock 2, variable 1")
subplot(223); plot(irf[3,:]'); hold(true);
plot(CIl[3,:]', color = "red"); hold(true); plot(CIh[3,:]', color = "red"); hold(true);
plot((irf[3,:]'+1.96*STD[3,:]'), color = "black"); hold(true); plot((irf[3,:]'-1.96*STD[3,:]'), color = "black");
title("shock 3, variable 1")
subplot(224); plot(irf[4,:]'); hold(true);
plot(CIl[4,:]', color = "red"); hold(true); plot(CIh[4,:]', color = "red"); hold(true);
plot((irf[4,:]'+1.96*STD[4,:]'), color = "black"); hold(true); plot((irf[4,:]'-1.96*STD[4,:]'), color = "black");
title("shock 4, variable 1")
suptitle("IRF")
