module VARs

type VAR
    Y::Array
    X::Array
    β::Array
    ϵ::Array
    Σ::Array
    p::Int64
    i::Bool
    VAR(Y,X,β,ϵ,Σ,p,i) = p <= 0 ? error("Lag-length error: 'p' must be strictly positive") : new(Y,X,β,ϵ,Σ,p,i)
end

function VAR(y::Array,p::Int64, i::Bool=true)
    (Y,X,β,ϵ,Σ,p) = fit(y,p,i)
    return VAR(Y,X,β,ϵ,Σ,p,i)
end

lagmatrix(x::Array,p::Int64) = vcat([x[:,p-i+1:end-i] for i = 1:p]...)

function fit(y::Array,p::Int64, i::Bool=true)
    (T,K) = size(y)
    T < K && error("error: there are more covariates than observation")
    y = transpose(y)
    Y = y[:,p+1:T]
    X = y
    if i == true
        X = vcat(ones(1,T-p),lagmatrix(X,p))
    else
        lagmatrix(X,p)
    end
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
function duplication(n::Int64)
    a = tril(ones(n,n))
    i = find(a)
    a[i] = 1:size(i,1)
    a = a + tril(a,-1)'
    j = trunc(Integer, vec(a))
    m = (n*(n+1)/2)
    m = trunc(Integer,m)
    d = zeros(n*n,m)
    for r = 1:size(d,1)
        d[r, j[r]] = 1
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

function get_VAR_lag_length(D::Array, pbar::Integer, ic::String, i::Bool=true)
    IC   = zeros(pbar,1)
    for p = 1:pbar
        i==true ? V = VAR(D,p,true) : V = VAR(D,p,false)
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

function get_VAR1_rep(V::VAR)
    K = size(V.Σ,1)
    if V.i == true
        v = [V.β[:,1]; zeros(K*(V.p-1),1)]; B = [V.β[:,2:end,]; eye(K*(V.p-1)) zeros(K*(V.p-1),K)]
    else B = [V.β; eye(K*(V.p-1)) zeros(K*(V.p-1),K)]
    end
    return B
end

function irf_ci_asymptotic(V::VAR, H)
    (T,K) = size(V.Y')
    if V.i == true
        SIGa = kron(inv(V.X*V.X'/(T-V.p)),V.Σ)
        SIGa = SIGa[K+1:end,K+1:end]
    else
        SIGa = kron(inv(V.X*V.X'/(T-V.p)),V.Σ)
    end
    # Calculation of stdev follows Lutkepohl(2005) p.111,93
    A = get_VAR1_rep(V)
    A0inv = full(chol(V.Σ)')
    STD   = zeros(K^2,H+1)
    COV2   = zeros(K^2,H+1)
    J = [eye(K) zeros(K,K*(V.p-1))]
    L = elimat(K)
    Kk = commutation(K,K)
    Hk = L'/(L*(eye(K^2)+Kk)*kron(A0inv,eye(K))*L')
    D = duplication(K)
    Dplus = (D'*D)\D'
    SIGsig = 2*Dplus*kron(V.Σ,V.Σ)*Dplus';
    Cbar0 = kron(eye(K),J*eye(K*V.p)*J')*Hk;
    STD[:,1] = vec((reshape(diag(real(sqrt(complex(Cbar0*SIGsig*Cbar0'/(T-V.p))))),K,K))')
    COV2[:,1] = vec((reshape(diag((Cbar0*SIGsig*Cbar0'/(T-V.p))),K,K))');
    for h=1:H
        Gi = zeros(K^2,K^2*V.p)
        for m=0:(h-1)
            Gi += kron(J*(A')^(h-1-m),J*(A^m)*J')
        end
        C = kron(A0inv',eye(K))*Gi
        Cbar = kron(eye(K),J*A^h*J')*Hk
        STD[:,h+1] = vec((reshape(diag(real(sqrt(complex(C*SIGa*C'+Cbar*SIGsig*Cbar')/(T-V.p)))),K,K))')
        COV2[:,h+1] = vec((reshape(diag(((Cbar*SIGsig*Cbar')/(T-V.p))),K,K))')
    end
    return STD,COV2
end

get_boot_init_int_draw(T::Int64,p::Int64) = trunc.(Int64,(T-p+1)*rand()+1)
get_boot_init_vector_draw(T::Int64,p::Int64) = trunc.(Int64,(T-p)*rand(T-p)+1)

function build_sample!(V::VAR,y::Array,u::Array)
    (K,T) = size(V.Y)
    if V.i == true
        @inbounds for i = (V.p+1):T
            y[:,i] = V.β[:,1] + u[:,i]
            y[:,i] += [V.β[:,(j-1)*K + 2:j*K+1]*y[:,i-j] for j in 1:V.p][end]
        end
    else
        @inbounds for i = (V.p+1):T
            y[:,i] = u[:,i]
            y[:,i] += [V.β[:,(j-1)*K + 1:j*K]*y[:,i-j] for j = 1:V.p][end]
        end
    end
    return y
end

col_mean(x::Array) = mean(x,2)
test_bias_correction(x::Array) =  any(abs.(eigvals(x)).<1)

function get_boot_ci!(V::VAR,u::Array,mIRFbc::Array,nrep::Int64, bDo_bias_corr::Bool)
    (K,T) = size(V.Y)
    @inbounds for j = 1:nrep
        # Draw block of initial pre-sample values
        yr = zeros(K,T)                                       # bootstrap data
        ur = zeros(K,T)                             # bootstrap innovations
        iDraw = get_boot_init_int_draw(T,V.p)            # position of initial draw
        yr[:,1:V.p] = V.Y[:,iDraw:iDraw+V.p-1]                   # drawing pre-sample obs
        # Draw innovations
        vDraw = get_boot_init_vector_draw(T,V.p)    # index for innovation draws
        ur[:, V.p+1:T] = u[:,vDraw]                  # drawing innovations
        # Recursively construct sample
        build_sample!(V,yr,ur)
        yr = transpose(yr .- col_mean(yr)) # demean yr bootstrap data
        #pr = V.p # also using lag length selection
        Vr = VAR(yr,V.p,true)
        # Bias correction: if the largest root of the companion matrix
        # is less than 1, do BIAS correction
        mVar1 = get_VAR1_rep(Vr)
        bBias_corr_test = test_bias_correction(mV1)
        if all([bDo_bias_corr, bBias_corr_test])
            mVar1 = bias_correction(Vr,mVar1)
        end
        mIRF = irf_chol(Vr,mVar1,H)
        mIRFbc[j,:] = transpose(vec(transpose(mIRF)))
    end                              # end bootstrap
end

function get_companion_vcv(V::VAR)
    (K,T) = size(V.Y)
    mSigma = [V.Σ zeros(K,K*V.p-K); zeros(K*V.p-K,K*V.p)]
end

function get_sigma_y(V::VAR, mVar1::Array, mSigma::Array)
    vSigma = (eye((K*V.p)^2)-kron(mVar1,mVar1))\vec(mSigma)    # Lutkepohl p.29 (2.1.39)
    return mSigma_y = reshape(vSigma, K*V.p, K*V.p)
end

function get_bias(mSigma::Array,mSigma_y::Array,B::Array,I::Array,mSum_eigen::Array)
    return mBias= mSigma*(inv(I - B) + B/(I-B^2) + mSum_eigen)/(mSigma_y)
end

# Bias-correction Pope (1990)
function bias_correction(V::VAR,mVar1::Array)
    (K,Y) = size(V.Y)
    mSigma = get_companion_vcv(V)
    mSigma_y = get_sigma_y(V,mVar1,mSigma)
    I = eye(K*V.p, K*V.p)
    B = mVar1'
    vEigen = eigvals(mVar1)
    mSum_eigen = [vEigen[h].\(I - vEigen[h]*B) for h = 1:K*V.p][end]
    mBias = get_bias(mSigma,mSigma_y,B,I,mSum_eigen)
    mAbias = -bias/T
    mBcA = similar(mVar1)
    return mBcA = get_proportional_bias_corr!(mBcA,mVar1,Abias)
end

function get_proportional_bias_corr!(mBcA::Array,mVar1::Array,Abias::Array;iBc_stab::Int64 = 9, iδ::Int64 = 1)
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
    [mCILv[:,i] = quantile(vec(mIRFbc[:,i]),0.025) for i = 1:N]
    [mCIHv[:,i] = quantile(vec(mIRFbc[:,i]),0.975) for i = 1:N]
    mCIL  = reshape(CILv',H+1,K^2)'
    mCIH  = reshape(CIHv',H+1,K^2)'
    return mCIL, mCIH
end

function irf_ci_bootstrap(V::VAR, H::Int64, nrep::Int64; bDo_bias_corr::Bool=true)
    (K,T) = size(V.Y)
    iScale_ϵ = sqrt((T-V.p)/(T-V.p-K*V.p-1))
    u = V.ϵ*iScale_ϵ   # rescaling residual (Stine, JASA 1987)
    mIRFbc = zeros(nrep, K^2*(H+1))
    get_boot_ci!(V,u,mIRFbc,nrep,bDo_bias_corr)
    # Calculate 95 perccent interval endpoints
    return (mCIL, mCIH) = get_boot_conf_interval(mIRFbc::Array,H::Int64,K::Int64)
end

function irf_chol(V::VAR, mVar1::Array, H::Int64)
    K = size(V.Σ,1)
    mSigma = full(cholfact(V.Σ,:L))  # Cholesky or reduced form
    J = [eye(K,K) zeros(K,K*(V.p-1))]
    mIRF = reshape((J*mVar1^0*J'*mSigma)',K^2,1)
    mIRF = hcat(mIRF, hcat([reshape((J*mVar1^i*J'*mSigma)',K^2,1) for i = 1:H]...))
    return mIRF
end

function irf_reduce_form(V::VAR, mVar1::Array, H::Int64)
    K = size(V.Σ,1)
    mSigma = eye(K,K)
    J = [eye(K,K) zeros(K,K*(V.p-1))]
    mIRF = reshape((J*mVar1^0*J'*mSigma)',K^2,1)
    mIRF = hcat(mIRF, hcat([reshape((J*mVar1^i*J'*mSigma)',K^2,1) for i = 1:H]...))
    return mIRF
end

# Allow to specify a particular shock
function irf(VAR::VAR, H::Int64, shock::Int64, cholesky::Bool=true)
    K = size(V.Σ,1)
    cholesky == true ? Sigma = chol(V.Σ)' : Sigma = eye(K,K)
    abs(shock)>K && error("shock must be between 1 and $K")
    B0 = get_VAR1_rep(V)
    J = [eye(K,K) zeros(K,K*(V.p-1))]
    IRF = reshape((J*B0^0*J'*Sigma)',K^2,1) # before B0^0 why?
    for i = 1:H
        IRF = [IRF reshape((J*B0^i*J'*Sigma)',K^2,1)] #Cholesky here has also the intercept
    end
    IRFs = zeros(K,H+1)
    IRFs[:,:] = IRF[shock:K:(size(IRF,1)-K+shock),:]
    return IRFs
end

function t_test(V::VAR)
    K = size(V.Σ,1)
    Kp = size(V.X,1)
    H = kron(inv(V.X*V.X'),V.Σ)
    SE = sqrt(diag(H))
    T = reshape(vec(V.β)./SE,K,Kp)
    return T
end

export VAR, t_test, get_VAR1_rep, get_VAR_lag_length, irf
export irf_ci_bootstrap, bias_correction, duplication, commutation, elimat, irf_ci_asymptotic
end # end of the module

# Example VAR K=4 T=100
using VARs
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
