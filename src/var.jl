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

function fit(y::Array,p::Int64, i::Bool=true)
  (T,K) = size(y)
  Y = (y[p+1:T,:])'
  X = ones(1,T-p)
  for j=1:p
      X = [X; y[p+1-j:T-j,:]']
  end
  i == true ? X = X : X = X[2:end,:]
  β = (Y*X')/(X*X')
  ϵ = Y - β*X
  Σ = ϵ*ϵ'/(T-p-p*K-1)
  return Y,X,β,ϵ,Σ,p
end

include("t_test.jl")
include("get_VAR1_rep.jl")
include("get_VAR_lag_length.jl")
include("irf.jl")
include("irf_ci_bootstrap.jl")
include("bias_correction.jl")
include("irf_ci_asymptotic.jl")
include("commutation.jl")
include("duplication.jl")
include("elimat.jl")
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
