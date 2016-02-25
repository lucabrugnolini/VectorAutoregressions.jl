module VARs

# y_t = α + β_1y_t-1 + ... + β_py_t-p + ϵ_t
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
export VAR, t_test, get_VAR1_rep, get_VAR_lag_length, irf, irf_ci_bootstrap, bias_correction
end # end of the module

# Example VAR K=4 T=100
using VARs
data = rand(100,4)
#p = get_VAR_lag_length(data,12,"aic",true) # select lag-length using aic, bic, hq, aicc
V = VAR(data,2,false) # fit a VAR model to data
T = t_test(V) # test coefficient significance
get_VAR1_rep(V) # get companion form
irf(V,20,true) # get impulse response function (reduce_form or cholesky)
bias_correction(V)
irf_ci_bootstrap(V,10,1000)
