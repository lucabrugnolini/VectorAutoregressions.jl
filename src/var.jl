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
export VAR, t_test, get_VAR1_rep
end # end of the module

# Example
V = VAR(rand(100,4),2,true)
T = t_test(V)
get_VAR1_rep(V)
