module VAR

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

function t_test(V::VAR)
  K = size(V.Σ,1)
  Kp = size(V.X,1)
  H = kron(inv(V.X*V.X'),V.Σ)
  SE = sqrt(diag(H))
  T = reshape(vec(V.β)./SE,K,Kp)
  return T
end

function get_VAR1_rep(V::VAR)
    K = size(V.Σ,1)
    if V.i == true
       v = [V.β[:,1]; zeros(K*(V.p-1),1)]; B = [V.β[:,2:end,]; eye(K*(V.p-1)) zeros(K*(V.p-1),K)]
    else B = [V.β; eye(K*(V.p-1)) zeros(K*(V.p-1),K)]
    end
    V.i == true ? (return B, v) : (return B)
end

# Example
V = VAR(rand(100,4),2,true)
T = t_test(V)
get_VAR1_rep(V)
