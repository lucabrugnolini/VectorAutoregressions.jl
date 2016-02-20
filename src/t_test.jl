function t_test(V::VAR)
  K = size(V.Σ,1)
  Kp = size(V.X,1)
  H = kron(inv(V.X*V.X'),V.Σ)
  SE = sqrt(diag(H))
  T = reshape(vec(V.β)./SE,K,Kp)
  return T
end
