# Bias-correction Pope (1990)
function bias_correction(V::VAR)
  (T,K) = size(V.Y')
  A = get_VAR1_rep(V)
  SIGMA = [V.Σ zeros(K,K*V.p-K); zeros(K*V.p-K,K*V.p)]
  vecSIGMAY = (eye((K*V.p)^2)-kron(A,A))\vec(SIGMA)    # Lutkepohl p.29 (2.1.39)
  SIGMAY = reshape(vecSIGMAY, K*V.p, K*V.p)
  I = eye(K*V.p, K*V.p)
  B = A'
  peigen = eigvals(A)
  sumeig = zeros(K*V.p, K*V.p)
  for h in 1:K*V.p
    sumeig += (peigen[h].\(I - peigen[h]*B)) #***
  end
  bias = SIGMA*(inv(I - B) + B/(I-B^2) + sumeig)/(SIGMAY) #***
  Abias = - bias/T
  bcstab = 9           # Arbitrary default value > 1
  delta = 1            # Adjustment factor
  while bcstab >= 1
    # Adjust bias-correction proportionately
    global bcA = A-delta*Abias
    bcmod = abs(eigvals(bcA))
    any(bcmod.>= 1) ?  bcstab = 1 : bcstab = 0
    delta += - 0.01
    (delta <= 0) && (bcstab = 0)
  end
  return bcA
end
