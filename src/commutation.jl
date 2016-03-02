# returns Magnus and Neudecker's commutation matrix of dimensions n by m
function commutation(n::Int64, m::Int64)
  k = reshape(kron(vec(eye(n)), eye(m)), n*m, n*m)
  return k
end
