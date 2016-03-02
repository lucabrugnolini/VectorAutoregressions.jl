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
