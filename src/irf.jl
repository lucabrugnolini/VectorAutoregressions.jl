function irf(V::VAR, H::Int64, cholesky::Bool=true)
         K = size(V.Σ,1)
         cholesky == true ? Sigma = chol(V.Σ)' : Sigma = eye(K,K) # Cholesky or reduced form
         B0 = get_VAR1_rep(V)
         J = [eye(K,K) zeros(K,K*(V.p-1))]
         IRF = reshape((J*B0^0*J'*Sigma)',K^2,1)
         for i = 1:H
           IRF = [IRF reshape((J*B0^i*J'*Sigma)',K^2,1)]
         end
         return IRF
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
