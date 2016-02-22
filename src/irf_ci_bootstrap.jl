function irf_ci_bootstrap(V::VAR, H::Int64, nrep::Int64)

  V.Y = V.Y'
  (T,K) = size(V.Y)
  IRFrmat = zeros(nrep, K^2*(H+1))
  u = V.ϵ*sqrt((T-V.p)/(T-V.p-K*V.p-1))          # rescaling residual (Stine, JASA 1987)
  for j in 1:nrep
    # Draw block of initial pre-sample values
    yr = zeros(K,T)                                       # bootstrap data
    pos = trunc(Integer, (rand()*(T-V.p+1)))+1            # position of initial draw
    yr[:,1:V.p] = V.Y[pos:pos+V.p-1,:]'                   # drawing pre-sample obs
    # Draw innovations
    ur = zeros(K,T)                             # bootstrap innovations
    index = trunc(Integer, (rand(T-V.p)*(T-V.p))+1)         # index for innovation draws
    ur[:, V.p+1:T] = u[:, index]                  # drawing innovations
    # Recursively construct sample
    for i in (V.p+1):T
      yr[:,i] = V.β[:,1] + ur[:,i]
      for jj in 1:V.p
        yr[:,i] += V.β[:,(jj-1)*K + 2:jj*K+1]*yr[:,i-jj]
      end
    end
    yr = yr'
    for i in 1:K                                   # demean yr bootstrap data
        yr[:,i] -= mean(yr[:,i])
    end
    #pr = V.p # also using lag length selection
    Vr = VAR(yr,V.p,true)
    #(A1r,SIGMA1r) = VAR(yr,pr, i=true)
    Ar = get_VAR1_rep(Vr)
    #Ar = [A1r[:,2:end]; [eye(K*(pr-1)) zeros(K*(pr-1),K)]]   # Slope parameter estimates in companion form
    SIGMAr = [Vr.Σ zeros(K,K*Vr.p-K); zeros(K*Vr.p-K,K*Vr.p)]   # variance of residuals in companion form
    # Bias correction: if the largest root of the companion matrix
    # is less than 1, do BIAS correction
    #if ~ any(abs(eigvals(Ar)).>=1)
    #  Ar = asybc(Ar,SIGMAr,T,K,Vr.p) #############################CHECK
    #end
    Ar = real(Ar)
    IRFr = irf(Ar[1:K,1:K*Vr.p], SIGMAr[1:K,1:K], H) ##############PROBLEM HERE
    IRFrmat[j,:] = vec(IRFr')'
  end                              # end bootstrap
  # Calculate 95 perccent interval endpoints
  CILv = zeros(1,size(IRFrmat,2))
  CIHv = zeros(1,size(IRFrmat,2))
  for i = 1:size(IRFrmat,2)
    CILv[:,i] = quantile(vec(IRFrmat[:,i]),0.025)
    CIHv[:,i] = quantile(vec(IRFrmat[:,i]),0.975)
  end
  CI = [CILv; CIHv]
  return CI
end
