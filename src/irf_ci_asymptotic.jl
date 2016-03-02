function irf_ci_asymptotic(V::VAR, H)
    (T,K) = size(V.Y')
    if V.i == true
        SIGa = kron(inv(V.X*V.X'/(T-V.p)),V.Σ)
        SIGa = SIGa[K+1:end,K+1:end]
    else
        SIGa = kron(inv(V.X*V.X'/(T-V.p)),V.Σ)
    end
    # Calculation of stdev follows Lutkepohl(2005) p.111,93
    A = get_VAR1_rep(V)
    A0inv = full(chol(V.Σ)')
    STD   = zeros(K^2,H+1)
    COV2   = zeros(K^2,H+1)
    J = [eye(K) zeros(K,K*(V.p-1))]
    L = elimat(K)
    Kk = commutation(K,K)
    Hk = L'/(L*(eye(K^2)+Kk)*kron(A0inv,eye(K))*L')
    D = duplication(K)
    Dplus = (D'*D)\D'
    SIGsig = 2*Dplus*kron(V.Σ,V.Σ)*Dplus';
    Cbar0 = kron(eye(K),J*eye(K*V.p)*J')*Hk;
    STD[:,1] = vec((reshape(diag(real(sqrt(complex(Cbar0*SIGsig*Cbar0'/(T-V.p))))),K,K))')
    COV2[:,1] = vec((reshape(diag((Cbar0*SIGsig*Cbar0'/(T-V.p))),K,K))');
    for h=1:H
        Gi = zeros(K^2,K^2*V.p)
        for m=0:(h-1)
            Gi += kron(J*(A')^(h-1-m),J*(A^m)*J')
        end
        C = kron(A0inv',eye(K))*Gi
        Cbar = kron(eye(K),J*A^h*J')*Hk
        STD[:,h+1] = vec((reshape(diag(real(sqrt(complex(C*SIGa*C'+Cbar*SIGsig*Cbar')/(T-V.p)))),K,K))')
        COV2[:,h+1] = vec((reshape(diag(((Cbar*SIGsig*Cbar')/(T-V.p))),K,K))')
    end
    return STD,COV2
end
