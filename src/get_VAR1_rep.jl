function get_VAR1_rep(V::VAR)
    K = size(V.Σ,1)
    if V.i == true
       v = [V.β[:,1]; zeros(K*(V.p-1),1)]; B = [V.β[:,2:end,]; eye(K*(V.p-1)) zeros(K*(V.p-1),K)]
    else B = [V.β; eye(K*(V.p-1)) zeros(K*(V.p-1),K)]
    end
    V.i == true ? (return B, v) : (return B)
end
