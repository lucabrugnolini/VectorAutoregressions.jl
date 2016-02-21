function get_VAR_lag_length(D::Array, pbar::Integer, ic::ASCIIString, i::Bool=true)
        IC   = zeros(pbar,1)
        for p = 1:pbar
            i==true ? V = VAR(D,p,true) : V = VAR(D,p,false)
            n,m = size(V.X)
            t = n-p
            sig = V.Σ/t
        if ic == "aic"
            IC[p] = log(det(sig))+2*p*m^2/t
        elseif ic == "bic"
            IC[p] = log(det(sig))+(m^2*p)*log(t)/t
        elseif ic == "hqc"
            IC[p] = log(det(sig))+2*log(log(t))*m^2*p/t
        elseif ic == "aicc"
            b = t/(t-(p*m+m+1))
            IC[p] = t*(log(det(sig))+m)+2*b*(m^2*p+m*(m+1)/2)
        elseif error("'ic' must be aic, bic, hqc or aicc")
        end
     end
     length_ic = indmin(IC)
     println("Using $ic the best lag-length is $length_ic")
     return length_ic
end
