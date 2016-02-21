# VAR
Vector autoregressive model for Julia

# Status
-[x] Basic constructor
-[x] Lag-length selection (AIC, BIC, HQ, AICC)
-[x] VAR(1) form
-[x] Impulse response function (reduce form, Cholesky)

# Example
```
using VARs
V = VAR(Y, p, i)
```
Where `Y` is a matrix with data, `p` is the lag-length and `i` is a Boolean for including an intercept (default is true). It returns a fitted VAR(`p`) model with the following structure
```
type VAR
  Y::Array # dep. variables
  X::Array # covariates
  β::Array # parameters
  ϵ::Array # residuals
  Σ::Array # VCV matrix
  p::Int64 # lag-length
  i::Bool # true or false for including an intercept (default is true)
end
```
You can access to each element writing `V.`
