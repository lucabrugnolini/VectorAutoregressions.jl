# VAR
Vector autoregressive model for Julia

## Status
- [x] Basic constructor
- [x] Lag-length selection
 - [x] AIC
 - [x] AICC
 - [x] BIC
 - [x] HQC
- [x] VAR(1) form
- [x] Impulse response function (IRFs)
  - [ ] Identification 
    - [x] Reduce form
    - [x] Cholesky
    - [ ] BQ
    - [ ] Uligh
  - [x] Confidence bands
    - [x] Asymptotic
    - [x] Bootstrap
    - [x] Bootstrap-after-bootstrap
   - [ ] Local projection IRFs
     - [ ] Lag-length selection
     - [ ] Confidence bands
       - [ ] Standard
        - [ ] Bootstrap

## Example
```
using VARs
V = VAR(Y, p, i)
```
Where `Y` is a matrix with data, `p` is the lag-length and `i` is a Boolean for including an intercept (default is true). It returns a fitted VAR(p) model with the following structure:
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
You can access to each element writing `V.` and than the element you are interested in (for example for the covariates `V.X`).
