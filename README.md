# VectorAutoregressions.jl
Vector autoregressive models for Julia

[![Build Status](https://travis-ci.org/lucabrugnolini/VectorAutoregressions.jl.svg?branch=master)](https://travis-ci.org/lucabrugnolini/VectorAutoregressions.jl)
[![Coverage Status](https://coveralls.io/repos/github/lucabrugnolini/VectorAutoregressions.jl/badge.svg?branch=master)](https://coveralls.io/github/lucabrugnolini/VectorAutoregressions.jl?branch=master)
[![codecov](https://codecov.io/gh/lucabrugnolini/VectorAutoregressions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/lucabrugnolini/VectorAutoregressions.jl)



## Installation
```julia
Pkg.clone("https://github.com/lucabrugnolini/VectorAutoregressions.jl")
```
## Introduction
This package is a work in progress for the estimation and identification of Vector Autoregressive (VAR) models.

## Status
- [x] VAR
  - [x] VAR(1) form
  - [x] Lag-length selection
    - [x] AIC
    - [x] AICC
    - [x] BIC
    - [x] HQC
  - [x] VAR impulse response function (IRFs)
    - [ ] Identification
      - [x] Reduce form
      - [x] Cholesky
      - [ ] Long-run restrictions
      - [ ] Sign restrictions
      - [ ] Heteroskedasticity
      - [ ] External instruments (ex. high-frequency,narrative)
    - [x] Confidence bands
      - [x] Asymptotic
      - [x] Bootstrap
      - [x] Bootstrap-after-bootstrap
  - [ ] Forecasting
    - [ ] BVAR
    - [ ] FAVAR
- [ ] Local projection IRFs
    - [ ] Lag-length selection
    - [ ] Confidence bands
      - [ ] Standard
      - [ ] Bootstrap
  - [ ] Bayesian Local Projection

## Example
```julia
using VectorAutoregressions
V = VAR(Y, p, i)
```
Where `Y` is a matrix with data, `p` is the lag-length and `i` is a Boolean for including an intercept (default is true). It returns a fitted VAR(p) model with the following structure:
```julia
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
You can access to each element writing `V.` and than the element you are interested in (for example for the covariates `V.X`). The example below shows how to fit a VAR model to a VAR(1) DGP, computing IRFs and CI with both asymptotic and bootstrap procedures.

```julia
## Example:
using VectorAutoregressions, Plots

y = readdlm(joinpath(Pkg.dir("VectorAutoregressions"),"test","cholvar_test_data.csv"), ',')
intercept = false #intercept in the estimation
p = 2 #select lag-length
H = 15 # IRFs horizon
nrep = 500 #bootstrap sample

# Fit VAR(4) to data
V = VAR(y,p,intercept)

# Estimate IRFs - Cholesky identification
T,K = size(y) 
mIRFa = IRFs_a(V,H,intercept) #asymptotic conf. bandf
mIRFb = IRFs_b(V,H,nrep,intercept) #bootstrap conf. bands

# Plot irf + asy ci
pIRF_asy = plot(layout = grid(K,K));
[plot!(pIRF_asy, [mIRFa.CI.CIl[i,:] mIRFa.IRF[i,:] mIRFa.CI.CIh[i,:]], color = ["red" "red" "red"],
line = [:dash :solid :dash], legend = false, subplot = i) for i in 1:K^2]
gui(pIRF_asy)

# Plot irf + bootstraped ci
pIRF_boot = plot(layout = grid(K,K));
[plot!(pIRF_boot, [mIRFb.CI.CIl[i,:] mIRFb.IRF[i,:] mIRFb.CI.CIh[i,:]], color = ["blue" "blue" "blue"],
line = [:dash :solid :dash], legend = false, subplot = i) for i in 1:K^2]
gui(pIRF_boot)
```
