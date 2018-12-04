# Local projection
# Example based on Kilian and Kim 2011 dataset

#-----------Load packages------------------------------------------
using VectorAutoregressions

#-----------set base-path------------------------------------------
path = Pkg.dir("VectorAutoregressions")
# path = "/home/lbrugnol/Dropbox/my_code/VectorAutoregressions.jl/"
#-----------Load data----------------------------------------------
y      = readcsv(joinpath(path,"test","lp_data.csv"))
irfv   = readcsv(joinpath(path,"test","lp_test_var_irf.csv"))
irfl   = readcsv(joinpath(path,"test","lp_test_lp_chol_irf.csv"))
stdv   = readcsv(joinpath(path,"test","lp_test_var_stdv.csv"))
stdl   = readcsv(joinpath(path,"test","lp_test_lp_std.csv"))
COVsig = readcsv(joinpath(path,"test","lp_test_var_covsig.csv"))
rfirfl = readcsv(joinpath(path,"test","lp_test_lp_rf_irf.csv"))

#-----------Hyperparameter-----------------------------------------
const pbar = 12 # max order of lag to test
const H = 24    # horizon
const intercept = true

#-----------Lag-length selection for local projection adn red.form IRFs-------------------
#Ex. 1
p = lp_lagorder(y,pbar,H,"aic") # using aic selection criteria
mIRFs = IRFs_localprojection(y, p, H)
#Ex. 2
p = lp_lagorder(y,pbar,H,"bic") # using bic selection criteria
mIRFs = IRFs_localprojection(y, p, H)
#Ex. 3
p = lp_lagorder(y,pbar,H,"aicc") # using aic correced selection criteria
mIRFs = IRFs_localprojection(y, p, H)
#Ex. 4
p = lp_lagorder(y,pbar,H,"hqc") # using Hannan-Quinn selection criteria
mIRFs = IRFs_localprojection(y, p, H)
#Ex. 5
p = 12                                       # fixing the lag-length for all projections
mIRFs = IRFs_localprojection(y, p, H)

#-----------Structural local projection IRFs-------------------
#Ex. 1 -- using a VAR(pbar) as auxiliary model for cholesky identification
p_var = 12
V = VAR(y,p_var,true)
A0inv = V.Σ |> λ -> cholfact(λ)[:L] |> full
mStd,mCov_Σ = irf_ci_asymptotic(V, H, V.inter)

mIRFs = IRFs_localprojection(y, p, H, A0inv, mCov_Σ)
IRF,CI = mIRFs.IRF, mIRFs.CI
CIl,CIh = CI.CIl, CI.CIh
