# Local projection
# Example based on Kilian and Kim 2011 dataset

#-----------Load packages------------------------------------------
using VectorAutoregressions
using DelimitedFiles, LinearAlgebra, Statistics, GrowableArrays

#-----------set base-path------------------------------------------
# path = Pkg.dir("VectorAutoregressions")
# import VectorAutoregressions; 
# joinpath(dirname(pathof(VectorAutoregressions)), "..", paths...)
path = "/home/lbrugnol/Dropbox/my_code/VectorAutoregressions.jl/"
#-----------Load data----------------------------------------------
y      = readdlm(joinpath(path,"test","lp_data.csv"),',')
irfv   = readdlm(joinpath(path,"test","lp_test_var_irf.csv"),',')
irfl   = readdlm(joinpath(path,"test","lp_test_lp_chol_irf.csv"),',')
stdv   = readdlm(joinpath(path,"test","lp_test_var_stdv.csv"),',')
stdl   = readdlm(joinpath(path,"test","lp_test_lp_std.csv"),',')
COVsig = readdlm(joinpath(path,"test","lp_test_var_covsig.csv"),',')
rfirfl = readdlm(joinpath(path,"test","lp_test_lp_rf_irf.csv"),',')

#-----------Hyperparameter-----------------------------------------
const pbar = 12 # max order of lag to test
const H = 24    # horizon
const intercept = true 

#-----------Lag-length selection for local projection and red.form IRFs-------------------
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
A0inv = V.Σ |> λ -> Array((cholesky(λ)).L)
mStd,mCov_Σ = irf_ci_asymptotic(V, H, V.inter)

mIRFs = IRFs_localprojection(y, p, H, A0inv, mCov_Σ)
IRF,CI = mIRFs.IRF, mIRFs.CI
CIl,CIh = CI.CIl, CI.CIh
