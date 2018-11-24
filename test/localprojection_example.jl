# Local projection
# Example based on Kilian and Kim 2011

#-----------Load packages------------------------------------------
using VectorAutoregressions

#-----------set base-path------------------------------------------
path = Pkg.dir("VectorAutoregressions")

#-----------Load data----------------------------------------------
y = readcsv(joinpath(path,"test","localprojection_dataset"))
irf_lp = readcsv(joinpath(path,"test","localprojection_irf_data.csv"))
cil_lp = readcsv(joinpath(path,"test","localprojection_cil_data.csv"))
cih_lp = readcsv(joinpath(path,"test","localprojection_cih_data.csv"))

#-----------Hyperparameter-----------------------------------------
const pbar = 12 # max order of lag to test
const H = 24    # horizon

#-----------Compute local projection IRFs and CI-------------------
localprojection_lagorder(y,pbar,H,"aic")
mIRFs = IRFs_localprojection(y, p, H, A0inv,cov_Σ)
IRF,CI = mIRFs.IRF, mIRFs.CI
CIl,CIh = CI.CIl, CI.CIh




j=10
plot([irfl[j,:] IRF[j,:] CIl[j,:] CIh[j,:] cil[j,:] cih[j,:]] )

