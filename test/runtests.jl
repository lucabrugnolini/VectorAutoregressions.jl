using VectorAutoregressions
using Test
using DelimitedFiles: readdlm
using LinearAlgebra: cholesky

#-----------Set base-path------------------------------------------
path = joinpath(dirname(pathof(VectorAutoregressions)), "..")

#--------------------Test VAR model---------------------------------------------------------
# comparison against http://www-personal.umich.edu/~lkilian/figure9_1_chol.zip

@testset "VAR" begin

y = readdlm(joinpath(path,"test","cholvar_test_data.csv"),',')

V = VAR(y, 4, true)

@test V.p == 4
@test V.Σ ≈ [
        312.524637669663 0.773643992117187 0.919341486304653
        0.773643992117187 0.0515304724687740 0.0148802276681216
        0.919341486304653 0.0148802276681216 0.557021891337960
        ]

@test V.β[:,1:4] ≈ [
        -0.680676727884458 -0.00637960178073601 0.936519611007488 1.75104737240770
        -0.0152242907833718 0.00177981748889471 0.599086054531876 0.0281955980964510
        0.436329751995733 -0.00236721376013309 -0.218737236527547 0.332361646200618
        ]

@test V.ϵ[:,1:4] ≈ [
        70.3934221013155 -0.443580106581005 4.21164200500382 1.22299717886091
        0.189207494365181 0.287694213161629 0.692414550347607 0.405418261036420
        -1.39379479232130 0.277340573620367 -0.573151724739961 -0.395501452903569
        ]

mIRFa = IRFs_a(V,4,true)

@test [mIRFa.IRF[1,1:4]'; mIRFa.IRF[4,1:4]'; mIRFa.IRF[7,1:4]'] ≈ [
        17.6783663744607 0.0192642533976305 -2.85156060358692 0.594909295963393
        0.0437621879606954 0.0591478592307352 0.113331057827868 0.109932348126692
        0.0520037579735191 -0.0341368375885218 -0.111284843056732 -0.0491881671489620
        ]

mIRFb = IRFs_b(V,4,10,true)

@test [mIRFb.IRF[1,1:4]'; mIRFb.IRF[4,1:4]'; mIRFb.IRF[7,1:4]'] ≈ [
        17.6783663744607 0.0192642533976305 -2.85156060358692 0.594909295963393
        0.0437621879606954 0.0591478592307352 0.113331057827868 0.109932348126692
        0.0520037579735191 -0.0341368375885218 -0.111284843056732 -0.0491881671489620
        ]

end

#--------------------Test BVAR model---------------------------------------------------------
# comparison against http://cremfi.econ.qmul.ac.uk/outgoing/bvar.zip
#= include(joinpath(path,"src","bvar.jl")) 
y = readdlm(joinpath(path,"test","bvar_data.csv"), ',')
y = y[:,1:3]
prior = Hyperparameter()
mForecast = fit_bvar(y,prior)

@test isapprox([0.8510 1.4081 2.2570 2.3415 2.4622 2.5835 2.6867 2.5790 2.5897 2.5767],median(mForecast[:,:,1],1); atol = 1)
@test isapprox([1.9614 2.2587 1.8328 1.8745 2.0870 2.2014 2.3303 2.5225 2.5453 2.59387],median(mForecast[:,:,2],1); atol = 1)
@test isapprox([-0.3827 -0.2272 -0.1532 -0.0735 0.0784 0.1620 0.3764 0.5247 0.7264 0.8861],median(mForecast[:,:,3],1); atol = 1)
 =#
#--------------------Test Local Projection IRFs---------------------------------------------------------
# comparison against Kilian and Kim (2011) generated data from VAR(12)

#-----------Load data----------------------------------------------
y      = readdlm(joinpath(path,"test","lp_data.csv"), ',')
irfl   = readdlm(joinpath(path,"test","lp_test_lp_chol_irf.csv"), ',')
stdl   = readdlm(joinpath(path,"test","lp_test_lp_std.csv"), ',')
stdv   = readdlm(joinpath(path,"test","lp_test_var_stdv.csv"), ',')
COVsig = readdlm(joinpath(path,"test","lp_test_var_covsig.csv"), ',')
rfirfl = readdlm(joinpath(path,"test","lp_test_lp_rf_irf.csv"), ',')
mlag   = readdlm(joinpath(path,"test","lp_test_lp_laglength.csv"), ',')


#-----------Hyperparameter-----------------------------------------
const p = 12    # lag length
const H = 24    # horizon
const intercept = true 

@testset "Local projection IRFs" begin
#-----------Reduced form local projection IRFs-------------------
RF_IRFs = IRFs_localprojection(y, p, H)
mRFIRFs,CI = RF_IRFs.IRF, RF_IRFs.CI

#-----------Structural local projection IRFs-------------------
# Using a VAR(pbar) as auxiliary model for cholesky identification
V = VAR(y,p,intercept)
A0inv = V.Σ |> λ -> cholesky(λ).L |> Matrix
mStd,mCov_Σ = irf_ci_asymptotic(V, H, V.inter)

IRF = IRFs_localprojection(y, p, H, A0inv, mCov_Σ)
mIRF,CI = IRF.IRF, IRF.CI
CIl,CIh = CI.CIl, CI.CIh

@test isapprox(mRFIRFs,rfirfl)
@test isapprox(A0inv,[0.721465    0.0          0.0        0.0
               -0.0842337   2.68627      0.0        0.0
                2.61134     0.724746    26.8394     0.0
                0.0886017  -0.00940385   0.0693017  0.486568],
                atol = 0.0001)
@test isapprox(V.Σ, [0.520512   -0.0607717    1.88399   0.063923
              -0.0607717   7.22317      1.7269   -0.0327246
               1.88399     1.7269     727.695     2.08457
               0.063923   -0.0327246    2.08457   0.249489],
               atol = 0.0001)
@test isapprox(mStd, stdv, atol = 0.1)
@test isapprox(mCov_Σ, COVsig, atol = 0.1)
@test isapprox(mIRF, irfl)
@test isapprox(CIl,(irfl - 1.96*stdl),atol = 0.1)
@test isapprox(CIh,(irfl + 1.96*stdl),atol = 0.1)

end

#-----------Test LP with different lag-length at different horizon-------
const vP = p*ones(Int64,H) # vector of lag-length

@testset "Local projections (II)" begin
#-----------Reduced form local projection IRFs-------------------
RF_IRFs = IRFs_localprojection(y, vP, H)
mRFIRFs,CI = RF_IRFs.IRF, RF_IRFs.CI

#-----------Structural local projection IRFs-------------------
# Using a VAR(pbar) as auxiliary model for cholesky identification
V = VAR(y,p,intercept)
A0inv = V.Σ |> λ -> cholesky(λ).L |> Matrix
mStd,mCov_Σ = irf_ci_asymptotic(V, H, V.inter)

IRF = IRFs_localprojection(y, vP, H, A0inv, mCov_Σ)
mIRF,CI = IRF.IRF, IRF.CI
CIl,CIh = CI.CIl, CI.CIh

@test isapprox(mRFIRFs,rfirfl)
@test isapprox(mIRF, irfl)
@test isapprox(CIl,(irfl - 1.96*stdl),atol = 0.1)
@test isapprox(CIh,(irfl + 1.96*stdl),atol = 0.1)

end

#-----------Test LP lag-length selecion procedure----------------
const pbar = 12      # max order of lag to test

@testset "select lag-length" begin
#-----------Select lag-length with AIC, BIC, AICC, HQC-----------
mVARlag = (var_lagorder(y,pbar,ic) for ic in ["aic","bic","aicc","hqc"]) |> λ -> hcat(collect(λ)...)
mLPlag  = (lp_lagorder(y,pbar,H,ic) for ic in ["aic","bic","aicc","hqc"]) |> λ -> hcat(collect(λ)...)

@test mLPlag == mlag
@test mVARlag == [12 2 12 3]

end
