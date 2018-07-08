## Example: 
## DGP - Bivariate VAR(1) Model from Kilian, RESTAT, 1998
# B11 set to 0.01
using VectorAutoregressions, Plots
plotly()

const T,K = 1000,2
const H = 24
const nrep = 1000
const p = 1
const intercept = false
const burnin = 100


srand(1234)
mR  = [0.9 0.01; 0.5 0.5]             #  Coefficients of the 1st lag B11 set to 0.01
mΣ_true = [1 0.3; 0.3 1]              #  vcv of u
mP = cholfact(mΣ_true)[:L]            #  Cholesky PP'= Σ
mI = eye(K) 
mIRF_true = At_mul_Bt(mP,mI)[:]       # Initialize structural IRF
# True IRF
for h=1:H     
    vIRF = At_mul_Bt(mP,mR^h)[:] 
    mIRF_true = hcat(mIRF_true,vIRF)
end

# Generate data from the VAR(1) DGP
y = zeros(T+burnin,K)
gen_var1_data!(y,mR,mP,burnin)

# Fit VAR(1) to data
V = VAR(y,p,intercept)
mIRFb = IRFs_b(V,H,nrep,intercept)
mIRFa = IRFs_a(V,H,intercept)

# Plot true irf vs model irf + asy ci
pIRF_asy = plot(layout = grid(K,K));
[plot!(pIRF_asy, [mIRF_true[i,:] mIRFa.IRF[i,:] mIRFa.CI.CIl[i,:] mIRFa.CI.CIh[i,:]], color = ["blue" "red" "red" "red"],
line = [:solid :solid :dash :dash], legend = false, label = ["true" "model"], subplot = i) for i in 1:K^2]
gui(pIRF_asy)

# Plot true irf vs model irf + bootstraped ci
pIRF_boot = plot(layout = grid(K,K));
[plot!(pIRF_boot, [mIRF_true[i,:] mIRFb.CI.CIl[i,:] mIRFb.IRF[i,:] mIRFb.CI.CIh[i,:]], color = ["blue" "black" "black" "black"],
line = [:solid :solid :dash :dash], legend = false, label = ["true" "model"], subplot = i) for i in 1:K^2]
gui(pIRF_boot)

