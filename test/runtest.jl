# Example VAR K=2 T=100
using VARs, Plots

const T,K = 100,2
const H = 24
const nrep = 1000
const p = 2
const intercept = true

y = rand(T,K)
V = VAR(y,p,intercept)
mIRFb = IRFs_b(V,H,nrep,intercept)
mIRFa = IRFs_a(V,H,intercept)

pIRF = plot(layout = grid(K,K))
[plot!(pIRF, [mIRFb.CI.CIl[i,:] mIRFb.IRF[i,:] mIRFb.CI.CIh[i,:]], color = ["red" "black" "red"],
line = [:dot :solid :dot], legend = false, subplot = i) for i in 1:K^2]
gui(pIRF)
