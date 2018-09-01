using VectorAutoregressions, Plots
X,names = readdlm("/home/lbrugnol/Dropbox/MONPOLSHOCK/persistance_var/LB_var_Target_GOV10YW.csv",',',header=true)

Y = convert(Array,X[:,1:6])
Z = convert(Array,X[:,7:7])
H = 2401
nrep = 1000
p = 2
intercept = true
α = [.05, .10, .15, .25] # confidence interval

V = VAR(Y,p,intercept)
IRF = IRFs_ext_instrument(V, Z, H, nrep, α, intercept)

plot(IRF.IRF[:,1])
plot!([IRF.CI.CIl[:,1] IRF.CI.CIh[:,1]])
plot!([IRF.CI.CIl[:,2] IRF.CI.CIh[:,2]])
plot!([IRF.CI.CIl[:,3] IRF.CI.CIh[:,3]])
plot!([IRF.CI.CIl[:,4] IRF.CI.CIh[:,4]])

