using Plots, VectorAutoregressions
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

pIRF = plot(layout = grid(6,1))
for i in 1:6
plot!(pIRF,IRF.IRF[:,i], subplot = i)
# plot!(pIRF,[IRF.CI.CIl[1,:,i] IRF.CI.CIh[1,:,i]], subplot = i)
# plot!(pIRF,[IRF.CI.CIl[2,:,i] IRF.CI.CIh[2,:,i]], subplot = i)
# plot!(pIRF,[IRF.CI.CIl[3,:,i] IRF.CI.CIh[3,:,i]], subplot = i)
plot!(pIRF,[IRF.CI.CIl[4,:,i] IRF.CI.CIh[4,:,i]], subplot = i)
end
gui(pIRF)

## Replicating Gertler and Karadi 2015
X,names = readdlm("/home/lbrugnol/Dropbox/my_code/VectorAutoregressions.jl/test/gertler_karadi_var_data.csv",',',header=true)

Y = convert(Array{Float64},X[:,3:6])
Z = convert(Array{Float64},X[findlast(λ-> λ =="",X[:,7])+1:end,8:8])
H = 50
nrep = 1000
p = 12
intercept = true
α = [.05, .10, .15, .25] # confidence interval
K = size(Y,2)

# Cholesky VAR
V = VAR(Y,p,intercept)
mIRFa = IRFs_a(V,H,intercept)

pIRF_asy = plot(layout = grid(K,1));
count = 0
for i in collect(3:K:15)
    count +=1
plot!(pIRF_asy, [mIRFa.IRF[i,:] mIRFa.CI.CIl[i,:] mIRFa.CI.CIh[i,:]], color = ["red" "red" "red"],
line = [:solid :dash :dash], legend = false, label = ["cholesky"], subplot = count)
end
gui(pIRF_asy)


# External instrument VAR
Y_z = Y[:,[3, 1, 2, 4]] # fedfund first
V = VAR(Y_z,p,intercept)

IRF = IRFs_ext_instrument(V, Z, H, nrep, α, intercept)

pIRF = plot(layout = grid(4,1));
for i in [3 1 2 4]
plot!(pIRF,[IRF.IRF[:,i] IRF.CI.CIl[4,:,i] IRF.CI.CIh[4,:,i]], subplot = i, legend = false, color = ["blue" "blue" "blue"], line = [:solid :dash :dash])
# plot!(pIRF,[IRF.CI.CIl[1,:,i] IRF.CI.CIh[1,:,i]], subplot = i)
# plot!(pIRF,[IRF.CI.CIl[2,:,i] IRF.CI.CIh[2,:,i]], subplot = i)
# plot!(pIRF,[IRF.CI.CIl[3,:,i] IRF.CI.CIh[3,:,i]], subplot = i)
end
gui(pIRF)
