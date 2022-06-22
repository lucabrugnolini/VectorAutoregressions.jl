#--------------Replicating Gertler and Karadi 2015-----------------------

#---------------------Load Packages--------------------------------------
using VectorAutoregressions
using Plots
plotly()
using Random, LinearAlgebra 
using Statistics, GrowableArrays
using DelimitedFiles: readdlm

#-----------Set base-path------------------------------------------
path = joinpath(dirname(pathof(VectorAutoregressions)), "..")

#--------------------Load Data-------------------------------------------
# X,names = readdlm("/home/lbrugnol/Dropbox/my_code/VectorAutoregressions.jl/test/gertler_karadi_var_data.csv",',',header=true)
X,names = readdlm(joinpath(path,"test","gertler_karadi_var_data.csv"),',',header=true)

#-------------------Setting-up the variables-----------------------------
Y = convert(Array{Float64},X[:,3:6])
Z = convert(Array{Float64},X[findlast(λ-> λ =="",X[:,7])+1:end,8:8])

#------------------Hyperparameters---------------------------------------
α = [.05, .10, .15, .25] # confidence interval
intercept = true
K = size(Y,2)
nrep = 1000
H = 50
p = 12

#---------------Cholesky VAR---------------------------------------------
V = VAR(Y,p,intercept)
mIRFa = IRFs_a(V,H,intercept)

pIRF_asy = plot(layout = grid(K,1));
let counter = 0
    for i in collect(3:K:15)
        counter +=1
        plot!(pIRF_asy, [mIRFa.IRF[i,:] mIRFa.CI.CIl[i,:] mIRFa.CI.CIh[i,:]], color = ["red" "red" "red"],
        line = [:solid :dash :dash], legend = false, label = ["cholesky"], subplot = counter)
    end
    gui(pIRF_asy)
end
#-------------External instrument VAR-------------------------------------
Y_z = Y[:,[3, 1, 2, 4]] # fedfund first
V = VAR(Y_z,p,intercept)

IRF = IRFs_ext_instrument(V, Z, H, nrep, α, intercept)

pIRF = plot(layout = grid(4,1));
for i in [3 1 2 4]
    plot!(pIRF,[IRF.IRF[:,i] IRF.CI.CIl[4,:,i] IRF.CI.CIh[4,:,i]], subplot = i, legend = false, color = ["blue" "blue" "blue"], line = [:solid :dash :dash])
end
gui(pIRF)
