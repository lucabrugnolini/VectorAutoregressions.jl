# Example - BVAR
using StatsBase
using DelimitedFiles: readdlm
using Plots
using VectorAutoregressions
plotly()

#-----------Set base-path------------------------------------------
path = joinpath(dirname(pathof(VectorAutoregressions)), "..")

# Add bvar functions
include(joinpath(path,"src","bvar.jl")) 

#--------------------Load Data-------------------------------------------
y = readdlm(joinpath(path,"test","bvar_data.csv"),',',header=false)



y = convert(Array,y[:,1:3])
const λ = 0.1:0.1:1
const τ = 10*(0.1:0.1:1)
const ε = 0.0001
const p = 4
const H = 10
const reps = 10000
const burnin = 8000
const max_try = 1000
const update = 1000
prior = Hyperparameter(λ,τ,ε,p,H,reps,burnin,max_try,update)

mForecast = fit_bvar(y,prior)
pFore = plot(layout = grid(1,3));
for j in 1:size(mForecast,3)
    plot!(pFore,size(y,1)-H:size(y,1)+H, [y[size(y,1)-H:end,j]; median(mForecast[:,:,j],dims=1)'], legend = false,subplot = j);
    plot!(pFore,size(y,1)+1:size(y,1)+H, [[percentile(mForecast[:,i,j],16) for i in 1:H] median(mForecast[:,:,j],dims=1)' [percentile(mForecast[:,i,j],84) for i in 1:H]], color = "red", line = [:dash :solid :dash], subplot = j)
end

gui(pFore)