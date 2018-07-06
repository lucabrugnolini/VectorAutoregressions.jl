# Example - BVAR
using Plots
include(joinpath(Pkg.dir("VectorAutoregressions.jl"),"src","")) 
plotly()

y = CSV.read(joinpath(Pkg.dir("VectorAutoregressions.jl"),"test","bvar_data.csv"), header = false)
y = convert(Array,y[:,1:3])
const λ = 0.1:0.1:1
const τ = 10*(0.1:0.1:1)
const ε = 0.0001
const p = 4
const H = 10
const reps = 10000
const burnin = 1000
const max_try = 1000
const update = 1000
prior = Hyperparameter(λ,τ,ε,p,H,reps,burnin,max_try,update)

mForecast = fit_bvar(y,prior)
plot(layout = grid(1,3))
for j in 1:3
plot(1:size(y,1), y[:,j])
plot!(size(y,1)+1:size(y,1)+H, [[percentile(mForecast[:,i,j],16) for i in 1:H] median(mForecast[:,:,j],1)'  [percentile(mForecast[:,i,j],84) for i in 1:H]])
end