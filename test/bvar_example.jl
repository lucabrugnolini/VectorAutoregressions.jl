# Example - BVAR
using Plots, CSV, StatsBase
include(joinpath(Pkg.dir("VectorAutoregressions"),"src","bvar.jl")) 
plotly()

y = CSV.read(joinpath(Pkg.dir("VectorAutoregressions"),"test","bvar_data.csv"), header = false)
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
plot!(pFore,size(y,1)-H:size(y,1)+H, [y[size(y,1)-H:end,j]; median(mForecast[:,:,j],1)'], legend = false,subplot = j);
plot!(pFore,size(y,1)+1:size(y,1)+H, [[percentile(mForecast[:,i,j],16) for i in 1:H] median(mForecast[:,:,j],1)' [percentile(mForecast[:,i,j],84) for i in 1:H]], color = "red", line = [:dash :solid :dash], subplot = j)
end

gui(pFore)


using Base.Test
@test isapprox([0.8510 1.4081 2.2570 2.3415 2.4622 2.5835 2.6867 2.5790 2.5897 2.5767],median(mForecast[:,:,1],1) ;atol = 0.5)
@test isapprox([1.9614 2.2587 1.8328 1.8745 2.0870 2.2014 2.3303 2.5225 2.5453 2.59387],median(mForecast[:,:,2],1) ;atol = 0.5)
@test isapprox([-0.3827 -0.2272 -0.1532 -0.0735 0.0784 0.1620 0.3764 0.5247 0.7264 0.8861],median(mForecast[:,:,3],1) ;atol = 0.5)

