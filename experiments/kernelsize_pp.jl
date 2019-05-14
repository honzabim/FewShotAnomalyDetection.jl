using DataFrames
using CSV
using FileIO
using Statistics

include("experiments/experimentalutils.jl")

resultsFolder = mainfolder * "experiments/kernelsizetest/"
files = readdir(resultsFolder)

results = []
for f in files
    if isfile(resultsFolder * f)
        push!(results, CSV.read(resultsFolder * f))
    end
end
results = vcat(results...)

aggres = []
for (g, d) in Base.product(unique(results[:γ]), unique(results[:dataset]))
    ddf = results[(results[:dataset] .== d) .& (results[:γ] .== g), :]
    mean_auc_pxv = maximum(ddf[:auc_pxv])
    mean_auc_pxis = maximum(ddf[:auc_pxis])
    mean_auc_pz = maximum(ddf[:auc_pz])
    push!(aggres, DataFrame(dataset = d, γ = g, auc_pxv = mean_auc_pxv, auc_pxis = mean_auc_pxis, auc_pz = mean_auc_pz))
end
aggres = vcat(aggres...)
