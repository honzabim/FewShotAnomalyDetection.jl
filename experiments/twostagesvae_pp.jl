using DataFrames
using CSV
using FileIO
using Statistics

include("experiments/experimentalutils.jl")

resultsFolder = mainfolder * "experiments/twostagesvae/"
files = readdir(resultsFolder)

results = []
for f in files
    if isfile(resultsFolder * f)
        push!(results, CSV.read(resultsFolder * f))
    end
end
results = vcat(results...)

aggres = []
for d in unique(results[:dataset])
    ddf = results[results[:dataset] .== d, :]
    mean_auc_pxv = mean(ddf[:auc_pxv])
    mean_auc_pz = mean(ddf[:auc_pz])
    push!(aggres, DataFrame(dataset = d, auc_pxv = mean_auc_pxv, auc_pz = mean_auc_pz))
end
aggres = vcat(aggres...)
