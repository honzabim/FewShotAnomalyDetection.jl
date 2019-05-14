using DataFrames
using CSV
using FileIO
using Statistics

include("experiments/experimentalutils.jl")

resultsFolder = mainfolder * "experiments/twostagesvae_scoretest_wide/"
files = readdir(resultsFolder)

results = []
for f in files
    if isfile(resultsFolder * f) && occursin(".csv", f)
        push!(results, CSV.read(resultsFolder * f))
    end
end
results = vcat(results...)

using Plots
using StatsPlots
plotly()

# for n in names(results)[7:end]
#     @df results boxplot(:dataset, n, size = [700, 700])
# end
pp = []
for n in names(results)[7:12]
    i = 1
    p = plot(title = String(n))
    for d in unique(results[:dataset])
        data = vec(results[results[:dataset] .== d, n])
        scatter!(p, ones(size(data)) .* i, data, ylim = (0,1), label = d)
        i += 1
    end
    push!(pp, p)
end
plot(pp..., layout = (2,3), size = (1200, 1000))

# aggres = []
# for d in unique(results[:dataset])
#     ddf = results[results[:dataset] .== d, :]
#     mean_auc_pxv = mean(ddf[:auc_pxv])
#     mean_auc_pz = mean(ddf[:auc_pz])
#     push!(aggres, DataFrame(dataset = d, auc_pxv = mean_auc_pxv, auc_pz = mean_auc_pz))
# end
# aggres = vcat(aggres...)
