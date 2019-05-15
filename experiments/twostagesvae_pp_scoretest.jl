using DataFrames
using CSV
using FileIO
using Statistics

include("experiments/experimentalutils.jl")

resultsFolder = mainfolder * "experiments/twostagesvae_scoretest_mmdpval/"
files = readdir(resultsFolder)

results = []
for f in files
    if isfile(resultsFolder * f) && occursin(".csv", f)
        push!(results, DataFrame(CSV.read(resultsFolder * f)[1, :]))
    end
end
results = vcat(results...)

using Plots
using StatsPlots
plotlyjs()

# p1 = plot()
# for (i, n) in enumerate(names(results)[7:13])
#     scatter!(p1, ones(4) .* i, results[n], ylim = (0,1), label = String(n))
# end

# p2 = plot()
# for i in 1:4
#     scatter!(p2, 1:7, [results[i, c] for c in 7:13], ylim = (0,1))
# end

# for n in names(results)[7:end]
#     @df results boxplot(:dataset, n, size = [700, 700])
# end
pp = []
for n in names(results)[7:13]
    i = 1
    p = plot(title = String(n))
    for d in unique(results[:dataset])
        data = vec(results[results[:dataset] .== d, n])
        scatter!(p, ones(size(data)) .* i, data, ylim = (0,1), label = d)
        # scatter!(p, ones(size(data)) .* i, data, zcolor = vec(results[results[:dataset] .== d, 14]), ylim = (0,1), label = d)
        i += 1
    end
    push!(pp, p)
end
push!(pp, plot())
plot(pp..., layout = (2,4), size = (1200, 700))


# aggres = []
# for d in unique(results[:dataset])
#     ddf = results[results[:dataset] .== d, :]
#     mean_auc_pxv = mean(ddf[:auc_pxv])
#     mean_auc_pz = mean(ddf[:auc_pz])
#     push!(aggres, DataFrame(dataset = d, auc_pxv = mean_auc_pxv, auc_pz = mean_auc_pz))
# end
# aggres = vcat(aggres...)
