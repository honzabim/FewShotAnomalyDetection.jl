using DataFrames
using CSV
using FileIO
using Statistics

include("experiments/experimentalutils.jl")

resultsFolder = mainfolder * "experiments/twostage_svae_corrjac/"
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
for n in names(results)[8:14]
    i = 1
    p = plot(title = String(n))
    for d in unique(results[:dataset])
        data = vec(results[results[:dataset] .== d, n])
        scatter!(p, ones(size(data)) .* i, data, zcolor = vec(results[results[:dataset] .== d, :ldim]), ylim = (0,1), label = d)
        # scatter!(p, ones(size(data)) .* i, data, zcolor = vec(results[results[:dataset] .== d, 14]), ylim = (0,1), label = d)
        i += 1
    end
    push!(pp, p)
end
push!(pp, plot())
plot(pp..., layout = (2,4), size = (1200, 700))

aucs = names(results)[8:14]
# fit = names(results)[15:20]
fit = names(results)[[15, 16, 19, 20]]

pp = []
for (i, f) in enumerate(fit)
    for (j, a) in enumerate(aucs)
        p = plot()
        for d in unique(results[:dataset])
            p = scatter!(results[results[:dataset] .== d, a], results[results[:dataset] .== d, f]) 
        end
        if j == 1
            ylabel!(p, String(f))
        end
        if i == length(fit)
            xlabel!(p, String(a))
        end
        push!(pp, p)
    end
end
plot(pp..., layout = (length(fit), length(aucs)), size = (1400, 800), legend = nothing)

ps = plot()
for d in unique(results[:dataset])
    scatter3d!(ps, results[results[:dataset] .== d, :rec_err_test], results[results[:dataset] .== d, :u_test_dst], results[results[:dataset] .== d, :auc_pxv_pz_jacobian_dec])
end
xlabel!(ps, "rec_err_test")
ylabel!(ps, "u_test_dst")
plot(ps, title = "auc_pxv_pz_jacobian_dec")

ps = plot()
d = "blood-transfusion"
res = results[results[:dataset] .== d, :]
for i in unique(results[:i])
    scatter3d!(ps, res[res[:i] .== i, :rec_err_test], res[res[:i] .== i, :u_test_dst], res[res[:i] .== i, :auc_pxv_pz_jacobian_dec])
end
xlabel!(ps, "rec_err_test")
ylabel!(ps, "u_test_dst")
plot(ps, title = "auc_pxv_pz_jacobian_dec")


gdfs = [by(results, [:dataset, :i], df -> DataFrame([[maximum(df[n])]],[n])) for n in names(results)[8:14]]
all = reduce((a...) -> join(a..., on = [:dataset, :i]), gdfs)


pp = []
for n in names(all)[3:9]
    i = 1
    p = plot(title = String(n))
    for d in unique(all[:dataset])
        data = vec(all[all[:dataset] .== d, n])
        scatter!(p, ones(size(data)) .* i, data, ylim = (0,1), label = d)
        # scatter!(p, ones(size(data)) .* i, data, zcolor = vec(all[all[:dataset] .== d, 14]), ylim = (0,1), label = d)
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


