using DataFrames
using CSV
using Statistics
using FileIO
using Plots
plotlyjs()

include("experimentalutils.jl")

data_folder = mainfolder * "experiments/svae_goodness_of_fit_dev/"

files = readdir(data_folder)
results = []
for f in files
    if isfile(data_folder * f) && occursin("metrics.csv", f)
        push!(results, CSV.read(data_folder * f))
    end
end
results = vcat(results...)

# for d in unique(results[:dataset])
#     pp = []
#     for n in names(results)[10:17]
#         i = 1
#         p = plot(title = String(n))
#         data = vec(results[results[:dataset] .== d, n])
#         scatter!(p, ones(size(data)) .* i, data, zcolor = vec(results[results[:dataset] .== d, :ldim]), ylim = (0,1), label = d)
#         # scatter!(p, ones(size(data)) .* i, data, zcolor = vec(results[results[:dataset] .== d, 14]), ylim = (0,1), label = d)
#         i += 1
#         push!(pp, p)
#     end
#     push!(pp, plot())
#     plot(pp..., layout = (2,4), size = (1200, 700))
# end

aucs = names(results)[10:17]
fit = names(results)[[8, 9, 18, 19, 30, 31, 32]]
for d in unique(results[:dataset])
    pp = []
    for (i, f) in enumerate(fit)
        for (j, a) in enumerate(aucs)
            p = plot()
            p = scatter(results[results[:dataset] .== d, a], results[results[:dataset] .== d, f]) 
            if j == 1
                ylabel!(p, String(f))
            end
            if i == length(fit)
                xlabel!(p, String(a))
            end
            push!(pp, p)
        end
    end
    display(plot(pp..., layout = (length(fit), length(aucs)), size = (1400, 800), legend = nothing))
end

aucs = names(results)[22:29]
for d in unique(results[:dataset])
    pp = []
    for (i, f) in enumerate(fit)
        for (j, a) in enumerate(aucs)
            p = plot()
            p = scatter(results[results[:dataset] .== d, a], results[results[:dataset] .== d, f]) 
            if j == 1
                ylabel!(p, String(f))
            end
            if i == length(fit)
                xlabel!(p, String(a))
            end
            push!(pp, p)
        end
    end
    display(plot(pp..., layout = (length(fit), length(aucs)), size = (1400, 800), legend = nothing))
end