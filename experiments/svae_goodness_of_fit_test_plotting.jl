using DataFrames
using CSV
using Statistics
using FileIO
using Plots

include("experimentalutils.jl")

data_folder = mainfolder * "experiments/svae_goodness_of_fit_dev/"

files = filter(s -> occursin("large_metrics_is.csv", s), readdir(data_folder))
results = []
l = []
for (i,f) in enumerate(files)
    if isfile(data_folder * f)
        df = CSV.read(data_folder * f)
        push!(results, df)
        push!(l, size(df, 2))
        if size(df, 1) > 1
            println("$f seems corrupted, saving just first row...")
            CSV.write(data_folder * f, DataFrame(df[1, :]))
        end
        # if (df[:dataset][1] == "3459766")
        #     println("$f has a missing dataset!")
        # end
    end
end
results = vcat(results...)
unique(results[:dataset])

using StatsPlots
plotlyjs()
@df results violin(:dataset,:auc_pxv_x_test, side=:left, linewidth = 0, label="pxv", size = (1200, 800))
@df results violin!(:dataset,:auc_pxv_pz_jaco_deco_test, linewidth = 0, side=:right, label="pxv_pz_jaco_deco")

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

function sel_pxvita(dff)
    u = sort(dff,:log_pxv_x_train)
    DataFrame(u[1,[:auc_pxv_x_test]])
end

function sel(dff)
    # I = sortperm(dff[:mean_disc_scores_z_test], rev = true)

    # qz = quantile(dff[:log_pxv_z_train], 0.95)
    # dff = filter(row -> row[:log_pxv_z_train]>=qz, dff)

    q = quantile(dff[:z_mmd_pval_train],0.95)
    dff = filter(row -> row[:z_mmd_pval_train]>=q, dff)

    I = sortperm(dff[:log_pxv_x_train])
    DataFrame(dff[I[1],[:auc_pxv_pz_jaco_deco_test, :z_mmd_pval_train]])
end

df = results
df1 = by(df, [:dataset, :i], sel_pxvita)
df2 = by(df, [:dataset, :i], sel)
join(df1, df2, on = [:dataset, :i])

aucs = names(results)[10:17]
fit = names(results)[[18, 19, 30, 31, 32]]

for d in unique(results[:dataset])
    pp = []
    for (i, f) in enumerate(fit)
        for (j, a) in enumerate(aucs)
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