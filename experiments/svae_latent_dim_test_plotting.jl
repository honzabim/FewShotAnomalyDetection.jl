using DataFrames
using CSV
using Statistics
using FileIO
using Plots

include("experimentalutils.jl")

data_folder = mainfolder * "experiments/svae_latent_dim_test/"

files = filter(s -> occursin("large_metrics_is.csv", s), readdir(data_folder))
results = []
for f in files
    if isfile(data_folder * f)
        push!(results, CSV.read(data_folder * f))
    end
end
results = vcat(results...)

using StatsPlots
gr()
p = @df results boxplot(:ldim, :auc_pxv_x_test, label="pxv", size = (500, 300), alpha=0.5)
p = @df results boxplot!(:ldim, :auc_pxv_pz_jaco_deco_test, label="jaco_deco", size = (500, 300), alpha=0.5, legend=false)
xlabel!(p, "Latent dimension")
ylabel!(p, "AUC")