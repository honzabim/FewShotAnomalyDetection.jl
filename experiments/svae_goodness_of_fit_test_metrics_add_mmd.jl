using FewShotAnomalyDetection
using Flux
using MLDataPattern
using FluxExtensions
using Adapt
using DataFrames
using CSV
using Serialization
using Random
using Statistics
using IPMeasures
using FileIO

include("experimentalutils.jl")

data_folder = mainfolder * "experiments/svae_goodness_of_fit_dev/"

function subsample(data, labels, max_anomalies)
    anom_idx = labels .== 1
    norm_data = data[:, .!anom_idx]
    norm_labels = labels[.!anom_idx]

    num_anom_idx = findall(anom_idx)
    new_anom_idx = randperm(length(num_anom_idx))[1:min(max_anomalies, length(num_anom_idx))]

    return hcat(norm_data, data[:, new_anom_idx]), vcat(norm_labels, labels[new_anom_idx])
end

function process_file(f)
    println("$f: loading...")
    param_df = CSV.read(data_folder * f)
    dataset = param_df[:dataset][1]
    i = param_df[:i][1]
    hdim = param_df[:hdim][1]
    ldim = param_df[:ldim][1]
    num_pseudoinputs = param_df[:num_pseudoinputs][1]
    β = param_df[:β][1]
    γ = param_df[:γ][1]
    β_str = β == 1 ? "1.0" : β == 10 ? "10.0" : "$β"
    γ_str = γ == 1 ? "1.0" : "$γ"
    run_name = "$dataset-$i-$hdim-$ldim-$num_pseudoinputs-$β_str-$γ_str"
    
    println("$f: loading existing metrics...")
    types = DataType[String, Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, String, String, String, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64]
    df = DataFrame(CSV.read(data_folder * "$run_name-large_metrics.csv", types = types))
    
    svae = deserialize(data_folder * "$run_name-svae.jls")
    (x_train, labels_train) = deserialize(data_folder * "$dataset-$i-train.jls")
    (x_test, labels_test) = deserialize(data_folder * "$dataset-$i-test.jls")

    z_mmd_dst_train, z_mmd_pval_train = compute_metrics(svae, x_train, labels_train .- 1)

    println("$f: saving data...")
    df[:z_mmd_dst_train] = z_mmd_dst_train
    df[:z_mmd_pval_train] = z_mmd_pval_train

    CSV.write(data_folder * "$run_name-large_metrics.csv", df)
end

function compute_metrics(model::SVAE, x, labels)
    z = zparams(model, x)[1].data
    xp = model.g(z).data
    zp = zparams(model, xp)[1].data

    z_mmd_dst = nothing
    z_mmd_pval = 0
    println("computing null dst...")
    z_norm = z[:, labels .== 0]
    z_inds = randperm(size(z_norm, 2))[1:min(300, size(z_norm, 2))]

    null_dst, null_γ = null_distr_distances(model, length(z_inds))

    println("computing mmd dst...")
    z_mmd_dst = 0
    for i in 1:10
    z_mmd_dst += Flux.Tracker.data(IPMeasures.mmd(IPMeasures.IMQKernel(null_γ), sampleVamp(model, length(z_inds)).data, z_norm[:, z_inds], IPMeasures.pairwisecos))
    end
    z_mmd_dst /= 10
    z_mmd_pval = mmdpval(null_dst, z_mmd_dst)

    return z_mmd_dst, z_mmd_pval
end

function null_distr_distances(model, k = 500)
    z1 = sampleVamp(model, 500).data
    z2 = sampleVamp(model, 500).data 
    γ = get_γ(z1, z2)
    null_dst = zeros(Float32, 100)
    for i in 1:100
        z1 = sampleVamp(model, k).data
        z2 = sampleVamp(model, k).data
        null_dst[i] = IPMeasures.mmd(IPMeasures.IMQKernel(γ), z1, z2, IPMeasures.pairwisecos)
    end
    sort!(null_dst)
    return null_dst, γ
end

function get_γ(x, y)
    γs = -10:0.05:2
    cs = [IPMeasures.crit_mmd2_var(IPMeasures.IMQKernel(10.0 ^ γ), x, y, IPMeasures.pairwisecos) for γ in γs]
    γ = 10 ^ γs[argmax(cs)]
end

mmdpval(null_dst, x) = searchsortedfirst(null_dst, x) / length(null_dst)

files = readdir(data_folder)
files = filter(f -> occursin("large_metrics.csv", f), readdir(data_folder))
if length(ARGS) != 0
    contains = ARGS[1]
    files = filter(f -> occursin(contains, f), files)
end

for f in files
    if isfile(data_folder * f)
        process_file(f)
    end
end
