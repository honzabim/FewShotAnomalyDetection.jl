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

    if isfile(data_folder * "$run_name-large_metrics_is.csv")
        println("Skipping $f because it was processed already...")
        return
    end
    
    println("$f: loading existing metrics...")
    types = DataType[String, Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, String, String, String, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64]
    df = DataFrame(CSV.read(data_folder * "$run_name-large_metrics.csv", types = types))
    
    svae = deserialize(data_folder * "$run_name-svae.jls")
    (x_train, labels_train) = deserialize(data_folder * "$dataset-$i-train.jls")
    (x_test, labels_test) = deserialize(data_folder * "$dataset-$i-test.jls")

    log_px_is_train, auc_px_is_train = compute_metrics(svae, x_train, labels_train .- 1)
    log_px_is_test, auc_px_is_test = compute_metrics(svae, x_test, labels_test .- 1)

    println("$f: saving data...")
    df[:log_px_is_train] = mean(log_px_is_train)
    df[:log_px_is_test] = mean(log_px_is_test)
    df[:auc_px_is_train] = auc_px_is_train
    df[:auc_px_is_test] = auc_px_is_test

    CSV.write(data_folder * "$run_name-large_metrics_is.csv", df)
end

function compute_metrics(model::SVAE, x, labels)
    
    log_px_is = log_px(model, x)
    println("computing aucs...")
    auc_px_is = computeauc(.-log_px_is, labels)

    return log_px_is, auc_px_is
end


files = readdir(data_folder)
files = filter(f -> occursin("large_metrics.csv", f), readdir(data_folder))
if length(ARGS) != 0
    contains = ARGS[1]
    files = filter(f -> occursin(contains, f), files)
end

for f in reverse(files)
    if isfile(data_folder * f)
        process_file(f)
    end
end
