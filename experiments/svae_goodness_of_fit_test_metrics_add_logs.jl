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
    
    if isfile(data_folder * "$run_name-large_metrics.csv")
        println("Skipping $f because it has logs added already...")
        return
    end

    df = CSV.read(data_folder * "$run_name-metrics.csv")

    svae = deserialize(data_folder * "$run_name-svae.jls")
    (x_train, labels_train) = deserialize(data_folder * "$dataset-$i-train.jls")
    (x_test, labels_test) = deserialize(data_folder * "$dataset-$i-test.jls")

    println("$f: computing metrics m1...")
    m1 = Tuple(zeros(13))
    for j in 1:5
        x_train1, labels_train1 = subsample(x_train, labels_train .- 1, 1)
        m1 = m1 .+ compute_metrics(svae, x_train1, labels_train1)
    end
    m1 = m1 ./ 10

    println("$f: computing metrics m5...")
    m5 = Tuple(zeros(13))
    for j in 1:5
        x_train5, labels_train5 = subsample(x_train, labels_train .- 1, 5)
        m5 = m5 .+ compute_metrics(svae, x_train5, labels_train5)
    end
    m5 = m5 ./ 10

    println("$f: saving data...")
    df[:auc_pxv_x_train_a1] = m1[1]
    df[:auc_pxv_z_train_a1] = m1[2]
    df[:auc_pz_train_a1] = m1[3]
    df[:auc_pz_jaco_enco_train_a1] = m1[4]
    df[:auc_pz_jaco_deco_train_a1] = m1[5]
    df[:auc_pxv_pz_train_a1] = m1[6]
    df[:auc_pxv_pz_jaco_enco_train_a1] = m1[7]
    df[:auc_pxv_pz_jaco_deco_train_a1] = m1[8]
    df[:log_pxv_x_train_a1] = m1[9]
    df[:log_pxv_z_train_a1] = m1[10]
    df[:log_pz_train_a1] = m1[11]
    df[:log_pz_jaco_enco_train_a1] = m1[12]
    df[:log_pz_jaco_deco_train_a1] = m1[13]

    df[:auc_pxv_x_train_a5] = m5[1]
    df[:auc_pxv_z_train_a5] = m5[2]
    df[:auc_pz_train_a5] = m5[3]
    df[:auc_pz_jaco_enco_train_a5] = m5[4]
    df[:auc_pz_jaco_deco_train_a5] = m5[5]
    df[:auc_pxv_pz_train_a5] = m5[6]
    df[:auc_pxv_pz_jaco_enco_train_a5] = m5[7]
    df[:auc_pxv_pz_jaco_deco_train_a5] = m5[8]
    df[:log_pxv_x_train_a5] = m5[9]
    df[:log_pxv_z_train_a5] = m5[10]
    df[:log_pz_train_a5] = m5[11]
    df[:log_pz_jaco_enco_train_a5] = m5[12]
    df[:log_pz_jaco_deco_train_a5] = m5[13]

    CSV.write(data_folder * "$run_name-large_metrics.csv", df)
end

function compute_metrics(model::SVAE, x, labels)
    z = zparams(model, x)[1].data
    xp = model.g(z).data
    zp = zparams(model, xp)[1].data

    println("computing likelihoods...")
    log_pxv_x = vec(collect(FewShotAnomalyDetection.log_normal(x, xp)'))
    log_pxv_z = vec(collect(sum((z .- zp) .^ 2, dims = 1)'))

    log_pz_ = vec(collect(log_pz(model, x)'))
    log_pz_jaco_enco = vec(collect(log_pz_jacobian_encoder(model, x)'))
    log_pz_jaco_deco = vec(collect(log_pz_jacobian_decoder(model, z)'))

    println("computing aucs...")
    auc_pxv_x = computeauc(.-log_pxv_x, labels)
    auc_pxv_z = computeauc(.-log_pxv_z, labels)
    auc_pz = computeauc(.-log_pz_, labels)
    auc_pz_jaco_enco = computeauc(.-(log_pz_jaco_enco), labels)
    auc_pz_jaco_deco = computeauc(.-(log_pz_jaco_deco), labels)
    auc_pxv_pz = computeauc(.-(log_pxv_x .+ log_pz_), labels)
    auc_pxv_pz_jaco_enco = computeauc(.-(log_pxv_x .+ log_pz_jaco_enco), labels)
    auc_pxv_pz_jaco_deco = computeauc(.-(log_pxv_x .+ log_pz_jaco_deco), labels)

    return auc_pxv_x, auc_pxv_z, auc_pz, auc_pz_jaco_enco, auc_pz_jaco_deco, auc_pxv_pz, auc_pxv_pz_jaco_enco, auc_pxv_pz_jaco_deco, mean(log_pxv_x), mean(log_pxv_z), mean(log_pz_), mean(log_pz_jaco_enco), mean(log_pz_jaco_deco)
end

files = readdir(data_folder)
files = filter(f -> occursin("metrics.csv", f), readdir(data_folder))
if length(ARGS) != 0
    contains = ARGS[1]
    files = filter(f -> occursin(contains, f), files)
end

for f in files
    if isfile(data_folder * f)
        process_file(f)
    end
end
