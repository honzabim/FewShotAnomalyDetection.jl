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
    df = CSV.read(data_folder * "$run_name-metrics.csv")
    if :log_pz_train in names(df)
        println("Skipping $f because it has logs added already...")
        return
    end

    svae = deserialize(data_folder * "$run_name-svae.jls")
    (x_train, labels_train) = deserialize(data_folder * "$dataset-$i-train.jls")
    (x_test, labels_test) = deserialize(data_folder * "$dataset-$i-test.jls")

    println("$f: computing metrics...")
    log_pz_train, log_pz_jaco_enco_train, log_pz_jaco_deco_train = compute_metrics(svae, x_train, labels_train .- 1)
    log_pz_test, log_pz_jaco_enco_test, log_pz_jaco_deco_test = compute_metrics(svae, x_test, labels_test .- 1)
    
    println("$f: saving data...")
    df[:log_pz_train] = log_pz_train
    df[:log_pz_jaco_enco_train] = log_pz_jaco_enco_train
    df[:log_pz_jaco_deco_train] = log_pz_jaco_deco_train
    df[:log_pz_test] = log_pz_test
    df[:log_pz_jaco_enco_test] = log_pz_jaco_enco_test
    df[:log_pz_jaco_deco_test] = log_pz_jaco_deco_test
    CSV.write(data_folder * "$run_name-metrics.csv", df)
end

function compute_metrics(model, x, labels)
    z = zparams(model, x)[1].data
    xp = model.g(z).data
    zp = zparams(model, xp)[1].data

    println("computing likelihoods...")
    log_pxv_x = vec(collect(FewShotAnomalyDetection.log_normal(x, xp)'))
    log_pxv_z = vec(collect(sum((z .- zp) .^ 2, dims = 1)'))

    log_pz_ = vec(collect(log_pz(model, x)'))
    log_pz_jaco_enco = vec(collect(log_pz_jacobian_encoder(model, x)'))
    log_pz_jaco_deco = vec(collect(log_pz_jacobian_decoder(model, z)'))

    return mean(log_pz_), mean(log_pz_jaco_enco), mean(log_pz_jaco_deco)
end

files = readdir(data_folder)
# files = readdir(data_folder)[1]
for f in files
    if isfile(data_folder * f) && occursin("metrics.csv", f)
        process_file(f)
    end
end
