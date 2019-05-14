using FewShotAnomalyDetection
using Flux
using MLDataPattern
using FluxExtensions
using Adapt
using DataFrames
using CSV
using Serialization

import FewShotAnomalyDetection: loss, zparams
include("experimentalutils.jl")
include("vae.jl")

outputFolder = mainfolder * "experiments/twostagesvae_scoretest_wide/"
mkpath(outputFolder)

function runExperiment(datasetName, train, test, inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, batchSize = 100, numBatches = 10000, i = 0)

    T = Float32

    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, (latentDim - 1) * 2, numLayers + 1, nonlinearity, "linear", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder((latentDim - 1), hiddenDim, inputDim + 1, numLayers + 1, nonlinearity, "linear", layerType))
    outerVAE = VAE(encoder, decoder, β, :scalarsigma)
    opt = Flux.Optimise.ADAM(1e-4)
    cb = Flux.throttle(() -> println("$datasetName outer VAE: $(printingloss(outerVAE, train[1]))"), 5)
    Flux.train!((x, y) -> loss(outerVAE, x), Flux.params(outerVAE), RandomBatches((train[1], zero(train[2])), batchSize, numBatches), opt, cb = cb)

    svae = SVAEtwocaps(latentDim - 1, hiddenDim, latentDim, numLayers, nonlinearity, layerType, :scalarsigma, T)
    FewShotAnomalyDetection.set_normal_μ_nonparam(svae, vcat(T(1), zeros(latentDim - 1)))
    learnRepresentation!(data, labels) = wloss(svae, data, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, 1))
    printing_learnRepresentation!(data, labels) = printing_wloss(svae, data, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, 1))
    opt = Flux.Optimise.ADAM(1e-4)
    cb = Flux.throttle(() -> println("$datasetName inner SVAE: $(printing_learnRepresentation!(samplez(outerVAE, train[1]), zero(train[2])))"), 5)
    println("Before training - $datasetName inner SVAE: $(printing_learnRepresentation!(samplez(outerVAE, train[1]), zero(train[2])))")
    Flux.train!((x, y) -> learnRepresentation!(samplez(outerVAE, x), y), Flux.params(svae), RandomBatches((train[1], zero(train[2])), batchSize, numBatches), opt, cb = cb)


    data = Flux.Tracker.data(zparams(outerVAE, test[1])[1])
    log_pxv = vec(collect(.-log_pxexpectedz(svae, data)'))
    log_pzs = vec(collect(.-log_pz(svae, data)'))
    log_pxis = vec(collect(.-log_px(svae, data)'))
    log_pzs_jacobian = vec(collect(.-log_pz_jacobian(svae, data)'))

    cz = closestz(svae, data)
    mz = manifoldz(svae, data)

    log_pxv_cz = vec(collect(.-log_pxexpectedz(svae, data, cz)'))
    log_pzs_cz = vec(collect(.-log_pz_from_z(svae, cz)'))
    log_pzs_jacobian_cz = vec(collect(.-log_pz_jacobian(svae, data, cz)'))

    log_pxv_mz = vec(collect(.-log_pxexpectedz(svae, data, mz)'))
    log_pzs_mz = vec(collect(.-log_pz_from_z(svae, mz)'))
    log_pzs_jacobian_mz = vec(collect(.-log_pz_jacobian(svae, data, mz)'))


    auc_pxv = computeauc(log_pxv, test[2] .- 1)
    auc_pz = computeauc(log_pzs, test[2] .- 1)
    auc_pxis = computeauc(log_pxis, test[2] .- 1)
    auc_pz_jacobian = computeauc(log_pzs_jacobian, test[2] .- 1)
    auc_pxv_pz = computeauc(log_pxv .+ log_pzs, test[2] .- 1)
    auc_pxv_pz_jacobian = computeauc(log_pxv .+ log_pzs_jacobian, test[2] .- 1)

    auc_pxv_cz = computeauc(log_pxv_cz, test[2] .- 1)
    auc_pz_cz = computeauc(log_pzs_cz, test[2] .- 1)
    auc_pz_jacobian_cz = computeauc(log_pzs_jacobian_cz, test[2] .- 1)
    auc_pxv_pz_cz = computeauc(log_pxv_cz .+ log_pzs_cz, test[2] .- 1)
    auc_pxv_pz_jacobian_cz = computeauc(log_pxv_cz .+ log_pzs_jacobian_cz, test[2] .- 1)

    auc_pxv_mz = computeauc(log_pxv_mz, test[2] .- 1)
    auc_pz_mz = computeauc(log_pzs_mz, test[2] .- 1)
    auc_pz_jacobian_mz = computeauc(log_pzs_jacobian_mz, test[2] .- 1)
    auc_pxv_pz_mz = computeauc(log_pxv_mz .+ log_pzs_mz, test[2] .- 1)
    auc_pxv_pz_jacobian_mz = computeauc(log_pxv_mz .+ log_pzs_jacobian_mz, test[2] .- 1)

    serialize(outputFolder * "twostagesvae-$datasetName-$i-svae.jls", svae)
    serialize(outputFolder * "twostagesvae-$datasetName-$i-vae.jls", outerVAE)
    serialize(outputFolder * "twostagesvae-$datasetName-$i-train.jls", train)
    serialize(outputFolder * "twostagesvae-$datasetName-$i-test.jls", test)

    return DataFrame(dataset = datasetName, idim = inputDim, hdim = hiddenDim, ldim = latentDim, layers = numLayers, i = i, auc_pxv = auc_pxv, auc_pz = auc_pz, auc_pxis = auc_pxis, auc_pz_jacobian = auc_pz_jacobian,
                    auc_pxv_pz = auc_pxv_pz, auc_pxv_pz_jacobian = auc_pxv_pz_jacobian, auc_pxv_cz = auc_pxv_cz, auc_pz_cz = auc_pz_cz, auc_pz_jacobian_cz = auc_pz_jacobian_cz, auc_pxv_pz_cz = auc_pxv_pz_cz,
                    auc_pxv_pz_jacobian_cz = auc_pxv_pz_jacobian_cz, auc_pxv_mz = auc_pxv_mz, auc_pz_mz = auc_pz_mz, auc_pz_jacobian_mz = auc_pz_jacobian_mz, auc_pxv_pz_mz = auc_pxv_pz_mz, auc_pxv_pz_jacobian_mz = auc_pxv_pz_jacobian_mz)
end

datasets = ["breast-cancer-wisconsin"]
difficulties = ["easy"]
batchSize = 200
iterations = 10000

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

for i in 1:5
    for (dn, df) in zip(datasets, difficulties)
        train, test, clusterdness = loaddata(dn, df)

        println("$dn")

        evaluateOneConfig = p -> runExperiment(dn, train, test, size(train[1], 1), p..., batchSize, iterations, i)
        results = gridsearch(evaluateOneConfig, [64], [3], [3], ["swish"], ["Dense"], [1.])

        CSV.write(outputFolder * "twostagesvae-$dn-$i.csv", vcat(results...))
    end
end
