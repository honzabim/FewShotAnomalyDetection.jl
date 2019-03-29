using FewShotAnomalyDetection
using Flux
using MLDataPattern
using FluxExtensions
using Adapt
using DataFrames
using CSV

import FewShotAnomalyDetection: loss, zparams
include("experimentalutils.jl")
include("vae.jl")

function runExperiment(datasetName, train, test, inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, batchSize = 100, numBatches = 10000)

    T = Float32

    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim * 2, numLayers, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim + 1, numLayers + 1, nonlinearity, "linear", layerType))
    outerVAE = VAE(encoder, decoder, β, :scalarsigma)
    opt = Flux.Optimise.ADAM(3e-4)
    cb = Flux.throttle(() -> println("$datasetName outer VAE: $(printingloss(outerVAE, train[1]))"), 5)
    Flux.train!((x, y) -> loss(outerVAE, x), Flux.params(outerVAE), RandomBatches((train[1], zero(train[2])), batchSize, numBatches), opt, cb = cb)

    svae = SVAEtwocaps(latentDim, latentDim, latentDim, numLayers, nonlinearity, layerType, :scalarsigma, T)
    learnRepresentation!(data, labels) = wloss(svae, data, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, 1))
    printing_learnRepresentation!(data, labels) = printing_wloss(svae, data, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, 1))
    opt = Flux.Optimise.ADAM(3e-4)
    cb = Flux.throttle(() -> println("$datasetName inner SVAE: $(printing_learnRepresentation!(samplez(outerVAE, train[1]), zero(train[2])))"), 5)
    Flux.train!((x, y) -> learnRepresentation!(samplez(outerVAE, x), y), Flux.params(svae), RandomBatches((train[1], zero(train[2])), batchSize, numBatches), opt, cb = cb)

    pxv = vec(collect(.-pxexpectedz(svae, zparams(outerVAE, test[1])[1])'))
    pzs = vec(collect(.-pz(svae, zparams(outerVAE, test[1])[1])'))
    auc_pxv = computeauc(pxv, test[2] .- 1)
    auc_pz = computeauc(pzs, test[2] .- 1)

    println("AUC P(X) Vita: $auc_pxv | AUC learnt prior P(Z): $auc_pz")

    zs = zparams(svae, zparams(outerVAE, test[1])[1])[1]
    μz = mean(zs, dims = 2)
    FewShotAnomalyDetection.set_normal_μ(svae, μz)

    pzs = vec(collect(.-pz(svae, zparams(outerVAE, test[1])[1])'))
    auc_pz = computeauc(pzs, test[2] .- 1)

    println("AUC mean prior P(Z): $auc_pz")

    return DataFrame(dataset = datasetName, idim = inputDim, hdim = hiddenDim, ldim = latentDim, layers = numLayers, β = β, auc_pxv = auc_pxv, auc_pz = auc_pz)
end

outputFolder = mainfolder * "experiments/twostagesvae/"
mkpath(outputFolder)

datasets = ["breast-cancer-wisconsin"]
difficulties = ["easy"]
batchSize = 100
iterations = 10000

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

for i in 1:10
    for (dn, df) in zip(datasets, difficulties)
        train, test, clusterdness = loaddata(dn, df)

        println("$dn")

        evaluateOneConfig = p -> runExperiment(dn, train, test, size(train[1], 1), p..., batchSize, iterations)
        results = gridsearch(evaluateOneConfig, [32], [8], [3], ["swish"], ["Dense"], [1.])

        CSV.write(outputFolder * "twostagesvae-$dn-$df-$i.csv", vcat(results...))
    end
end
