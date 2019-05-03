using FewShotAnomalyDetection
using Flux
using MLDataPattern
using FluxExtensions
using Adapt
using DataFrames
using CSV
using Statistics

import FewShotAnomalyDetection: loss, zparams
include("experimentalutils.jl")

function runExperiment(datasetName, train, test, inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, γ, batchSize = 100, numBatches = 10000)
    T = Float32
    svae = SVAEtwocaps(size(train[1], 1), hiddenDim, latentDim, numLayers, nonlinearity, layerType, :scalarsigma)
    FewShotAnomalyDetection.set_normal_μ_nonparam(svae, vcat(T(1), zeros(latentDim - 1)))
    learnRepresentation(data, labels) = wloss(svae, data, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, γ))
    printing_learnRepresentation(data, labels) = printing_wloss(svae, data, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, γ))
    opt = Flux.Optimise.ADAM(1e-4)
    cb = Flux.throttle(() -> println("$datasetName inner SVAE: $(printing_learnRepresentation(train[1], zero(train[2])))"), 10)
    println("Before training - $datasetName inner SVAE: $(printing_learnRepresentation(train[1], zero(train[2])))")
    Flux.train!(learnRepresentation, Flux.params(svae), RandomBatches((train[1], zero(train[2])), batchSize, numBatches), opt, cb = cb)

    pxv = vec(collect(.-pxexpectedz(svae, test[1])'))
    pzs = vec(collect(.-pz(svae, test[1])'))
    pxis = vec(collect(.-px(svae, Flux.Tracker.data(test[1]))'))
    # println(pxis)
    auc_pxv = computeauc(pxv, test[2] .- 1)
    auc_pz = computeauc(pzs, test[2] .- 1)
    auc_pxis = computeauc(pxis, test[2] .- 1)

    println("AUC P(X) Vita: $auc_pxv | AUC P(X) IS: $auc_pxis AUC | learnt prior P(Z): $auc_pz")

    zs = zparams(svae, test[1])[1]
    μz = mean(zs, dims = 2)
    FewShotAnomalyDetection.set_normal_μ(svae, μz)
    pzs = vec(collect(.-pz(svae, test[1])'))
    auc_pz = computeauc(pzs, test[2] .- 1)

    println("AUC mean prior P(Z): $auc_pz")

    return DataFrame(dataset = datasetName, idim = inputDim, hdim = hiddenDim, ldim = latentDim, layers = numLayers, β = β, γ = γ, auc_pxv = auc_pxv, auc_pxis = auc_pxis, auc_pz = auc_pz)
end

outputFolder = mainfolder * "experiments/kernelsizetest/"
mkpath(outputFolder)

datasets = ["breast-cancer-wisconsin", "cardiotocography"]
difficulties = ["easy", "easy"]
batchSize = 100
iterations = 10000

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

for i in 1:10
    for (dn, df) in zip(datasets, difficulties)
        println("Loading $dn : $df")
        train, test, clusterdness = loaddata(dn, df)

        println("$dn")

        evaluateOneConfig = p -> runExperiment(dn, train, test, size(train[1], 1), p..., batchSize, iterations)
        results = gridsearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [1.], [0.001 0.01 0.1 1])

        CSV.write(outputFolder * "kernelsizetest-$dn-$df-$i.csv", vcat(results...))
    end
end