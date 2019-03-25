using FewShotAnomalyDetection
using Adapt
using Flux
using MLDataPattern
using JLD2
using FileIO
using FluxExtensions
using ADatasets
using StatsBase

folderpath = "D:/dev/julia/"
# folderpath = "/home/bimjan/dev/julia/

"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridSearch(f, parameters...) = map(p -> printAndRun(f, p), Base.product(parameters...))

function printAndRun(f, p)
    println(p)
    (p, f(p))
end

function createSVAEWithMem(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, memorySize, k, labelCount, β, α = 0.1, T = Float64)
    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))

    svae = SVAEbase(encoder, decoder, hiddenDim, latentDim, T)
    # train!, classify, trainOnLatent! = augmentModelWithMemory((x) -> zfromx(svae, x), memorySize, latentDim, k, labelCount, α, T)

    function learnRepresentation!(data, labels)
        # return wloss(svae, data, β, (x, y) -> mmd_imq(x, y, 1))
        return loss(svae, data, β)
    end

    function learnAnomaly!(data, labels)
        # trainOnLatent!(zfromx(svae, data), labels)
        return wloss(svae, data, β, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, 1))
        # return loss(svae, data, β)
    end

    return svae, learnRepresentation!, learnAnomaly!
end

function runExperiment(datasetName, train, test, createModel, anomalyCounts, batchSize = 100, numBatches = 10000)
    (model, learnRepresentation!, learnAnomaly!) = createModel()
    opt = Flux.Optimise.ADAM(Flux.params(model), 1e-3)
    cb = Flux.throttle(() -> println("$datasetName : $(learnAnomaly!(train[1], zero(train[2]) .+ 2))"), 5)
    Flux.train!(learnAnomaly!, RandomBatches((train[1], zero(train[2]) .+ 2), batchSize, numBatches), opt, cb = cb)
    # FluxExtensions.learn(learnRepresentation!, opt, RandomBatches((train[1], train[2] .- 1), batchSize, numBatches), ()->(), 100)

    learnRepresentation!(train[1], train[2] .- 1)

    results = []
    anomalies = train[1][:, train[2] .- 1 .== 1] # TODO needs to be shuffled!!!
    for ac in anomalyCounts
        if ac <= size(anomalies, 2)
            l = learnAnomaly!(anomalies[:, ac], [1])
        else
            break;
        end

        values, probScore = classify(test[1])
        values = Flux.Tracker.data(values)
        probScore = Flux.Tracker.data(probScore)

        rocData = roc(test[2] .- 1, values)
        f1 = f1score(rocData)
        # auc = EvalCurves.auc(EvalCurves.roccurve(probScore, test[2] .- 1)...)
        auc = pyauc(test[2] .- 1, probScore)
        push!(results, (ac, f1, auc, values, probScore, rstrn, rstst, knnauc, knnprec, knnrecall))
    end
    return results
end

outputFolder = folderpath * "OSL/experiments/NewImplementationTests/"
mkpath(outputFolder)

# datasets = ["breast-cancer-wisconsin", "sonar", "wall-following-robot", "waveform-1"]
datasets = ["abalone"]
difficulties = ["easy"]
const dataPath = folderpath * "data/loda/public/datasets/numerical"
batchSize = 100
iterations = 10000

loadData(datasetName, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(datasetName, difficulty, dataPath)..., 0.8, "low")

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

for (dn, df) in zip(datasets, difficulties)
    train, test, clusterdness = loadData(dn, df)

    println("$dn")
    println("Running svae...")

    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAEWithMem(size(train[1], 1), p...), 1:5, batchSize, iterations)
    results = gridSearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [16 32 64 128 256], [16 32], [1], [0.01 0.05 0.1])
    results = reshape(results, length(results), 1)
    save(outputFolder * dn * "-svae.jld2", "results", results)
end
