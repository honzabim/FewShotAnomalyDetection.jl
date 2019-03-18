using FewShotAnomalyDetection
using Flux
using MLDataPattern

function createSVAE(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, T = Float32)
    svae = SVAEbase(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, T)
    learnRepresentation!(data, labels) = loss(svae, data, β)
    return svae, learnRepresentation!
end

function runExperiment(datasetName, train, test, createModel, anomalyCounts, batchSize = 100, numBatches = 10000)
    (model, learnRepresentation!) = createModel()
    opt = Flux.Optimise.ADAM()
    cb = Flux.throttle(() -> println("$datasetName : $(learnRepresentation!(train[1], zero(train[2]) .+ 2))"), 5)
    Flux.train!(learnRepresentation!, Flux.params(model), RandomBatches((train[1], zero(train[2]) .+ 2), batchSize, numBatches), opt, cb = cb)

    learnRepresentation!(train[1], train[2] .- 1)
end

outputFolder = FewShotAnomalyDetection.mainfolder * "OSL/experiments/NewImplementationTests/"
mkpath(outputFolder)

datasets = ["abalone"]
difficulties = ["easy"]
batchSize = 100
iterations = 10000

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

for (dn, df) in zip(datasets, difficulties)
    train, test, clusterdness = loaddata(dn, df)

    println("$dn")
    println("Running svae...")

    evaluateOneConfig = p -> runExperiment(dn, train, test, () -> createSVAE(size(train[1], 1), p...), 1:5, batchSize, iterations)
    results = gridsearch(evaluateOneConfig, [32], [8], [3], ["relu"], ["Dense"], [0.01 0.05 0.1])
end
