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

inputDim = 1
hiddenDim = 100
latentDim = 1
numLayers = 2
nonlinearity = "relu"
layerType = "Dense"

# Create data
T = Float64
train = T[10000]
train = randn(T, 1, 5000) .- 1
train = randn(T, 1, 5000) .+ 1

batchSize = 100
numBatches = 10000

encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim * 2, numLayers, nonlinearity, "", layerType))
decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim + 1, numLayers + 1, nonlinearity, "linear", layerType))
outerVAE = VAE(encoder, decoder, T(1), :scalarsigma)
opt = Flux.Optimise.ADAM(3e-4)
cb = Flux.throttle(() -> println("$datasetName outer VAE: $(printingloss(outerVAE, train[1]))"), 5)
Flux.train!((z, y) -> loss(outerVAE, z), Flux.params(outerVAE), RandomBatches((train, zero(train)), batchSize, numBatches), opt, cb = cb)

