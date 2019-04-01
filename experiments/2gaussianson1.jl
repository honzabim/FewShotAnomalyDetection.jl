using FewShotAnomalyDetection
using Flux
using MLDataPattern
using FluxExtensions
using Adapt
using DataFrames
using CSV
using Plots
plotly()

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
T = Float32
train = zeros(T, 1, 10000)
train[1:5000] = randn(T, 1, 5000) .- 3
train[5001:end] = randn(T, 1, 5000) .+ 3
histogram(vec(train), label = "data")

batchSize = 100
numBatches = 10000

encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim * 2, numLayers + 1, nonlinearity, "linear", layerType))
decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim + 1, numLayers + 1, nonlinearity, "linear", layerType))
outerVAE = VAE(encoder, decoder, T(1.), :scalarsigma)
opt = Flux.Optimise.ADAM(1e-3)
cb = Flux.throttle(() -> println("VAE: $(printingloss(outerVAE, train))"), 5)
Flux.train!((x, y) -> loss(outerVAE, x), Flux.params(outerVAE), RandomBatches((train, zero(train)), batchSize, numBatches), opt, cb = cb)

histogram(vec(Flux.Tracker.data(zparams(outerVAE, train)[1])), label = "means z")
histogram(vec(Flux.Tracker.data(samplez(outerVAE, train))), label = "sampled z")
