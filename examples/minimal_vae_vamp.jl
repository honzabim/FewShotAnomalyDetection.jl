using FewShotAnomalyDetection
using UCI
using Flux
using MLDataPattern
using IPMeasures
using Adapt
using FluxExtensions

hiddenDim = 32
latentDim = 3
numLayers = 3
nonlinearity = "relu"
layerType = "Dense"
β = 0.1 # ratio between reconstruction error and the distance between p(z) and q(z)
α = 0.01 # weight of anomalies in the loss function (normal samples will have (1 - α))
γ = 0.001 # width of the IMQ kernel in the loss function
ncentroids = 10 # initial number of remembered centroids that describe the prior dist.

batchSize = 100
numBatches = 1000

dataset = "breast-cancer-wisconsin" # name of the UCI dataset. You can find the names in the e.g. in the LODA paper http://agents.fel.cvut.cz/stegodata/pdfs/Pev15-Loda.pdf - last page
# dataset = "statlog-vehicle"

data, normal_labels, anomaly_labels = UCI.get_umap_data(dataset)
# This method iterates over all subdataset in a multiclass dataset or it only runs once for a single class dataset
subdatasets = UCI.create_multiclass(data, normal_labels, anomaly_labels)
subdata, class_label = subdatasets[1]
println(dataset*" "*class_label)
_X_tr, _y_tr, _X_tst, _y_tst = UCI.split_data(subdata, 0.8) #train/test split

# dirty data normalization - I would suggest something that makes them scale from -1 to 1. In a perfect world it would be centered at 0
all_data = hcat(_X_tr, _X_tst)
min_x = minimum(all_data, dims = 2)
max_x = maximum(all_data, dims = 2)
_X_tr .-= min_x
_X_tst .-= min_x
_X_tr ./= (max_x .- min_x) 
_X_tst ./= (max_x .- min_x)

# no need to have it this way but we are used to it from legacy code
train = (_X_tr, _y_tr)
test = (_X_tst, _y_tst)

##############################################
# VAE with VAMP
##############################################
inputdim = size(train[1], 1)
encoder = Adapt.adapt(Float32, FluxExtensions.layerbuilder(inputdim, hiddenDim, latentDim, numLayers + 1, nonlinearity, "linear", layerType))
decoder = Adapt.adapt(Float32, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputdim, numLayers + 1, nonlinearity, "linear", layerType))
centroids = rand(Float32, inputdim, ncentroids)
labels = zeros(Int, ncentroids)
vae = VAEvamp(encoder, decoder, centroids, labels)

# Basic Wasserstein loss to train the vae on unlabelled data
learn_without_labels(data) = wloss(vae, data, (x, y) -> IPMeasures.mmd(IPMeasures.IMQKernel(γ), x, y), β)

# Loss that computes different distance for normal and anomalous samples. In order to do so, it samples the same amount of samples
# from the prior as there are data with given label. Therefore, watch out for not having enough of anomalies in your batch!
# My recommended use would be to make sure that your batch !ALWAYS! has the same and sufficient amount of normal and anomalous samples
# and you use the weight alpha to correct their respective power of the gradient by setting it to expected ratio of anomalies in your dataset
learn_with_labels(data, labels) = wloss(vae, data, labels, (x, y) -> IPMeasures.mmd(IPMeasures.IMQKernel(γ), x, y), β, α)

# Unsupervised learning
opt = Flux.Optimise.ADAM(1e-4) # Watch our, the learning rate may have a lot of impact on your learning...
cb = Flux.throttle(() -> println("VAE - loss: $(learn_without_labels(train[1]))"), 5)
Flux.@epochs 10 Flux.train!(learn_without_labels, Flux.params(vae), RandomBatches((train[1],), size = batchSize, count = numBatches), opt, cb = cb)
println("Train err: $(learn_without_labels(train[1])) vs test error: $(learn_without_labels(test[1]))")

# Adding some anomal centroids to the centroids memory of VAE
anomalies = _X_tst[:, _y_tst .== 1]
for i in 1:size(anomalies, 2)
    add_labeled_centroid!(vae, anomalies[:, i], 1)
end

# This is clearly wrong for real learning but since the UCI datasets have no anomaly in the train set, I put the together just to have some reasonable data...
# so now I mixed trianing normal samples + anomalies from test so the score on test doesn't make any sense...
fit_times = div(size(train[1], 2), size(anomalies, 2)) + 1
anomalies = collect(repeat(anomalies, 1, fit_times)[:, 1:size(train[1], 2)])

# learn with labels
cb = Flux.throttle(() -> println("VAE loss on test: $(learn_with_labels(test[1], test[2]))"), 10)
loss(normal, anomalous) = learn_with_labels(hcat(normal, anomalous), vcat(zeros(Int, size(normal, 2)), ones(Int, size(anomalous, 2))))
Flux.@epochs 10 Flux.train!(loss, Flux.params(vae), RandomBatches((train[1], anomalies), size = batchSize, count = numBatches), opt, cb = cb)