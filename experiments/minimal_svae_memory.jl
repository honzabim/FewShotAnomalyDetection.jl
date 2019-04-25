using FewShotAnomalyDetection
using UCI
using Flux
using MLDataPattern


hiddenDim = 32
latentDim = 3
numLayers = 3
nonlinearity = "relu"
layerType = "Dense"
β = 0.1
α = 0.1
loss_α = 0.1
memorySize = 256
k = 256
labelCount = 1


batchSize = 100
numBatches = 10000

dataset = "breast-cancer-wisconsin" # name of the UCI dataset. You can find the names in the e.g. in the LODA paper http://agents.fel.cvut.cz/stegodata/pdfs/Pev15-Loda.pdf - last page
dataset = "statlog-vehicle"

data, normal_labels, anomaly_labels = UCI.get_umap_data(dataset)
# This method iterates over all subdataset in a multiclass dataset or it only runs once for a single class dataset
subdatasets = UCI.create_multiclass(data, normal_labels, anomaly_labels)
subdata, class_label = subdatasets[1]
println(dataset*" "*class_label)
_X_tr, _y_tr, _X_tst, _y_tst = UCI.split_data(subdata, 0.8) #train/test split

# dirty data normalization
_X_tr .-= minimum(_X_tr)
_X_tr ./= maximum(_X_tr)
_X_tst .-= minimum(_X_tst)
_X_tst ./= maximum(_X_tst)

# no need to have it this way but we are used to it from legacy code
train = (_X_tr, _y_tr)
test = (_X_tst, _y_tst)

##############################################
# SVAE itself
##############################################
inputdim = size(train[1], 1)
svae = SVAEbase(inputdim, hiddenDim, latentDim, numLayers, nonlinearity, layerType)
mem = KNNmemory{Float32}(memorySize, inputdim, k, labelCount, (x) -> zparams(svae, x)[1], α)

# Basic Wasserstein loss to train the svae on unlabelled data
trainRepresentation(data) = wloss(svae, data, β, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, 1))
# inserts the data into the memory
remember(data, labels) = trainQuery!(mem, data, labels)
# Expects anomalies in the data with correct label (some of them
trainWithAnomalies(data, labels) = mem_wloss(svae, mem, data, labels, β, (x, y) -> mmd_imq(x, y, 1), loss_α)

# Unsupervised learning
opt = Flux.Optimise.ADAM(1e-5)
cb = Flux.throttle(() -> println("SVAE: $(trainRepresentation(train[1]))"), 5)
# there is a hack with RandomBatches because so far I can't manage to get them to work without the tuple - I have to find a different sampling iterator
Flux.train!((x) -> trainRepresentation(getobs(x)), Flux.params(svae), RandomBatches((train[1],), size = batchSize, count = numBatches), opt, cb = cb)
println("Train err: $(trainRepresentation(train[1])) vs test error: $(trainRepresentation(test[1]))")

# Adding stuff into the memory
remember(train[1], train[2])
remember(test[1], test[2]) # we don't have anomalies in train for UCI datasets :(

# learn with labels
cb = Flux.throttle(() -> println("SVAE: $(trainWithAnomalies(test[1]))"), 5)
# there is a hack with RandomBatches because so far I can't manage to get them to work without the tuple - I have to find a different sampling iterator
Flux.train!((x) -> trainWithAnomalies(getobs(x)), Flux.params(svae), RandomBatches((test[1], test[2]), size = batchSize, count = numBatches), opt, cb = cb)
