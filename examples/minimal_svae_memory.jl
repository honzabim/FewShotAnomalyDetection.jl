using FewShotAnomalyDetection
using UCI
using Flux
using MLDataPattern


hiddenDim = 32
latentDim = 3
numLayers = 3
nonlinearity = "relu"
layerType = "Dense"
β = 0.1 # ratio between reconstruction error and the distance between p(z) and q(z)
α = 0.1 # threshold in the memory that does not matter to us at the moment!
loss_α = 0.1 # importance ratio between anomalies and normal data in mem_loss
memorySize = 256
k = 256
labelCount = 1


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
trainWithAnomalies(data, labels) = FewShotAnomalyDetection.mem_wloss(svae, mem, data, labels, β, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, 1), loss_α)

# Unsupervised learning
opt = Flux.Optimise.ADAM(1e-5)
cb = Flux.throttle(() -> println("SVAE: $(trainRepresentation(train[1]))"), 5)
Flux.train!((x) -> trainRepresentation(getobs(x)), Flux.params(svae), RandomBatches((train[1],), size = batchSize, count = numBatches), opt, cb = cb)
println("Train err: $(trainRepresentation(train[1])) vs test error: $(trainRepresentation(test[1]))")

# Adding stuff into the memory
remember(train[1], train[2])
remember(test[1][:, 230:end], test[2][230:end]) # we don't have anomalies in train for UCI datasets :( and let's add just a couple so we don't have to wait so long

numBatches = 1000 # it will take a looong time

# learn with labels
cb = Flux.throttle(() -> println("SVAE mem loss: $(trainWithAnomalies(test[1], test[2]))"), 60)
Flux.train!(trainWithAnomalies, Flux.params(svae), RandomBatches((test[1], test[2]), size = batchSize, count = numBatches), opt, cb = cb)
