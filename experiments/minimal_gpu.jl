using FewShotAnomalyDetection
using UCI
using Flux
using MLDataPattern
using CuArrays


hiddenDim = 32
latentDim = 3
numLayers = 3
nonlinearity = "relu"
layerType = "Dense"
β = 0.01

batchSize = 100
numBatches = 10000

function learn(train, test)
	svae = SVAEtwocaps(size(train[1], 1), hiddenDim, latentDim, numLayers, nonlinearity, layerType, :unit) |> gpu
    learnRepresentation(data) = wloss(svae, data, β, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, 1))
	opt = Flux.Optimise.ADAM(1e-5)
    cb = Flux.throttle(() -> println("SVAE: $(learnRepresentation(train[1]))"), 5)
	# there is a hack with RandomBatches because so far I can't manage to get them to work without the tuple - I have to find a different sampling iterator
    Flux.train!((x, y) -> learnRepresentation(x), Flux.params(svae), RandomBatches((train[1], zero(train[2])), size = batchSize, count = numBatches), opt, cb = cb)
	println("Train err: $(learnRepresentation(train[1])) vs test error: $(learnRepresentation(test[1]))")
end

dataset = "breast-cancer-wisconsin" # name of the UCI dataset. You can find the names in the e.g. in the LODA paper http://agents.fel.cvut.cz/stegodata/pdfs/Pev15-Loda.pdf - last page
dataset = "statlog-vehicle"

data, normal_labels, anomaly_labels = UCI.get_umap_data(dataset)
# This method iterates over all subdataset in a multiclass dataset or it only runs once for a single class dataset
subdatasets = UCI.create_multiclass(data, normal_labels, anomaly_labels)
train = []
test = []
for (subdata, class_label) in subdatasets
		println(dataset*" "*class_label)
		_X_tr, _y_tr, _X_tst, _y_tst = UCI.split_data(subdata, 0.8) #train/test split

		# dirty data normalization
		_X_tr .-= minimum(_X_tr)
		_X_tr ./= maximum(_X_tr)
		_X_tst .-= minimum(_X_tst)
		_X_tst ./= maximum(_X_tst)

		# no need to have it this way but we are used to it from legacy code
		train = (_X_tr, _y_tr) |> gpu
		test = (_X_tst, _y_tst) |> gpu

		# learn(train, test)
end
