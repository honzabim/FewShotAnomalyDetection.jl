using FewShotAnomalyDetection
using UCI
using Flux
using MLDataPattern
using IPMeasures
using IPMeasures: crit_mmd2_var, crit_mxy_over_mltpl, crit_mxy_over_sum


hiddenDim = 32
latentDim = 3
numLayers = 3
nonlinearity = "swish"
layerType = "Dense"
β = 1
num_pseudoinputs = 1

batchSize = 100
numBatches = 1000

# dataset = "breast-cancer-wisconsin" # name of the UCI dataset. You can find the names in the e.g. in the LODA paper http://agents.fel.cvut.cz/stegodata/pdfs/Pev15-Loda.pdf - last page
# # dataset = "statlog-vehicle"
#
# data, normal_labels, anomaly_labels = UCI.get_umap_data(dataset)
# # This method iterates over all subdataset in a multiclass dataset or it only runs once for a single class dataset
# subdatasets = UCI.create_multiclass(data, normal_labels, anomaly_labels)
# (subdata, class_label) = subdatasets[1]
# println(dataset*" "*class_label)
# _X_tr, _y_tr, _X_tst, _y_tst = UCI.split_data(subdata, 0.8) #train/test split
#
# # dirty data normalization
# _X_tr .-= minimum(_X_tr)
# _X_tr ./= maximum(_X_tr)
# _X_tst .-= minimum(_X_tst)
# _X_tst ./= maximum(_X_tst)
#
# # no need to have it this way but we are used to it from legacy code
# train = (_X_tr, _y_tr)
# test = (_X_tst, _y_tst)

M = 2
N = 1000
s = Float32.([30 10; 10 1])
X = s*randn(Float32,M,N) .+ 100

svae = SVAEvampmeans(size(X, 1), hiddenDim, latentDim, numLayers, nonlinearity, layerType, num_pseudoinputs)



criterion = crit_mxy_over_mltpl
criterion = crit_mxy_over_sum
criterion = crit_mmd2_var

opt = Flux.Optimise.RMSProp(1e-4)
for i in 1:15
# Find the best width of the mmd kernel
z = FewShotAnomalyDetection.samplez(svae, zparams(svae, X)...).data
zp = FewShotAnomalyDetection.sampleVamp(svae, size(z, 2)).data
γs = -10:0.05:2
cs = [criterion(IPMeasures.IMQKernel(10.0 ^ γ), z, zp, IPMeasures.pairwisecos) for γ in γs]
γ = 10 ^ γs[argmax(cs)]
println("We chose kernal size $γ")

learnRepresentation(data) = wloss(svae, data, β, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, γ))
cb = Flux.throttle(() -> println("SVAE: $(learnRepresentation(X))"), 10)
# there is a hack with RandomBatches because so far I can't manage to get them to work without the tuple - I have to find a different sampling iterator
Flux.train!(learnRepresentation, Flux.params(svae), RandomBatches((X,), size = batchSize, count = numBatches), opt, cb = cb)
end


s = [700, 700]
#Plotting

using Plots
plotlyjs()
scatter(X[1, :], X[2, :], size = s)
mus = Flux.Tracker.data(zparams(svae, X)[1])
xgz = Flux.Tracker.data(svae.g(mus))
scatter!(xgz[1, :], xgz[2, :], size = s)

scatter3d(mus[1, :], mus[2, :], mus[3, :], size = s)
zs = Flux.Tracker.data(FewShotAnomalyDetection.zfromx(svae, X))
scatter3d(zs[1, :], zs[2, :], zs[3, :], size = s, title = "sampled Z")

(μ, k) = zparams(svae, svae.pseudo_inputs)
lkhs = FewShotAnomalyDetection.pz(svae, X)
scatter3d(mus[1, :], mus[2, :], mus[3, :], zcolor = vec(lkhs), size = s)
scatter3d!(μ[1, :].data, μ[2, :].data, μ[3, :].data, color = "green", m = :xcross, markersize = 5, title = "Z means")

xs = minimum(X[1, :]):1:maximum(X[1, :])
ys = minimum(X[2, :]):1:maximum(X[2, :])

score = (x, y) -> FewShotAnomalyDetection.as_jacobian(svae, [x, y])
fillc = true
nlevels = 20
csvae = contour(xs, ys, score, fill = fillc, levels = nlevels, title = "Jacobian score")
scatter!(csvae, X[1, :], X[2, :], color = "green", alpha = 0.5)

score = (x, y) -> FewShotAnomalyDetection.pxexpectedz(svae, [x, y])[1]
csvae = contour(xs, ys, score, fill = fillc, levels = nlevels, title = "P(X) Vita")
scatter!(csvae, X[1, :], X[2, :], alpha = 0.5)

score = (x, y) -> log(FewShotAnomalyDetection.pz(svae, [x, y])[1])
csvae = contour(xs, ys, score, fill = fillc, levels = nlevels, title = "P(z)")
scatter!(csvae, X[1, :], X[2, :], alpha = 0.5)

score = (x, y) -> log(FewShotAnomalyDetection.pz(svae, [x, y])[1]) + FewShotAnomalyDetection.pxexpectedz(svae, [x, y])[1]
csvae = contour(xs, ys, score, fill = fillc, levels = nlevels, title = "P(X) + P(z)")
scatter!(csvae, X[1, :], X[2, :], alpha = 0.5)



# # one learning step
# for i in 1:1000
# @show l = learnRepresentation(X)
# Flux.Tracker.back!(l)
# for p in params(svae)
#    if any(isnan.(p.grad))
#       println("$p has gradient NaN")
#    end
#    Δ = Flux.Optimise.apply!(opt, p.data, p.grad)
#    p.data .-= Δ
#    p.grad .= 0
# end
# end
