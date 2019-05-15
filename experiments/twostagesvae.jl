using FewShotAnomalyDetection
using Flux
using MLDataPattern
using FluxExtensions
using Adapt
using DataFrames
using CSV
using Serialization
using IPMeasures
using IPMeasures: crit_mmd2_var
using Plots
plotlyjs()

include("experimentalutils.jl")
include("vae.jl")

outputFolder = mainfolder * "experiments/twostagesvae_scoretest_mmdpval/"
mkpath(outputFolder)

s = SVAEtwocaps(8, 8, 8, 2, "relu", "Dense", :scalarsigma, Float32)
FewShotAnomalyDetection.set_normal_μ_nonparam(s, vcat(Float32(1), zeros(7)))
samplepz(n::Int) = FewShotAnomalyDetection.samplez(s, ones(8, n) .* FewShotAnomalyDetection.normalizecolumns(s.priorμ), ones(1, n) .* s.priorκ)

n = 1000
z1 = samplepz(n)
z2 = samplepz(n)
γs = -10:0.05:2
cs = [crit_mmd2_var(IPMeasures.IMQKernel(10.0 ^ γ), z1, z2, IPMeasures.pairwisecos) for γ in γs]
plot(cs)
global gγ = 10 ^ γs[argmax(cs)]
println("We chose kernal size $gγ for p(z) null hyp")
null_dst = zeros(Float64, n)
for i in 1:n
    null_dst[i] = FewShotAnomalyDetection.mmd_imq(samplepz(n), samplepz(n), gγ)
end
sort!(null_dst)
mmdpval(x) = searchsortedfirst(null_dst, x) / n



function runExperiment(datasetName, train, test, inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, batchSize = 100, numBatches = 10000, i = 0)

    T = Float32

    encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, (latentDim - 1) * 2, numLayers + 1, nonlinearity, "linear", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder((latentDim - 1), hiddenDim, inputDim + 1, numLayers + 1, nonlinearity, "linear", layerType))
    outerVAE = VAE(encoder, decoder, β, :scalarsigma)
    opt = Flux.Optimise.ADAM(1e-4)
    cb = Flux.throttle(() -> println("$datasetName outer VAE: $(printingloss(outerVAE, train[1]))"), 5)
    Flux.train!((x, y) -> loss(outerVAE, x), Flux.params(outerVAE), RandomBatches((train[1], zero(train[2])), batchSize, numBatches), opt, cb = cb)

    svae = SVAEtwocaps(latentDim - 1, hiddenDim, latentDim, numLayers, nonlinearity, layerType, :scalarsigma, T)
    FewShotAnomalyDetection.set_normal_μ_nonparam(svae, vcat(T(1), zeros(latentDim - 1)))
    opt = Flux.Optimise.RMSProp(1e-4)

    for i in 1:(numBatches ÷ 1000)
        z = FewShotAnomalyDetection.samplez(svae, FewShotAnomalyDetection.zparams(svae, zparams(outerVAE, train[1])[1])...).data
        zp = FewShotAnomalyDetection.samplez(svae, ones(latentDim, size(train[1], 2)) .* FewShotAnomalyDetection.normalizecolumns(svae.priorμ), ones(1, size(train[1], 2)) .* svae.priorκ)
        γs = -10:0.05:2
        cs = [crit_mmd2_var(IPMeasures.IMQKernel(10.0 ^ γ), z, zp, IPMeasures.pairwisecos) for γ in γs]
        gamma = 10 ^ γs[argmax(cs)]
        println("We chose kernal size $gamma for svae learning")
        learnRepresentation!(data, labels) = wloss(svae, data, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, gamma))
        printing_learnRepresentation!(data, labels) = printing_wloss(svae, data, (x, y) -> FewShotAnomalyDetection.mmd_imq(x, y, gamma))
        cb = Flux.throttle(() -> println("$datasetName inner SVAE: $(printing_learnRepresentation!(samplez(outerVAE, train[1]), zero(train[2])))"), 5)
        Flux.train!((x, y) -> learnRepresentation!(samplez(outerVAE, x), y), Flux.params(svae), RandomBatches((train[1], zero(train[2])), batchSize, 1000), opt, cb = cb)
    end

    ztrain = Flux.Tracker.data(FewShotAnomalyDetection.zparams(svae, zparams(outerVAE, train[1])[1])[1])
    ztest = Flux.Tracker.data(FewShotAnomalyDetection.zparams(svae, zparams(outerVAE, test[1])[1])[1])
    ztrain_dst = FewShotAnomalyDetection.mmd_imq(samplepz(n), ztrain, gγ)
    ztest_dst = FewShotAnomalyDetection.mmd_imq(samplepz(n), ztest, gγ)
    ztrain_mmd_pval = mmdpval(ztrain_dst)
    ztest_mmd_pval = mmdpval(ztest_dst)

    data = Flux.Tracker.data(zparams(outerVAE, test[1])[1])
    zs = Flux.Tracker.data(FewShotAnomalyDetection.zparams(svae, data)[1])
    us = Flux.Tracker.data(svae.g(zs)[1:end-1, :])
    log_pxv = vec(collect(log_pxexpectedz(outerVAE, test[1], us)'))
    log_pzs = vec(collect(log_pz(svae, data)'))
    log_pzs_jacobian_enc = vec(collect(log_pz_jacobian_encoder(svae, data)'))
    log_pzs_jacobian_dec = vec(collect(log_pz_jacobian_decoder(svae, zs)'))
    log_det_jac_enc_vae = vec(collect(log_det_jacobian_encoder(outerVAE, test[1])'))
    log_det_jac_dec_vae = vec(collect(log_det_jacobian_decoder(outerVAE, us)'))

    auc_pxv = computeauc(.-log_pxv, test[2] .- 1)
    auc_pz = computeauc(.-log_pzs, test[2] .- 1)
    auc_pz_jacobian_enc = computeauc(.-(log_pzs_jacobian_enc .+ log_det_jac_enc_vae), test[2] .- 1)
    auc_pz_jacobian_dec = computeauc(.-(log_pzs_jacobian_dec .- log_det_jac_dec_vae), test[2] .- 1)
    auc_pxv_pz = computeauc(.-(log_pxv .+ log_pzs), test[2] .- 1)
    auc_pxv_pz_jacobian_enc = computeauc(.-(log_pxv .+ log_pzs_jacobian_enc .+ log_det_jac_enc_vae), test[2] .- 1)
    auc_pxv_pz_jacobian_dec = computeauc(.-(log_pxv .+ log_pzs_jacobian_dec .- log_det_jac_dec_vae), test[2] .- 1)

    serialize(outputFolder * "twostagesvae-$datasetName-$i-svae.jls", svae)
    serialize(outputFolder * "twostagesvae-$datasetName-$i-vae.jls", outerVAE)
    serialize(outputFolder * "twostagesvae-$datasetName-$i-train.jls", train)
    serialize(outputFolder * "twostagesvae-$datasetName-$i-test.jls", test)

    return DataFrame(dataset = datasetName, idim = inputDim, hdim = hiddenDim, ldim = latentDim, layers = numLayers, i = i, auc_pxv = auc_pxv, auc_pz = auc_pz, auc_pz_jacobian_enc = auc_pz_jacobian_enc,
                    auc_pz_jacobian_dec = auc_pz_jacobian_dec, auc_pxv_pz = auc_pxv_pz, auc_pxv_pz_jacobian_enc = auc_pxv_pz_jacobian_enc, auc_pxv_pz_jacobian_dec = auc_pxv_pz_jacobian_dec,
                    ztrain_mmd_pval = ztrain_mmd_pval, ztest_mmd_pval = ztest_mmd_pval)
end

datasets = ["breast-cancer-wisconsin"]
difficulties = ["easy"]
batchSize = 100
iterations = 10000

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

for i in 1:10
    for (dn, df) in zip(datasets, difficulties)
        train, test, clusterdness = loaddata(dn, df)

        evaluateOneConfig = p -> runExperiment(dn, train, test, size(train[1], 1), p..., batchSize, iterations, i)
        results = gridsearch(evaluateOneConfig, [64], [8], [3], ["swish"], ["Dense"], [1.])

        CSV.write(outputFolder * "twostagesvae-$dn-$i.csv", vcat(results...))
    end
end
