using FewShotAnomalyDetection
using Flux
using MLDataPattern
using FluxExtensions
using Adapt
using DataFrames
using CSV
using Serialization
using IPMeasures
using Random
using Statistics

include("experimentalutils.jl")

outputFolder = mainfolder * "experiments/twostage_svae_corrjac/"
mkpath(outputFolder)

function null_distr_distances(dim, size, k = 500)
    z1 = zeros(Float32, dim, k)
    z2 = copy(z1)
    randn!(z1)
    randn!(z2)
    γs = -10:0.05:2
    cs = [IPMeasures.crit_mmd2_var(IPMeasures.IMQKernel(10.0 ^ γ), z1, z2, IPMeasures.pairwisel2) for γ in γs]
    γ = 10 ^ γs[argmax(cs)]
    null_dst = zeros(Float32, k)
    for i in 1:k
        randn!(z1)
        randn!(z2)
        null_dst[i] = IPMeasures.mmd(IPMeasures.IMQKernel(γ), z1, z2)
    end
    sort!(null_dst)
    return null_dst, γ
end

function get_gamma(m::SVAE, x)
    z = Flux.Tracker.data(zfromx(m, x))
    zp = randn(size(z))
    γs = -10:0.05:2
    cs = [IPMeasures.crit_mmd2_var(IPMeasures.IMQKernel(10.0 ^ γ), z, zp, 1000, IPMeasures.pairwisel2) for γ in γs]
    γ = 10 ^ γs[argmax(cs)]
end

function get_gamma(m::VAE, x)
    z = Flux.Tracker.data(samplez(m, x))
    zp = randn(size(z))
    γs = -10:0.05:2
    cs = [IPMeasures.crit_mmd2_var(IPMeasures.IMQKernel(10.0 ^ γ), z, zp, 1000, IPMeasures.pairwisel2) for γ in γs]
    γ = 10 ^ γs[argmax(cs)]
end

mmdpval(null_dst, x) = searchsortedfirst(null_dst, x) / length(null_dst)

function runExperiment(datasetName, train, test, inputDim, hiddenDim, hiddenDim2, latentDim, numLayers, nonlinearity, layerType, batchSize = 100, numBatches = 10000, γ_step = 1000, i = 0)

    T = Float32
    default_γ = 0.001

    println("$datasetName: computing null distribution distances...")
    null_dst, null_γ = null_distr_distances(latentDim, size(train[1], 2))

    println("$datasetName: creating networks...")
    outer_encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, (latentDim - 1) * 2, numLayers + 1, nonlinearity, "linear", layerType))
    outer_decoder = Adapt.adapt(T, FluxExtensions.layerbuilder((latentDim - 1), hiddenDim, inputDim + 1, numLayers + 1, nonlinearity, "linear", layerType))
    outerVAE = VAE(outer_encoder, outer_decoder, T(1), :scalarsigma)

    innerSVAE = SVAEtwocaps(latentDim - 1, hiddenDim, latentDim, numLayers, nonlinearity, layerType, :scalarsigma, T)
    FewShotAnomalyDetection.set_normal_μ_nonparam(innerSVAE, vcat(T(1), zeros(latentDim - 1)))

    println("$datasetName: training outer VAE...")
    opt = Flux.Optimise.ADAM(1e-4)
    # for i in 1:(numBatches ÷ γ_step)
    #     γ = get_gamma(outerVAE, train[1])
    #     # γ = default_γ
    #     dist_fun = (x, y) -> IPMeasures.mmd(IPMeasures.IMQKernel(γ), x, y)
    #     cb = Flux.throttle(() -> println("$datasetName outer VAE: $(printing_wloss(outerVAE, train[1], dist_fun))"), 5)
    #     Flux.train!(x -> wloss(outerVAE, x, dist_fun), Flux.params(outerVAE), RandomBatches((train[1],), batchSize, γ_step), opt, cb = cb)
    # end
    cb = Flux.throttle(() -> println("$datasetName outer VAE: $(rloss(outerVAE, train[1]))"), 5)
    Flux.train!(x -> rloss(outerVAE, x,), Flux.params(outerVAE), RandomBatches((train[1],), batchSize, numBatches), opt, cb = cb)

    z_train = Flux.Tracker.data(zparams(outerVAE, train[1])[1])
    z_test = Flux.Tracker.data(zparams(outerVAE, test[1])[1])

    println("$datasetName: training inner S-VAE...")
    opt = Flux.Optimise.ADAM(1e-4)
    for i in 1:(numBatches ÷ γ_step)
        γ = get_gamma(innerSVAE, z_train)
        # γ = default_γ
        dist_fun = (x, y) -> IPMeasures.mmd(IPMeasures.IMQKernel(γ), x, y)
        cb = Flux.throttle(() -> println("$datasetName inner S-VAE: $(printing_wloss(innerSVAE, z_train, dist_fun))"), 5)
        Flux.train!(x -> wloss(innerSVAE, x, dist_fun), Flux.params(innerSVAE), RandomBatches((z_train,), batchSize, γ_step), opt, cb = cb)
    end

    # x -> z -> u -> zp -> xp
    println("$datasetName: computing performance metrics...")
    u_train = Flux.Tracker.data(zparams(innerSVAE, z_train)[1])
    u_test = Flux.Tracker.data(zparams(innerSVAE, z_test)[1])
    u_train_dst = IPMeasures.mmd(IPMeasures.IMQKernel(null_γ), randn(size(u_train)), u_train)
    u_test_dst = IPMeasures.mmd(IPMeasures.IMQKernel(null_γ), randn(size(u_test)), u_test)
    u_train_mmd_pval = mmdpval(null_dst, u_train_dst)
    u_test_mmd_pval = mmdpval(null_dst, u_test_dst)
    zp_train = Flux.Tracker.data(hsplit1softp(innerSVAE.g(u_train))[1])
    zp_test = Flux.Tracker.data(hsplit1softp(innerSVAE.g(u_test))[1])
    xp_train = Flux.Tracker.data(hsplit1softp(outerVAE.g(zp_train))[1])
    xp_test = Flux.Tracker.data(hsplit1softp(outerVAE.g(zp_test))[1])

    log_pxv_train = vec(collect(log_pxexpectedz(outerVAE, train[1], zp_train)'))
    log_pxv_test = vec(collect(log_pxexpectedz(outerVAE, test[1], zp_test)'))
    log_pzs = vec(collect(log_pz(innerSVAE, z_test)'))
    log_det_jac_enc_out = vec(collect(log_det_jacobian_encoder(outerVAE, test[1])'))
    log_det_jac_dec_out = vec(collect(log_det_jacobian_decoder(outerVAE, zp_test)'))
    log_pz_jac_enc_inn = vec(collect(log_pz_jacobian_encoder(innerSVAE, z_test)'))
    log_pz_jac_dec_inn = vec(collect(log_pz_jacobian_decoder(innerSVAE, u_test)'))

    labels = test[2] .- 1
    auc_pxv = computeauc(.-log_pxv_test, labels)
    auc_pz = computeauc(.-log_pzs, labels)
    auc_pz_jacobian_enc = computeauc(.-(log_det_jac_enc_out .+ log_pz_jac_enc_inn), labels)
    auc_pz_jacobian_dec = computeauc(.-(log_det_jac_dec_out .- log_pz_jac_dec_inn), labels)
    auc_pxv_pz = computeauc(.-(log_pxv_test .+ log_pzs), labels)
    auc_pxv_pz_jacobian_enc = computeauc(.-(log_pxv_test .+ log_det_jac_enc_out .+ log_pz_jac_enc_inn), labels)
    auc_pxv_pz_jacobian_dec = computeauc(.-(log_pxv_test .- log_det_jac_dec_out .- log_pz_jac_dec_inn), labels)

    serialize(outputFolder * "twostage_svae-$datasetName-$i-$hiddenDim-$hiddenDim2-$latentDim-innvae.jls", innerSVAE)
    serialize(outputFolder * "twostage_svae-$datasetName-$i-$hiddenDim-$hiddenDim2-$latentDim-outvae.jls", outerVAE)
    serialize(outputFolder * "twostage_svae-$datasetName-$i-$hiddenDim-$hiddenDim2-$latentDim-train.jls", train)
    serialize(outputFolder * "twostage_svae-$datasetName-$i-$hiddenDim-$hiddenDim2-$latentDim-test.jls", test)

    df = DataFrame(dataset = datasetName, idim = inputDim, hdim = hiddenDim, hdim2 = hiddenDim2, ldim = latentDim, layers = numLayers, i = i, auc_pxv = auc_pxv, auc_pz = auc_pz, auc_pz_jacobian_enc = auc_pz_jacobian_enc,
                    auc_pz_jacobian_dec = auc_pz_jacobian_dec, auc_pxv_pz = auc_pxv_pz, auc_pxv_pz_jacobian_enc = auc_pxv_pz_jacobian_enc, auc_pxv_pz_jacobian_dec = auc_pxv_pz_jacobian_dec,
                    rec_err_train = .-mean(log_pxv_train), rec_err_test = .-mean(log_pxv_test), u_train_mmd_pval = u_train_mmd_pval, u_test_mmd_pval = u_test_mmd_pval, u_train_dst = u_train_dst,
                    u_test_dst = u_test_dst, meanl2_train = mean(IPMeasures.pairwisel2(train[1], xp_train)), meanl2_test = mean(IPMeasures.pairwisel2(test[1], xp_test)),
                    medianl2_train = median(IPMeasures.pairwisel2(train[1], xp_train)), medianl2_test = median(IPMeasures.pairwisel2(test[1], xp_test)))
    
    CSV.write(outputFolder * "twostage_svae-$datasetName-$i-$hiddenDim-$hiddenDim2-$latentDim-results.csv", df)
    return df
end

datasets = ["abalone"]
difficulties = ["easy"]
batchSize = 100
iterations = 10000
γ_step = 1000

if length(ARGS) != 0
    datasets = [ARGS[1]]
    difficulties = ["easy"]
end

# datasets = ["abalone"]
# difficulties = ["easy"]

for i in 1:5
    for (dn, df) in zip(datasets, difficulties)
        train, test, clusterdness = loaddata(dn, df)

        evaluateOneConfig = p -> runExperiment(dn, train, test, size(train[1], 1), p..., batchSize, iterations, γ_step, i)
        results = gridsearch(evaluateOneConfig, [32], [32], [2, 3, 4, 5, 6, 7, 8, 9], [3], ["swish"], ["Dense"])

        # CSV.write(outputFolder * "twostage_vae-$dn-$i.csv", vcat(results...))
    end
end