module FewShotAnomalyDetection

using Flux
using NNlib
using Distributions
using SpecialFunctions
using Adapt
using Random
using LinearAlgebra
using FluxExtensions
using EvalCurves
using Pkg
using Statistics

include("utils.jl")
include("KNNmemory.jl")
include("bessel.jl")
include("svae.jl")
include("svaebase.jl")
include("svaetwocaps.jl")
include("svae_vamp.jl")
include("svae_vamp_means.jl")
include("vae.jl")
include("vae_vamp_wass.jl")
include("vae_ard.jl")
include("jacodeco.jl")

if in("CuArrays",keys(Pkg.installed())) && in("CUDAnative",keys(Pkg.installed()))
    using CuArrays
    using CUDAnative
    include("svae_gpu.jl")
end

export SVAE, SVAEbase, SVAEtwocaps, SVAEvamp, SVAEvampmeans, VAE, VAE_ard, x_from_z, elbo_loss, elbo_loss_ard, decomposed_elbo_loss, wloss, decomposed_wloss, rloss, log_pxexpectedz, pz, log_pz, log_px, log_pz_jacobian_encoder, log_pz_jacobian_decoder, samplez, sampleVamp, zparams, zfromx, printing_wloss, mem_wloss, log_det_jacobian_encoder, log_det_jacobian_decoder, hsplit1softp, VAEvamp, add_labeled_centroid!, correctedjacodeco, manifoldz, jacodeco

end
