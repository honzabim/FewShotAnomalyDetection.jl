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

include("KNNmemory.jl")
include("bessel.jl")
include("svae.jl")
include("svaebase.jl")
include("svaetwocaps.jl")
include("svae_vamp.jl")
include("svae_vamp_means.jl")

if in("CuArrays",keys(Pkg.installed()))
    using CuArrays
    include("svae_gpu.jl")
end

export SVAEbase, SVAEtwocaps, SVAEvamp, SVAEvampmeans, loss, wloss, log_pxexpectedz, pz, log_pz, log_px, log_pz_jacobian_encoder, log_pz_jacobian_decoder, zparams, printing_wloss, mem_wloss

end
