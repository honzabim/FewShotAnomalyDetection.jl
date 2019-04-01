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

include("bessel.jl")
include("svae.jl")
include("svaebase.jl")
include("svaetwocaps.jl")

export SVAEbase, SVAEtwocaps, loss, wloss, pxexpectedz, pz, px, zparams, printing_wloss

end
