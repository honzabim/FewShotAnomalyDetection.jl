module FewShotAnomalyDetection

using Flux
using NNlib
using Distributions
using SpecialFunctions
using Adapt
using Random
using LinearAlgebra
using ADatasets
using FluxExtensions
using EvalCurves

include("bessel.jl")
include("svae.jl")
include("svaebase.jl")
include("svaetwocaps.jl")
include("experimentalutils.jl")

export SVAEbase, SVAEtwocaps, loss, wloss, loaddata, gridsearch, printandrun, computeauc, pxexpectedz, pz, zparams, printing_wloss

end
