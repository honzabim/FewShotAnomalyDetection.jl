using FewShotAnomalyDetection
using Pkg

if in("CuArrays",keys(Pkg.installed()))
    include("test/gpu.jl")
else
    println("CuArrays aren't present")
end
