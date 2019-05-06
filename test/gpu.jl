using CuArrays
using Flux
using Random

a = CuArray([ 1 2 3
    4 5 6f0])

na = FewShotAnomalyDetection.normalizecolumns(a)

b = param(a)
FewShotAnomalyDetection.normalizecolumns(b)

c = Array(a)
FewShotAnomalyDetection.normalizecolumns(c)

a = CuArray(randn(Float32, 1000, 1000))
b = param(a)
c = Array(a)

@elapsed FewShotAnomalyDetection.normalizecolumns(a)
@elapsed FewShotAnomalyDetection.normalizecolumns(b)
@elapsed FewShotAnomalyDetection.normalizecolumns(c)
