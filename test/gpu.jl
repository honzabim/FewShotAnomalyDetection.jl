using CuArrays
using Flux

a = CuArray([ 1 2 3
    4 5 6.])

na = FewShotAnomalyDetection.normalizecolumns(a)