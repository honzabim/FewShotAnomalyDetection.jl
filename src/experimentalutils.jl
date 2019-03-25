"""
    gridSearch(f, parameters...)
Maps `f` to product of `parameters`.
"""
gridsearch(f, parameters...) = map(p -> f(p), Base.product(parameters...))

# function printandrun(f, p)
#     println(p)
#     (p, f(p))
# end

global mainfolder = "D:/dev/julia/"
if !isdir(mainfolder)
    mainfolder = "/home/bimjan/dev/julia/"
    if !isdir(mainfolder)
        error("The main folder is unknown")
    end
end
const datafolder = mainfolder * "data/loda/public/datasets/numerical"

loaddata(dataset, difficulty) =  ADatasets.makeset(ADatasets.loaddataset(dataset, difficulty, datafolder)..., 0.8, "low")

computeauc(score, labels) = EvalCurves.auc(EvalCurves.roccurve(score, labels)...)
