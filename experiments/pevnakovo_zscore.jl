using Flux, Printf, Statistics, IPMeasures, LinearAlgebra, MLDataPattern, FluxExtensions
using IPMeasures: mmd, IMQKernel
using FluxExtensions: log_normal
using Plots


data(n) = Float32.(cholesky([1 0.9; 0.9 1]).U * randn(2,n))

model = (f = Chain(Dense(2,10,tanh),Dense(10,1)), g = Chain(Dense(1,10,tanh), Dense(10,1)), μ = Dense(1,2), σ = Chain(Dense(1,10,tanh),Dense(10,2)))
imq(z, c = 0.01f0) = mmd(IMQKernel(0.01f0), randn(Float32,1,size(z,2)), z)

lkl_full(model, x, z) = FluxExtensions.log_normal(x, model.μ(model.g(z)), softplus.(model.σ(model.g(z))))
lkl_fixed(model, x, z) = FluxExtensions.log_normal(x, model.μ(model.g(z)))

function closestz(model, x, steps = 100)
z = param(Flux.data(model.f(x)))
ps = Flux.Tracker.Params([z])
opt = ADAM()
_lkl(model, x, z) = mean(lkl(model, x, z) .- FluxExtensions.log_normal(z))
_info() = println("likelihood = ", _lkl(model, x, z))
li = Flux.data(_lkl(model, x, z))
# Flux.train!((i) -> -_lkl(model, x, z), ps, 1:steps, opt, cb = () -> Flux.throttle(_info(),5))
Flux.train!((i) -> -_lkl(model, x, z), ps, 1:steps, opt, cb = () -> Flux.throttle(_info(),5))
le = Flux.data(_lkl(model, x, z))
println("initial = ",li, " final = ",le)
Flux.data(z)
end

function manifoldz(model, x, steps = 100)
z = param(Flux.data(model.f(x)))
ps = Flux.Tracker.Params([z])
opt = ADAM()
_info() = println("likelihood = ", lkl(model, x, z))
li = Flux.data(mean(lkl(model, x, z)))
# Flux.train!((i) -> -mean(lkl(model, x, z)), ps, 1:steps, opt, cb = () -> Flux.throttle(_info(),5))
Flux.train!((i) -> -mean(lkl(model, x, z)), ps, 1:steps, opt)
le = Flux.data(mean(lkl(model, x, z)))
println("initial = ",li, " final = ",le)
Flux.data(z)
end

lkl = lkl_full

function loss(model, x)
z = model.f(x)
imq(z) - mean(lkl(model, x, z))
end

function loss(model, x, ϵ)
z = model.f(x .+ ϵ .* randn(eltype(x),size(x)))
imq(z) - mean(lkl(model, x, z))
end

function info(model, x)
z = model.f(x)
println("imq = ", imq(z), " lkl = ", mean(lkl(model, x, z)))
end

xv = data(1000)
x = data(10000)

ps = Flux.Tracker.Params()
foreach(m -> push!(ps,params(m)...),model)
opt = ADAM()
Flux.train!(x -> loss(model, getobs(x), 0.05), ps, RandomBatches((x,), 100,1000), opt, cb = () -> Flux.throttle(info(model, xv),5))


#histogram of latent data
frmodel = FluxExtensions.freeze(model)
z = frmodel.f(xv)
t2m(x) = reshape([x[1],x[2]],:,1)
xr = range(minimum(xv[1,:]) - 1 , maximum(xv[1,:])+ 1 , length = 100)
yr = range(minimum(xv[2,:]) - 1 , maximum(xv[2,:])+ 1 , length = 100)
histogram(z[:],bins = round(Int, sqrt(length(z))))

#sample reconstructed data
xx = frmodel.μ(frmodel.g(z)) .+ randn(2,size(z,2)) .* softplus.(frmodel.σ(frmodel.g(z)))
# xx = frmodel.μ(frmodel.g(z))
scatter(xx[1,:],xx[2,:])

#generate new data
z = randn(1,1000)
xx = frmodel.μ(frmodel.g(z)) .+ randn(2,size(z,2)) .* softplus.(frmodel.σ(frmodel.g(z)))
scatter(xx[1,:],xx[2,:])

#likelihood of latent layer
z = frmodel.f(xv)
contour(xr, yr, (x,y) -> log_normal(frmodel.f([x,y]))[1], title = "p(z)");
scatter!(xv[1,:], xv[2,:], zcolor = log_normal(z)[:], title = "p(z)")

#likelihood of output layer
contour(xr, yr, (x...) -> lkl(frmodel, t2m(x), frmodel.f(t2m(x)))[1], title = "p(x|z)");
scatter!(xv[1,:], xv[2,:], zcolor = lkl(frmodel, xv, z)[:], title = "p(x|z)")

#joined likelihood
contour(xr, yr, (x...) -> log_normal(frmodel.f(t2m(x)))[1] + lkl(frmodel, t2m(x),  frmodel.f(t2m(x)))[1], title = "p(x|z) + p(z)");
scatter!(xv[1,:], xv[2,:], zcolor = log_normal(z)[:] + lkl(frmodel, xv, z)[:], title = "p(x|z) + p(z)")

#optimized likelihood
function opt_lkl(model, x)
z = closestz(model, x)
log_normal(z) + lkl(frmodel, x, z)
end
contour(xr, yr, (x...) -> opt_lkl(frmodel, t2m(x))[1], title = "optimized");
scatter!(xv[1,:], xv[2,:], zcolor = opt_lkl(frmodel, xv)[:], title = "optimized")
heatmap(xr, yr, (x...) -> opt_lkl(frmodel, t2m(x))[1], title = "optimized")

#optimized reconstruction error
function opt_lkl(model, x)
z = manifoldz(model, x)
log_normal(z) + lkl(frmodel, x, z)
end
contour(xr, yr, (x...) -> opt_lkl(frmodel, t2m(x))[1], title = "optimized");
scatter!(xv[1,:], xv[2,:], zcolor = opt_lkl(frmodel, xv)[:], title = "optimized")
heatmap(xr, yr, (x...) -> opt_lkl(frmodel, t2m(x))[1], title = "optimized")