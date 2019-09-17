using FluxExtensions, LinearAlgebra, MLDataPattern

FluxExtensions.log_normal(x, μ, σ2::T, d::Int) where {T<:Real}  = (d > 0) ? - sum((@. ((x - μ)^2) ./ σ2), dims=1) / 2 .- d * (log.(σ2) + log(2π)) / 2 : 0

"""
	LinearAlgebra.det(f, x::Vector)	
	determinant of a jacobian of `f` at point `x`
"""
function LinearAlgebra.det(f, x::Vector)	
	Σ = Flux.data(Flux.Tracker.jacobian(f, x))
	S = svd(Σ)
	2*sum(log.(S.S .+ 1f-6))
end

jacodeco(model, x::Vector, z::Vector) = sum(log_normal(z)) + sum(log_pxexpectedz(model, x, z)) - det(model.g, z)
function jacodeco(m, x, z)
	@assert nobs(x) == nobs(z)
	[jacodeco(m, getobs(x, i), getobs(z, i)) for i in 1:nobs(x)]
end
jacodeco(m, x) = jacodeco(m, x, m.g(x))

function correctedjacodeco(model, x::Vector, z::Vector)
	Σ = Flux.data(Flux.Tracker.jacobian(model.g, z))
	S = svd(Σ)
	J = inv(S)
	logd = 2*sum(log.(S.S .+ 1f-6))
	if (det(I + J * transpose(J)) < 0)
		println("determinant = $(det(I + J * transpose(J))) was negative, returning jacodeco = -Inf")
		println(J)
		println(J * transpose(J))
		return -Inf32
	end
	logpz = log_normal(z, zeros(Float32, size(z)...), I + J * J')[1] # TODO I changed the transpose(J) to J' - might be an issue
	logpx = FluxExtensions.log_normal(x, x_from_z(model, z).data, 1, length(x) - length(z))[1]
	logpz + logpx - logd
end

function correctedjacodeco(m, x, z)
	@assert nobs(x) == nobs(z)
	[correctedjacodeco(m, getobs(x, i), getobs(z, i)) for i in 1:nobs(x)]
end

correctedjacodeco(m, x) = correctedjacodeco(m, x, manifoldz(model, x))

"""
	manifoldz(model, x, steps = 100, rand_init = false)
	find the point in the latent space `z` such that it minimizes the 
	reconstruction error. This means that it finds the point on manifold 
	defined by `model.g(z)` and x
"""
function manifoldz(model, x, steps = 100, z = Flux.data(zparams(model,x)[1]))
	z = param(z)
	ps = Flux.Tracker.Params([z])
	# println("Size z: $(size(z)), size x: $(size(x))")
	li = Flux.data(mean(log_pxexpectedz(model, x, z)))
	Flux.train!((i) -> -mean(log_pxexpectedz(model, x, z)), ps, 1:steps, ADAM())
	le = Flux.data(mean(log_pxexpectedz(model, x, z)))
	@info "initial = $(li) final = $(le)"
	Flux.data(z)
end