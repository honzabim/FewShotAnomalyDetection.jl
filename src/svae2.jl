include("bessel.jl")

using Flux
using NNlib
using Distributions
using SpecialFunctions
using Adapt
using Random
using LinearAlgebra

"""
		Implementation of Hyperspherical Variational Auto-Encoders

		Original paper: https://arxiv.org/abs/1804.00891

		SVAE2(q,g,zdim,hue,μzfromhidden,κzfromhidden)

		q --- encoder - in this case encoder only encodes from input to a hidden
						layer which is then transformed into parameters for the latent
						layer by `μzfromhidden` and `κzfromhidden` functions
		g --- decoder
		zdim --- dimension of the latent space
		hue --- Hyperspherical Uniform Entropy that is part of the KL divergence but depends only on dimensionality so can be computed in constructor
		μzfromhidden --- function that transforms the hidden layer to μ parameter of the latent layer by normalization
		κzfromhidden --- transforms hidden layer to κ parameter of the latent layer using softplus since κ is a positive scalar
"""
mutable struct SVAE2
	q
	g
	zdim
	hue
	μzfromhidden
	κzfromhidden
	priorμ
	priorκ

	"""
	SVAE2(q, g, hdim, zdim, T) Constructor of the S-VAE where `zdim > 3` and T determines the floating point type (default Float32)
	"""
	# SVAE2(q, g, hdim::Integer, zdim::Integer, T = Float32) = new(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Chain(Dense(hdim, zdim), x -> normalizecolumns(x))), Adapt.adapt(T, Dense(hdim, 1, softplus)), Flux.param(Adapt.adapt(T, vcat(1., zeros(zdim - 1)...))), Flux.param(Adapt.adapt(T, [1.])))
	SVAE2(q, g, hdim::Integer, zdim::Integer, T = Float32) = new(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Chain(Dense(hdim, zdim), x -> normalizecolumns(x))), Adapt.adapt(T, Dense(hdim, 1, softplus)), Flux.param(Adapt.adapt(T, normalize(randn(zdim)))), Flux.param(Adapt.adapt(T, [1.])))
	# SVAE(q, g, hdim::Integer, zdim::Integer, T = Float32) = new(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Chain(Dense(hdim, zdim), x -> normalizecolumns(x))), Adapt.adapt(T, Dense(hdim, 1, softplus)), Adapt.adapt(T, vcat(1., zeros(zdim - 1)...)), Flux.param(Adapt.adapt(T, [1.])))
	# SVAE(q, g, hdim::Integer, zdim::Integer, T = Float32) = new(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Chain(Dense(hdim, zdim), x -> normalizecolumns(x))), Adapt.adapt(T, Dense(hdim, 1, softplus)), Adapt.adapt(T, vcat(1., zeros(zdim - 1)...)), Adapt.adapt(T, [1.]))
	# SVAE(q, g, hdim::Integer, zdim::Integer, T = Float32) = new(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Chain(Dense(hdim, zdim), x -> normalizecolumns(x))), Adapt.adapt(T, Dense(hdim, 1, softplus)), Flux.param(Adapt.adapt(T, vcat(1., zeros(zdim - 1)...))), Adapt.adapt(T, [1.]))
end

Flux.@treelike(SVAE2)

normalizecolumns(m) = m ./ sqrt.(sum(m .^ 2, dims = 1) .+ eps(eltype(Flux.Tracker.data(m))))

"""
	vmfentropy(m, κ)

	Entropy of Von Mises-Fisher distribution
"""
vmfentropy(m, κ) = .-κ .* besselix(m / 2, κ) ./ besselix(m / 2 - 1, κ) .- ((m ./ 2 .- 1) .* log.(κ) .- (m ./ 2) .* log(2π) .- (κ .+ log.(besselix(m / 2 - 1, κ))))

"""
	huentropy(m)

	Entropy of Hyperspherical Uniform distribution
"""
huentropy(m) = m / 2 * log(π) + log(2) - lgamma(m / 2)

"""
	kldiv(model::SVAE2, κ)

	KL divergence between Von Mises-Fisher and Hyperspherical Uniform distributions
"""
kldiv(model::SVAE2, κ) = .- vmfentropy(model.zdim, κ) .+ model.hue

"""
	loss(m::SVAE2, x)

	Loss function of the S-VAE combining reconstruction error and the KL divergence
"""
function loss(m::SVAE2, x, β)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * mean(kldiv(m, κz))
end

function pairwisecos(x, y)
	m = x' * y .* (1 - eps(Float32) * size(x, 1) * 5)
	acos.(m)
end
pairwisecos(x) = pairwisecos(x, x)

function samplehsuniform(size...)
	v = randn(size...)
	v = normalizecolumns(v)
end

k_imq(x,y,c) = sum( c./ (c .+ pairwisecos(x,y))) / (size(x,2) * size(y,2))
k_imq(x::T,c) where {T<:AbstractMatrix} = sum(c ./ (c .+ pairwisecos(x)))/(size(x,2) * (size(x,2) -1 ))
k_imq(x::T,c) where {T<:AbstractVector} = zero(eltype(x))

mmd_imq(x,y,c) = k_imq(x,c) + k_imq(y,c) - 2 * k_imq(x,y,c)

log_normal(x) = - sum((@. x ^ 2), 1) / 2 - size(x, 1) * log(2π) / 2
log_normal(x, μ) = log_normal(x - μ)

# Likelihood estimation of a sample x under VMF with given parameters taken from https://pdfs.semanticscholar.org/2b5b/724fb175f592c1ff919cc61499adb26996b1.pdf
# normalizing constant for density function of VMF
c(p, κ) = κ ^ (p / 2 - 1) / ((2π) ^ (p / 2) * besseli(p / 2 - 1, κ))

# log likelihood of one sample under the VMF dist with given parameters
log_vmf(x, μ, κ) = κ * μ' * x .+ log(c(length(μ), κ))

# Effective sample size
neff(weights) = sum(weights) ^ 2 / (sum(weights .^ 2))

function px(m::SVAE2, x::Matrix, k::Int = 100)
	x = [x[:, i] for i in 1:size(x, 2)]
	return map(a -> px(m, a, k), x)
end

function px(m::SVAE2, x::Vector, k::Int = 100)
	μz, κz = zparams(m, x)
	μz = repmat(Flux.Tracker.data(μz), 1, k)
	κz = repmat(Flux.Tracker.data(κz), 1, k)
	z = Flux.Tracker.data(samplez(m, μz, κz))
	xgivenz = Flux.Tracker.data(m.g(z))

	pxgivenz = log_normal(xgivenz, repmat(x, 1, k))
	pz = log_vmf(z, m.priorμ, m.priorκ[1])
	qzgivenx = log_vmf(z, μz[:, 1], κz[1])

	ess = neff(exp.(pz .- qzgivenx))
	println("Effective sample size: $ess")

	return log(sum(exp.(Flux.Tracker.data(pxgivenz .+ pz .- qzgivenx))))
end

function px2(m::SVAE2, x::Matrix, k::Int = 100)
	x = [x[:, i] for i in 1:size(x, 2)]
	return map(a -> px2(m, a, k), x)
end

function px2(m::SVAE2, x::Vector, k::Int = 100)
	μz, κz = zparams(m, x)
	μz = repmat(Flux.Tracker.data(μz), 1, k)
	κz = repmat(Flux.Tracker.data(κz), 1, k)
	z = samplez(m, μz, κz)
	xgivenz = Flux.Tracker.data(m.g(z))

	pxgivenz = log_normal(xgivenz, repmat(x, 1, k))
	pz = log_vmf(z, m.priorμ, m.priorκ[1])
	qzgivenx = log_vmf(z, μz[:, 1], κz[1])

	return sum(exp.(Flux.Tracker.data(qzgivenx)) .* (Flux.Tracker.data(pxgivenz .+ pz .- qzgivenx)))
end

function px3(m::SVAE2, x::Matrix, k::Int = 100)
	x = [x[:, i] for i in 1:size(x, 2)]
	return map(a -> px3(m, a, k), x)
end

function px3(m::SVAE2, x::Vector, k::Int = 100)
	μz, κz = zparams(m, x)
	μz = repmat(Flux.Tracker.data(μz), 1, k)
	κz = repmat(Flux.Tracker.data(κz), 1, k)
	z = samplez(m, μz, κz)
	xgivenz = Flux.Tracker.data(m.g(z))

	pxgivenz = log_normal(xgivenz, repmat(x, 1, k))
	pz = log_vmf(z, m.priorμ, m.priorκ[1])
	qzgivenx = log_vmf(z, μz[:, 1], κz[1])

	return sum(Flux.Tracker.data(pxgivenz .+ pz .- qzgivenx))
end

function px4(m::SVAE2, x::Matrix, k::Int = 100)
	x = [x[:, i] for i in 1:size(x, 2)]
	return map(a -> px4(m, a, k), x)
end

function px4(m::SVAE2, x::Vector, k::Int = 100)
	μz, κz = zparams(m, x)
	μz = repmat(Flux.Tracker.data(μz), 1, k)
	κz = repmat(Flux.Tracker.data(κz), 1, k)
	z = samplez(m, μz, κz)
	xgivenz = Flux.Tracker.data(m.g(z))

	pxgivenz = log_normal(xgivenz, repmat(x, 1, k))
	pz = log_vmf(z, m.priorμ, m.priorκ[1])
	qzgivenx = log_vmf(z, μz[:, 1], κz[1])

	return sum(Flux.Tracker.data(pxgivenz .+ pz))
end

function px5(m::SVAE2, x::Matrix, k::Int = 100)
	x = [x[:, i] for i in 1:size(x, 2)]
	return map(a -> px5(m, a, k), x)
end

function px5(m::SVAE2, x::Vector, k::Int = 100)
	μz, κz = zparams(m, x)
	μz = repmat(Flux.Tracker.data(μz), 1, k)
	κz = repmat(Flux.Tracker.data(κz), 1, k)
	z = samplez(m, μz, κz)
	xgivenz = Flux.Tracker.data(m.g(z))

	pxgivenz = log_normal(xgivenz, repmat(x, 1, k))
	pz = log_vmf(z, m.priorμ, m.priorκ[1])
	qzgivenx = log_vmf(z, μz[:, 1], κz[1])

	return sum(Flux.Tracker.data(pxgivenz .+ pz .- qzgivenx))
end

function pxvita(m::SVAE2, x)
	μz, κz = zparams(m, x)
	xgivenz = m.g(μz)
	Flux.Tracker.data(log_normal(xgivenz, x))
end

function wloss(m::SVAE2, x, β, d)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	# zp = samplehsuniform(size(z))
	prior = samplez(m, ones(size(μz)) .* normalizecolumns(m.priorμ), ones(size(κz)) .* m.priorκ)
	Ω = d(z, prior)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * Ω
end

"""
	infer(m::SVAE2, x)

	infer latent variables and sample output x
"""
function infer(m::SVAE2, x)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	xgivenz = m.g(z)
	return xgivenz
end

"""
	zparams(model::SVAE2, x)

	Computes μ and κ from the hidden layer
"""
function zparams(model::SVAE2, x)
	hidden = model.q(x)
	return model.μzfromhidden(hidden), model.κzfromhidden(hidden)
end

"""
	zfromx(m::SVAE2, x)

	convenience function that returns latent layer based on the input `x`
"""
zfromx(m::SVAE2, x) = samplez(m, zparams(m, x)...)

"""
	samplez(m::SVAE2, μz, κz)

	samples z layer based on its parameters
"""
function samplez(m::SVAE2, μz, κz)
	ω = sampleω(m, κz)
	normal = Normal()
	v = Adapt.adapt(eltype(Flux.Tracker.data(κz)), rand(normal, size(μz, 1) - 1, size(μz, 2)))
	v = normalizecolumns(v)
	z = householderrotation(vcat(ω, sqrt.(1 .- ω .^ 2) .* v), μz)
	return z
end

"""
	householderrotation(zprime, μ)

	uses Householder reflection to rotate the `zprime` vector according to mapping of e1 to μ
"""
function householderrotation(zprime, μ)
	e1 = similar(μ) .= 0
	e1[1, :] .= 1
	u = e1 .- μ
	normalizedu = normalizecolumns(u)
	return zprime .- 2 .* sum(zprime .* normalizedu, dims = 1) .* normalizedu
end

function sampleω(model::SVAE2, κ)
	m = model.zdim
	c = @. sqrt(4κ ^ 2 + (m - 1) ^ 2)
	b = @. (-2κ + c) / (m - 1)
	a = @. (m - 1 + 2κ + c) / 4
	d = @. (4 * a * b) / (1 + b) - (m - 1) * log(m - 1)
	ω = rejectionsampling(m, a, b, d)
	return ω
end

function rejectionsampling(m, a, b, d)
	beta = Beta((m - 1) / 2, (m - 1) / 2)
	T = eltype(Flux.Tracker.data(a))
	ϵ, u = Adapt.adapt(T, rand(beta, size(a)...)), Adapt.adapt(T, rand(T, size(a)))

	accepted = isaccepted(ϵ, u, m, Flux.data(a), Flux.Tracker.data(b), Flux.data(d))
	while !all(accepted)
		mask = .! accepted
		ϵ[mask] = Adapt.adapt(T, rand(beta, sum(mask)))
		u[mask] = Adapt.adapt(T, rand(T, sum(mask)))
		ia = isaccepted(mask, ϵ, u, m, Flux.data(a), Flux.data(b), Flux.data(d))
		accepted[mask] = ia
	end
	return @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
end

isaccepted(mask, ϵ, u, m:: Int, a, b, d) = isaccepted(ϵ[mask], u[mask], m, a[mask], b[mask], d[mask]);
function isaccepted(ϵ, u, m:: Int, a, b, d)
	ω = @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
	t = @. 2 * a * b / (1 - (1 - b) * ϵ)
	@. (m - 1) * log(t) - t + d >= log(u)
end
