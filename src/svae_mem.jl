include("bessel.jl")

using Flux
using NNlib
using Distributions
using SpecialFunctions
using Adapt

"""
		Implementation of Hyperspherical Variational Auto-Encoders

		Original paper: https://arxiv.org/abs/1804.00891

		SVAE(q,g,zdim,hue,μzfromhidden,κzfromhidden)

		q --- encoder - in this case encoder only encodes from input to a hidden
						layer which is then transformed into parameters for the latent
						layer by `μzfromhidden` and `κzfromhidden` functions
		g --- decoder
		zdim --- dimension of the latent space
		hue --- Hyperspherical Uniform Entropy that is part of the KL divergence but depends only on dimensionality so can be computed in constructor
		μzfromhidden --- function that transforms the hidden layer to μ parameter of the latent layer by normalization
		κzfromhidden --- transforms hidden layer to κ parameter of the latent layer using softplus since κ is a positive scalar
"""
struct SVAE
	q
	g
	zdim
	hue
	μzfromhidden
	κzfromhidden

	"""
	SVAE(q, g, hdim, zdim, T) Constructor of the S-VAE where `zdim > 3` and T determines the floating point type (default Float32)
	"""
	SVAE(q, g, hdim::Integer, zdim::Integer, T = Float32) = new(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Chain(Dense(hdim, zdim), x -> normalizecolumns(x))), Adapt.adapt(T, Dense(hdim, 1, softplus)))
end

Flux.@treelike(SVAE)

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
	kldiv(model::SVAE, κ)

	KL divergence between Von Mises-Fisher and Hyperspherical Uniform distributions
"""
kldiv(model::SVAE, κ) = .- vmfentropy(model.zdim, κ) .+ model.hue

"""
	loss(m::SVAE, x)

	Loss function of the S-VAE combining reconstruction error and the KL divergence
"""
function loss(m::SVAE, x, β)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * mean(kldiv(m, κz))
end

function pairwisecos(x, y)
	m = x' * y .* (1 - eps(Float32) * size(x, 1))
	acos.(m)
end
pairwisecos(x) = pairwisecos(x, x)

function samplehsuniform(size...)
	v = randn(size...)
	v = normalizecolumns(v)
end

k_imq(x,y,c) = sum( c./ (c .+ pairwisecos(x,y)))/(size(x,2) * size(y,2))
k_imq(x::T,c) where {T<:AbstractMatrix} = sum(c ./(c .+ pairwisecos(x)))/(size(x,2) * (size(x,2) -1 ))
k_imq(x::T,c) where {T<:AbstractVector} = zero(eltype(x))

mmd_imq(x,y,c) = k_imq(x,c) + k_imq(y,c) - 2 * k_imq(x,y,c)

function wloss_prior(m::SVAE, x, β, d)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	priorμ = zeros(size(z, 1))
	priorμ[1] = 1
	zp = samplez(m, repeat(priorμ, 1, size(z, 2)), repeat([1], size(z, 2)))
	Ω = d(z, zp)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * Ω
end

function wloss(m::SVAE, x, β, d)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	zp = samplehsuniform(size(z))
	Ω = d(z, zp)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * Ω
end

function mem_wloss(svae::SVAE, mem::KNNmemory, x, y, β, d, α)
	(μz, κz) = zparams(svae, x)
	z = samplez(svae, μz, κz)
	xgivenz = svae.g(z)

	repetitions = 10
	(priorμ, priorκ) = zparams(svae, repeat(mem.M', 1, repetitions))
	priorlabels = repeat(mem.V, repetitions)
	priorsamples = samplez(svae, priorμ, priorκ)

	loss = 0

	if count(y .== 1) > 0
		anom_ids = findall(y .== 1)
		anom_ids = anom_ids[rand(1:length(anom_ids), length(y) * 2)]
		μzanom = μz[:, anom_ids]
		κzanom = κz[anom_ids]
		zanom = samplez(svae, μzanom, collect(κzanom'))

		norm_ids = findall(y .== 0)
		norm_ids = norm_ids[rand(1:length(norm_ids), length(y) * repetitions)]
		μznorm = μz[:, norm_ids]
		κznorm = κz[norm_ids]
		znorm = samplez(svae, μznorm, collect(κznorm'))

		Ωnorm = d(znorm, priorsamples[:, priorlabels .== 0])
		Ωanom = d(zanom, priorsamples[:, priorlabels .== 1])

		return Flux.mse(x, xgivenz) + β * (α .* Ωnorm .+ (1 - α) .* Ωanom)
	else
		Ωnorm = d(z, priorsamples[:, priorlabels .== 0])
		return Flux.mse(x, xgivenz) + β * Ωnorm
	end

	# if isnan(loss) | (loss == Inf) | (loss == -Inf)
	# 	println("Watch out, loss is $loss")
	# 	println("There were $(count(y .== 1)) anomalies.")
	# end

	return loss
end

"""
	infer(m::SVAE, x)

	infer latent variables and sample output x
"""
function infer(m::SVAE, x)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	xgivenz = m.g(z)
	return xgivenz
end

"""
	zparams(model::SVAE, x)

	Computes μ and κ from the hidden layer
"""
function zparams(model::SVAE, x)
	hidden = model.q(x)
	return model.μzfromhidden(hidden), model.κzfromhidden(hidden)
end

"""
	zfromx(m::SVAE, x)

	convenience function that returns latent layer based on the input `x`
"""
zfromx(m::SVAE, x) = samplez(m, zparams(m, x)...)

"""
	samplez(m::SVAE, μz, κz)

	samples z layer based on its parameters
"""
function samplez(m::SVAE, μz, κz)
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

function sampleω(model::SVAE, κ)
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
		accepted[mask] = isaccepted(mask, ϵ, u, m, Flux.data(a), Flux.data(b), Flux.data(d))
	end
	return @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
end

isaccepted(mask, ϵ, u, m:: Int, a, b, d) = isaccepted(ϵ[mask], u[mask], m, a[mask], b[mask], d[mask]);
function isaccepted(ϵ, u, m:: Int, a, b, d)
	ω = @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
	t = @. 2 * a * b / (1 - (1 - b) * ϵ)
	@. (m - 1) * log(t) - t + d >= log(u)
end
