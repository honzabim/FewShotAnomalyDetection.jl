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
struct SVAEtwocaps <: SVAE
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

Flux.@treelike(SVAEtwocaps)

function wloss(m::SVAEtwocaps, x, β, d)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	# zp = samplehsuniform(size(z))
	prior = samplez(m, ones(size(μz)) .* normalizecolumns(m.priorμ), ones(size(κz)) .* m.priorκ)
	Ω = d(z, prior)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * Ω
end