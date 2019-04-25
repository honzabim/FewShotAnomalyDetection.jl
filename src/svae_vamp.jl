mutable struct SVAEvamp <: SVAE
	q
	g
	zdim
	hue
	μzfromhidden
	κzfromhidden
	pseudo_inputs

	"""
	SVAE_vamp(q, g, hdim, zdim, T) Constructor of the S-VAE where `zdim > 3` and T determines the floating point type (default Float32)
	"""
	SVAEvamp(q, g, hdim::Integer, zdim::Integer, num_pseudoinputs::Integer, T = Float32) = new(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Chain(Dense(hdim, zdim), x -> normalizecolumns(x))), Adapt.adapt(T, Dense(hdim, 1, softplus)), Flux.param(Adapt.adapt(T, normalizecolumns(randn(size(q[1].W, 2), num_pseudoinputs)))))
end

function SVAEvamp(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, num_pseudoinputs, T = Float32)
	encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))
	return SVAEvamp(encoder, decoder, hiddenDim, latentDim, num_pseudoinputs, T)
end

Flux.@treelike(SVAEvamp)

function sampleVamp(m::SVAEvamp, n::Integer)
	components = rand(1:size(m.pseudo_inputs, 2), n)
	(μz, κz) = zparams(m, m.pseudo_inputs[:, components])
	return samplez(m, μz, κz)
end

function wloss(m::SVAEvamp, x, β, d)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	zp = sampleVamp(m, size(z, 2))
	#prior = samplez(m, ones(size(μz)) .* normalizecolumns(m.priorμ), ones(size(κz)) .* m.priorκ)
	Ω = d(z, zp)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * Ω
end
