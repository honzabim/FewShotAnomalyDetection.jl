"""
		Implementation of Hyperspherical Variational Auto-Encoders

		Original paper: https://arxiv.org/abs/1804.00891

		SVAEbase(q,g,zdim,hue,μzfromhidden,κzfromhidden)

		q --- encoder - in this case encoder only encodes from input to a hidden
						layer which is then transformed into parameters for the latent
						layer by `μzfromhidden` and `κzfromhidden` functions
		g --- decoder
		zdim --- dimension of the latent space
		hue --- Hyperspherical Uniform Entropy that is part of the KL divergence but depends only on dimensionality so can be computed in constructor
		μzfromhidden --- function that transforms the hidden layer to μ parameter of the latent layer by normalization
		κzfromhidden --- transforms hidden layer to κ parameter of the latent layer using softplus since κ is a positive scalar
"""
struct SVAEbase <: SVAE
	q
	g
	zdim
	hue
	μzfromhidden
	κzfromhidden

	"""
	SVAEbase(q, g, hdim, zdim, T) Constructor of the S-VAE where `zdim > 3` and T determines the floating point type (default Float32)
	"""
end

SVAEbase(q, g, hdim::Int, zdim::Int, T = Float32) = SVAEbase(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Chain(Dense(hdim, zdim), x -> normalizecolumns(x))), Adapt.adapt(T, Dense(hdim, 1, x -> σ.(x) .* 100)))
function SVAEbase(inputDim::Int, hiddenDim::Int, latentDim::Int, numLayers::Int, nonlinearity::String, layerType::String, T = Float32)
	encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers, nonlinearity, "", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))
	return SVAEbase(encoder, decoder, hiddenDim, latentDim, T)
end

Flux.@treelike(SVAEbase)

"""
	loss(m::SVAEbase, x)

	Loss function of the S-VAE combining reconstruction error and the KL divergence
"""
function loss(m::SVAEbase, x, β)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * mean(kldiv(m, κz))
end

function wloss(m::SVAEbase, x, β, d)
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	zp = samplehsuniform(size(z))
	Ω = d(z, zp)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * Ω
end

function mem_wloss(svae::SVAEbase, mem::KNNmemory, x, y, β, d, α)
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
