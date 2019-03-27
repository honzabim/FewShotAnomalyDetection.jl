"""
		Implementation of Hyperspherical Variational Auto-Encoders

		Original paper: https://arxiv.org/abs/1804.00891

		SVAEtwocaps(q,g,zdim,hue,μzfromhidden,κzfromhidden)

		q --- encoder - in this case encoder only encodes from input to a hidden
						layer which is then transformed into parameters for the latent
						layer by `μzfromhidden` and `κzfromhidden` functions
		g --- decoder
		zdim --- dimension of the latent space
		hue --- Hyperspherical Uniform Entropy that is part of the KL divergence but depends only on dimensionality so can be computed in constructor
		μzfromhidden --- function that transforms the hidden layer to μ parameter of the latent layer by normalization
		κzfromhidden --- transforms hidden layer to κ parameter of the latent layer using softplus since κ is a positive scalar
"""
mutable struct SVAEtwocaps{V<:Val} <: SVAE
	q
	g
	zdim
	hue
	μzfromhidden
	κzfromhidden
	priorμ
	priorκ
	variant::V

	"""
	SVAEtwocaps(q, g, hdim, zdim, T) Constructor of the S-VAE where `zdim > 3` and T determines the floating point type (default Float32)
	"""
end
SVAEtwocaps(q, g, hdim::Int, zdim::Int, μ, v::Symbol = :unit, T = Float32) = SVAEtwocaps(q, g, zdim, convert(T, huentropy(zdim)), Adapt.adapt(T, Chain(Dense(hdim, zdim), x -> normalizecolumns(x))), Adapt.adapt(T, Dense(hdim, 1, softplus)), μ, Flux.param(Adapt.adapt(T, [1.])), Val(v))
SVAEtwocaps(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, v::Symbol = :unit, T::DataType = Float32) = SVAEtwocaps(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, Flux.param(Adapt.adapt(T, normalize(randn(latentDim)))), v, T)
function SVAEtwocaps(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, μ::AbstractVector, v::Symbol = :unit, T = Float32)
	encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, hiddenDim, numLayers - 1, nonlinearity, "", layerType))
	decoder = nothing
	if v == :unit
		decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))
	elseif v == :scalarsigma
		decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim + 1, numLayers + 1, nonlinearity, "linear", layerType))
	end
	return SVAEtwocaps(encoder, decoder, hiddenDim, latentDim, μ, v, T)
end

Flux.@treelike(SVAEtwocaps)

function pz(m::SVAEtwocaps, x)
	μz, _ = zparams(m, x)
	return log_vmf_c(Flux.Tracker.data(μz), Flux.Tracker.data(.-m.priorμ), Flux.Tracker.data(m.priorκ[1]))
end

function set_normal_μ(m::SVAEtwocaps, μ)
	κz = 1.
	T = eltype(m.hue)
	m.priorμ = Flux.param(Adapt.adapt(T, Flux.Tracker.data(μ)))
	m.priorκ = Flux.param(Adapt.adapt(T, [κz]))
end

function set_normal_μ_nonparam(m::SVAEtwocaps, μ)
	κz = 1.
	T = eltype(m.hue)
	m.priorμ = Adapt.adapt(T, Flux.Tracker.data(μ))
	m.priorκ = Adapt.adapt(T, [κz])
end

function set_normal_hypersphere(m::SVAEtwocaps, anomaly)
	μz, _ = zparams(m, anomaly)
	set_normal_μ(m, μz)
end

function wloss(m::SVAEtwocaps{V}, x, β, d) where {V <: Val{:unit}}
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	# zp = samplehsuniform(size(z))
	prior = samplez(m, ones(size(μz)) .* normalizecolumns(m.priorμ), ones(size(κz)) .* m.priorκ)
	Ω = d(z, prior)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * Ω
end

function wloss(m::SVAEtwocaps{V}, x, d) where {V <: Val{:scalarsigma}}
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	# zp = samplehsuniform(size(z))
	prior = samplez(m, ones(size(μz)) .* normalizecolumns(m.priorμ), ones(size(κz)) .* m.priorκ)
	Ω = d(z, prior)
	xgivenz = m.g(z)
	return -mean(log_normal(x, xgivenz[1:end - 1, :], collect(softplus.(xgivenz[end, :])'))) + Ω
	# return Flux.mse(x, xgivenz[1:end - 1, :]) + β * Ω
end

function printing_wloss(m::SVAEtwocaps{V}, x, d) where {V <: Val{:scalarsigma}}
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	# zp = samplehsuniform(size(z))
	prior = samplez(m, ones(size(μz)) .* normalizecolumns(m.priorμ), ones(size(κz)) .* m.priorκ)
	Ω = d(z, prior)
	xgivenz = m.g(z)
	re = -mean(log_normal(x, xgivenz[1:end - 1, :], collect(softplus.(xgivenz[end, :])')))
	println("loglkl: $re | Wass-dist: $Ω")
	return re + Ω
	# return Flux.mse(x, xgivenz[1:end - 1, :]) + β * Ω
end

function pxexpectedz(m::SVAEtwocaps{V}, x) where {V <: Val{:scalarsigma}}
	xgivenz = m.g(zparams(m, x)[1])[1:end - 1, :]
	Flux.Tracker.data(log_normal(xgivenz, x))
end


function wloss_semi_supervised(m::SVAEtwocaps{V}, x, y, β, d, α) where {V <: Val{:unit}}
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	xgivenz = m.g(z)

	if count(y .== 1) > 0
		anom_ids = findall(y .== 1)
		anom_ids = anom_ids[rand(1:length(anom_ids), length(y))]
		μzanom = μz[:, anom_ids]
		κzanom = κz[anom_ids]
		zanom = samplez(m, μzanom, collect(κzanom'))

		norm_ids = findall(y .== 0)
		norm_ids = norm_ids[rand(1:length(norm_ids), length(y))]
		μznorm = μz[:, norm_ids]
		κznorm = κz[norm_ids]
		znorm = samplez(m, μznorm, collect(κznorm'))

		anom_prior = samplez(m, ones(size(μz)) .* normalizecolumns(.-m.priorμ), ones(size(κz)) .* m.priorκ)
		norm_prior = samplez(m, ones(size(μz)) .* normalizecolumns(m.priorμ), ones(size(κz)) .* m.priorκ)
		Ωnorm = d(znorm, norm_prior)
		Ωanom = d(zanom, anom_prior)
		return Flux.mse(x, xgivenz) + β * (α .* Ωnorm .+ (1 - α) .* Ωanom)
	else
		norm_prior = samplez(m, ones(size(μz)) .* normalizecolumns(m.priorμ), ones(size(κz)) .* m.priorκ)
		Ωnorm = d(z, norm_prior)
		return Flux.mse(x, xgivenz) + β * Ωnorm
	end
end
