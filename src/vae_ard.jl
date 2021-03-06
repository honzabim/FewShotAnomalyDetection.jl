function gaussiansample(μ, σ2)
	ϵ = randn!(similar(μ,size(μ)))
	μ .+ sqrt.(σ2) .* ϵ
end

"""
		Implementation of VAE_ard with Gaussian prior and posterior.

		VAE_ard(q,g,β,s)

		q --- encoder
		g --- decoder

		β --- (variance of the posterior distribution) or the strength on KL-divergence ``D_{KL}(p(z)\\|q(z|x))``
		s = :unit  p(x|z) has a distribution ``N(\\mu(z),I)``
		s = :sigmadiag p(x|z) has a distribution ``N(\\mu(z),\\sigma^2(z))``

		Let's assume that latent dimension should be `n`, then encoder should have output dimension `2n`.
		Similarly, if `s = Val{:unit}`, the g should gave output dimension `d`, as it simply codes the mean
		```julia
			m = VAE_ard(layerbuilder(idim,hidden,2*zdim,3,"relu","linear","Dense"),
	      layerbuilder(zdim,hidden,idim,3,"relu","linear","Dense"),1.0,:unit)
		```
		but for inferring variance of the normal distribution `s = Val{sigmadiag}`, output dimension of `g` should be `2d`.

		```julia
			m = VAE_ard(layerbuilder(idim,hidden,2*zdim,3,"relu","linear","Dense"),
	      layerbuilder(zdim,hidden,2*idim,3,"relu","linear","Dense"),1.0,:sigma)
		```
"""
struct VAE_ard{T<:AbstractFloat,V<:Val}
	q  # encoder (inference modul)
	g  # decoder (generator)
    β::AbstractArray{T, 1} 	#penalization = log σ^2
    γ
	variant::V
end

VAE_ard(q,g,β,γ,s::Symbol = :unit) = VAE_ard(q,g,β,γ,Val(s))

function VAE_ard(inputDim, hiddenDim, latentDim, numLayers, nonlinearity, layerType, β, V = :unit, T = Float32)
	encoder = Adapt.adapt(T, FluxExtensions.layerbuilder(inputDim, hiddenDim, latentDim * 2, numLayers + 1, nonlinearity, "linear", layerType))
    decoder = Adapt.adapt(T, FluxExtensions.layerbuilder(latentDim, hiddenDim, inputDim, numLayers + 1, nonlinearity, "linear", layerType))
	return VAE_ard(encoder, decoder, param([T(β)]), param(ones(T, latentDim)), V)
end

Flux.@treelike(VAE_ard)

"""

	infer(m::VAE_ard{T,V},x)

	infer latent variables and sample output x

"""
# function infer(m::VAE_ard{T,V},x) where {T,V<:Val{:sigmadiag}}
# 	μz, σ2z = hsplitsoftp(m.q(x))
# 	z = gaussiansample(μz,σ2z)
# 	μx, σ2x = hsplitsoftp(m.g(z))
# 	μz, σ2z, μx, σ2x
# end

function infer(m::VAE_ard{T,V},x) where {T,V<:Val{:scalarsigma}}
	μz, σ2z = hsplitsoftp(m.q(x))
	z = gaussiansample(μz,σ2z)
	μx, σ2x = hsplit1softp(m.g(z))
	μz, σ2z, μx, σ2x
end

function infer(m::VAE_ard{T,V},x) where {T,V<:Val{:unit}}
	μz, σ2z = hsplitsoftp(m.q(x))
	z = gaussiansample(μz,σ2z)
	μx  = m.g(z)
	μz, σ2z, μx
end

"""

	loss(m::VAE_ard{T,V},x)

	loss of the Variational autoencoder ``\\mathbb{E}_{q(z|x)log(p(x|z) - KL(p(z)|q(z|x))``
	with the KL-divergence calculated analytically, since the p(z) and q(z|x) are both Gaussian distributions
"""
# function loss(m::VAE_ard{T,V},x) where {T,V<:Val{:sigmadiag}}
# 	μz, σ2z, μx, σ2x = infer(m,x)
# 	-mean(log_normal(x,μx,σ2x)) + m.β * mean(kldiv(μz,σ2z))
# end

function elbo_loss(m::VAE_ard{T,V},x) where {T,V<:Val{:scalarsigma}}
	μz, σ2z, μx, σ2x = infer(m,x)
	-mean(log_normal(x,μx,collect(σ2x'))) + mean(kldiv(μz,σ2z))
end

function elbo_loss(m::VAE_ard{T,V},x) where {T,V<:Val{:unit}}
	μz, σ2z, μx = infer(m,x)
	-mean(log_normal(x,μx)) + exp(m.β[1]) * mean(kldiv(μz,σ2z)) + size(x, 1) * size(x, 1) / 4f0 * m.β[1] * exp(m.β[1])
end

function elbo_loss_ard(m::VAE_ard{T,V}, x) where {T,V<:Val{:unit}}
    μz, σ2z = zparams(m, x)
    z = gaussiansample(μz, σ2z)
    μx = m.g(z .* m.γ)
	-mean(log_normal(x, μx)) + exp(m.β[1]) * mean(kldiv(μz,σ2z)) + size(x, 1) * size(x, 1) / 4f0 * m.β[1] * exp(m.β[1]) 
end

function x_from_z(m::VAE_ard{T,V}, z) where {T,V<:Val{:unit}}
	μx = m.g(z .* m.γ)
end

function decomposed_elbo_loss(m::VAE_ard{T,V},x) where {T,V<:Val{:unit}}
	μz, σ2z, μx = infer(m,x)
	-mean(log_normal(x,μx)), mean(kldiv(μz,σ2z))
end

function wass_dist(m::VAE_ard, x, d)
	μz, σ2z = zparams(m, x)
	z = gaussiansample(μz,σ2z)
	prior = randn(size(z))
	Ω = d(z, prior)
	return Ω, z
end

function wloss(m::VAE_ard{T,V}, x, d) where {T,V<:Val{:scalarsigma}}
	Ω, z = wass_dist(m, x, d)
	μx, σ2x = hsplit1softp(m.g(z))
	-mean(log_normal(x, μx, collect(σ2x'))) + Ω
end

function wloss(m::VAE_ard{T,V}, x, d) where {T,V<:Val{:unit}}
	Ω, z = wass_dist(m, x, d)
	μx = m.g(z)
	-mean(log_normal(x, μx)) + m.β * Ω
end

function rloss(m::VAE_ard{T,V}, x) where {T,V<:Val{:scalarsigma}}
	μz, σ2z = zparams(m, x)
	z = gaussiansample(μz,σ2z)
	μx, σ2x = hsplit1softp(m.g(z))
	-mean(log_normal(x, μx, collect(σ2x')))
end

function rloss(m::VAE_ard{T,V}, x) where {T,V<:Val{:unit}}
	μz, σ2z = zparams(m, x)
	z = gaussiansample(μz,σ2z)
	μx = m.g(z)
	-mean(log_normal(x, μx))
end

function samplez(m::VAE_ard{T,V}, x) where {T,V}
	z = gaussiansample(zparams(m, x)...)
end

function zparams(m::VAE_ard{T,V}, x) where {T,V}
	μz, σ2z = hsplitsoftp(m.q(x))
	return μz, σ2z
end

function printing_loss(m::VAE_ard{T,V},x) where {T,V<:Val{:scalarsigma}}
	μz, σ2z, μx, σ2x = infer(m,x)
	println("loglkl: $(-mean(log_normal(x,μx,collect(σ2x')))) | KL: $(mean(kldiv(μz,σ2z)))")
	-mean(log_normal(x,μx,collect(σ2x'))) + mean(kldiv(μz,σ2z))
end

function printing_loss(m::VAE_ard{T,V},x) where {T,V<:Val{:unit}}
	μz, σ2z, μx = infer(m,x)
	println("MSE:$(-mean(log_normal(x,μx))) Flux.mse: $(Flux.mse(x, μx)) KL: $(mean(kldiv(μz,σ2z)))")
	-mean(log_normal(x,μx)) + exp(m.β[1]) * mean(kldiv(μz,σ2z)) + size(x, 1) * size(x, 1) / 4f0 * m.β[1] * exp(m.β[1])
end

function printing_wloss(m::VAE_ard{T,V}, x, d) where {T,V<:Val{:scalarsigma}}
	Ω, z = wass_dist(m, x, d)
	μx, σ2x = hsplit1softp(m.g(z))
	lklh = -mean(log_normal(x, μx, collect(σ2x')))
	println("loglklh: $lklh | wass dist: $Ω")
	lklh + Ω
end

function printing_wloss(m::VAE_ard{T,V}, x, d) where {T,V<:Val{:unit}}
	Ω, z = wass_dist(m, x, d)
	lklh = -mean(log_normal(x, m.g(z)))
	println("loglklh: $lklh | wass dist: $Ω")
	lklh + m.β * Ω
end

log_pz(m::VAE_ard, x) = log_normal(hsplitsoftp(m.q(x))[1])

function log_pxexpectedz(m::VAE_ard{T,V},x, σ::AbstractFloat = 1.0) where {T, V<:Val{:unit}}
	μz, σ2z = hsplitsoftp(m.q(x))
	log_normal(x,m.g(μz .* m.γ),σ)
end

function log_pxexpectedz(m::VAE_ard{T,V},x, z::AbstractArray) where {T, V<:Val{:unit}}
	log_normal(x, m.g(z .* m.γ))
end

function log_pxexpectedz(m::VAE_ard{T,V}, x) where {T, V<:Val{:scalarsigma}}
	μz, σ2z = hsplitsoftp(m.q(x))
	μx, σ2x = hsplit1softp(m.g(μz .* m.γ))
	log_normal(x,μx,collect(σ2x'))
end

function log_pxexpectedz(m::VAE_ard{T,V}, x, z) where {T, V<:Val{:scalarsigma}}
	μx, σ2x = hsplit1softp(m.g(z .* m.γ))
	log_normal(x,μx,collect(σ2x'))
end

jacobian_encoder(m::VAE_ard, x) = Flux.Tracker.jacobian(a -> vec(hsplitsoftp(m.q(a))[1]), x)
jacobian_decoder(m::VAE_ard{T,V}, z) where {T, V<:Val{:scalarsigma}} = Flux.Tracker.jacobian(a -> vec(hsplit1softp(m.g(a))[1]), z) 
jacobian_decoder(m::VAE_ard{T,V}, z) where {T, V<:Val{:unit}} = Flux.Tracker.jacobian(a -> m.g(a), z) 

function log_det_jacobian_encoder(m::VAE_ard, x)
	if size(x, 2) > 1
		xs = [x[:, i] for i in 1:size(x, 2)]
		return map(x -> log_det_jacobian_encoder_singleinstance(m, x), xs)
	else
		return log_det_jacobian_encoder_singleinstance(m, x)
	end
end

function log_det_jacobian_encoder_singleinstance(m::VAE_ard, x)
	@assert size(x, 2) == 1
	s = svd(jacobian_encoder(m, x).data)
	d = reduce(+, log.(abs.(s.S))) * 2
end

function log_det_jacobian_decoder(m::VAE_ard, z)
	z = (z .* m.γ).data
	if size(z, 2) > 1
		zs = [z[:, i] for i in 1:size(z, 2)]
		return map(z -> log_det_jacobian_decoder_singleinstance(m, z), zs)
	else
		return log_det_jacobian_decoder_singleinstance(m, z)
	end
end

function log_det_jacobian_decoder_singleinstance(m::VAE_ard, z)
	@assert size(z, 2) == 1
	s = svd(jacobian_decoder(m, z).data)
	d = reduce(+, log.(abs.(s.S))) * 2
end

jacodeco(m::VAE_ard, x::Vector, z::Vector) = sum(log_normal((z .* m.γ).data)) + sum(log_pxexpectedz(m, x, z)) - det(m.g, (z .* m.γ).data)

function correctedjacodeco(model::VAE_ard, x::Vector, z::Vector)
	z = (z .* model.γ).data
	Σ = Flux.data(Flux.Tracker.jacobian(model.g, z))
	S = svd(Σ)
	J = inv(S)
	logd = 2*sum(log.(S.S .+ 1f-6))
	if (det(I + J * transpose(J)) < 0)
		# println("determinant = $(det(I + J * transpose(J))) was negative, returning jacodeco = -Inf")
		# println(J)
		# println(J * transpose(J))
		return -Inf32
	end
	logpz = log_normal(z, zeros(Float32, size(z)...), I + J * J')[1] # TODO I changed the transpose(J) to J' - might be an issue
	logpx = FluxExtensions.log_normal(x, x_from_z(model, z).data, 1, length(x) - length(z))[1]
	logpz + logpx - logd
end