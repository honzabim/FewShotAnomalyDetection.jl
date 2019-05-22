"""
		hsplitsoftp(x, ϵ = 1f-5)

		Splits x horizontally into two equal parts and use softplus to the lower part.
		ϵ is added to the soft-plus to prevent underflow and improve the numerical stability

"""
hsplitsoftp(x,ϵ = 1f-5) = x[1:size(x, 1) ÷ 2, :], softplus.(x[size(x, 1) ÷ 2 + 1 : 2 * (size(x, 1) ÷ 2), :] .+ ϵ)
hsplit1softp(x,ϵ = 1f-5) = x[1:size(x, 1) - 1, :], softplus.(x[end, :] .+ ϵ)

"""
		kldiv(μ,σ2)

		kl-divergence of a Gaussian min mean `μ` and diagonal variance `σ^2`
		to N(0,I)
"""
kldiv(μ,σ2) = - mean(sum((@.log(σ2) - μ^2 - σ2), dims = 1))

log_normal(x) = - sum((x.^2), dims = 1) ./ 2 .- size(x, 1) * log(2f0 * π) / 2
log_normal(x, μ) = log_normal(x - μ)
log_normal(x,μ, σ2::AbstractArray{T}) where {T<:Number} = - sum((x - μ) .^ 2 ./ σ2 .+ log.(σ2 .* 2π), dims = 1) / 2

function gaussiansample(μ, σ2)
	ϵ = randn!(similar(μ,size(μ)))
	μ .+ sqrt.(σ2) .* ϵ
end

"""
		Implementation of VAE with Gaussian prior and posterior.

		VAE(q,g,β,s)

		q --- encoder
		g --- decoder

		β --- (variance of the posterior distribution) or the strength on KL-divergence ``D_{KL}(p(z)\\|q(z|x))``
		s = :unit  p(x|z) has a distribution ``N(\\mu(z),I)``
		s = :sigmadiag p(x|z) has a distribution ``N(\\mu(z),\\sigma^2(z))``

		Let's assume that latent dimension should be `n`, then encoder should have output dimension `2n`.
		Similarly, if `s = Val{:unit}`, the g should gave output dimension `d`, as it simply codes the mean
		```julia
			m = VAE(layerbuilder(idim,hidden,2*zdim,3,"relu","linear","Dense"),
	      layerbuilder(zdim,hidden,idim,3,"relu","linear","Dense"),1.0,:unit)
		```
		but for inferring variance of the normal distribution `s = Val{sigmadiag}`, output dimension of `g` should be `2d`.

		```julia
			m = VAE(layerbuilder(idim,hidden,2*zdim,3,"relu","linear","Dense"),
	      layerbuilder(zdim,hidden,2*idim,3,"relu","linear","Dense"),1.0,:sigma)
		```
"""
struct VAE{T<:AbstractFloat,V<:Val}
	q  # encoder (inference modul)
	g  # decoder (generator)
	β::T 	#penalization
	variant::V
end

VAE(q,g,β,s::Symbol = :unit) = VAE(q,g,β,Val(s))

Flux.@treelike(VAE)

"""

	infer(m::VAE{T,V},x)

	infer latent variables and sample output x

"""
# function infer(m::VAE{T,V},x) where {T,V<:Val{:sigmadiag}}
# 	μz, σ2z = hsplitsoftp(m.q(x))
# 	z = gaussiansample(μz,σ2z)
# 	μx, σ2x = hsplitsoftp(m.g(z))
# 	μz, σ2z, μx, σ2x
# end

function infer(m::VAE{T,V},x) where {T,V<:Val{:scalarsigma}}
	μz, σ2z = hsplitsoftp(m.q(x))
	z = gaussiansample(μz,σ2z)
	μx, σ2x = hsplit1softp(m.g(z))
	μz, σ2z, μx, σ2x
end

function infer(m::VAE{T,V},x) where {T,V<:Val{:unit}}
	μz, σ2z = hsplitsoftp(m.q(x))
	z = gaussiansample(μz,σ2z)
	μx  = m.g(z)
	μz, σ2z, μx
end

"""

	loss(m::VAE{T,V},x)

	loss of the Variational autoencoder ``\\mathbb{E}_{q(z|x)log(p(x|z) - KL(p(z)|q(z|x))``
	with the KL-divergence calculated analytically, since the p(z) and q(z|x) are both Gaussian distributions
"""
# function loss(m::VAE{T,V},x) where {T,V<:Val{:sigmadiag}}
# 	μz, σ2z, μx, σ2x = infer(m,x)
# 	-mean(log_normal(x,μx,σ2x)) + m.β * mean(kldiv(μz,σ2z))
# end

function loss(m::VAE{T,V},x) where {T,V<:Val{:scalarsigma}}
	μz, σ2z, μx, σ2x = infer(m,x)
	-mean(log_normal(x,μx,collect(σ2x'))) + mean(kldiv(μz,σ2z))
end

function loss(m::VAE{T,V},x) where {T,V<:Val{:unit}}
	μz, σ2z, μx = infer(m,x)
	-mean(log_normal(x,μx)) + m.β * mean(kldiv(μz,σ2z))
end

function wass_dist(m::VAE, x, d)
	μz, σ2z = zparams(m, x)
	z = gaussiansample(μz,σ2z)
	prior = randn(size(z))
	Ω = d(z, prior)
	return Ω, z
end

function wloss(m::VAE{T,V}, x, d) where {T,V<:Val{:scalarsigma}}
	Ω, z = wass_dist(m, x, d)
	μx, σ2x = hsplit1softp(m.g(z))
	-mean(log_normal(x, μx, collect(σ2x'))) + Ω
end

function wloss(m::VAE{T,V}, x, d) where {T,V<:Val{:unit}}
	Ω, z = wass_dist(m, x, d)
	μx = m.g(z)
	-mean(log_normal(x, μx)) + m.β * Ω
end

function rloss(m::VAE{T,V}, x) where {T,V<:Val{:scalarsigma}}
	μz, σ2z = zparams(m, x)
	z = gaussiansample(μz,σ2z)
	μx, σ2x = hsplit1softp(m.g(z))
	-mean(log_normal(x, μx, collect(σ2x')))
end

function rloss(m::VAE{T,V}, x) where {T,V<:Val{:unit}}
	μz, σ2z = zparams(m, x)
	z = gaussiansample(μz,σ2z)
	μx = m.g(z)
	-mean(log_normal(x, μx))
end

function samplez(m::VAE{T,V}, x) where {T,V}
	z = gaussiansample(zparams(m, x)...)
end

function zparams(m::VAE{T,V}, x) where {T,V}
	μz, σ2z = hsplitsoftp(m.q(x))
	return μz, σ2z
end

function printing_loss(m::VAE{T,V},x) where {T,V<:Val{:scalarsigma}}
	μz, σ2z, μx, σ2x = infer(m,x)
	println("loglkl: $(-mean(log_normal(x,μx,collect(σ2x')))) | KL: $(mean(kldiv(μz,σ2z)))")
	-mean(log_normal(x,μx,collect(σ2x'))) + mean(kldiv(μz,σ2z))
end

function printing_loss(m::VAE{T,V},x) where {T,V<:Val{:unit}}
	μz, σ2z, μx = infer(m,x)
	println("MSE:$(-mean(log_normal(x,μx))) Flux.mse: $(Flux.mse(x, μx)) KL: $(mean(kldiv(μz,σ2z)))")
	-mean(log_normal(x,μx)) + m.β*mean(kldiv(μz,σ2z))
end

function printing_wloss(m::VAE{T,V}, x, d) where {T,V<:Val{:scalarsigma}}
	Ω, z = wass_dist(m, x, d)
	μx, σ2x = hsplit1softp(m.g(z))
	lklh = -mean(log_normal(x, μx, collect(σ2x')))
	println("loglklh: $lklh | wass dist: $Ω")
	lklh + Ω
end

function printing_wloss(m::VAE{T,V}, x, d) where {T,V<:Val{:unit}}
	Ω, z = wass_dist(m, x, d)
	lklh = -mean(log_normal(x, m.g(z)))
	println("loglklh: $lklh | wass dist: $Ω")
	lklh + m.β * Ω
end

log_pz(m::VAE, x) = log_normal(hsplitsoftp(m.q(x))[1])

function log_pxexpectedz(m::VAE{T,V},x, σ = 1.0) where {T, V<:Val{:unit}}
	μz, σ2z = hsplitsoftp(m.q(x))
	log_normal(x,m.g(μz),σ)
end

function log_pxexpectedz(m::VAE{T,V}, x) where {T, V<:Val{:scalarsigma}}
	μz, σ2z = hsplitsoftp(m.q(x))
	μx, σ2x = hsplit1softp(m.g(μz))
	log_normal(x,μx,collect(σ2x'))
end

function log_pxexpectedz(m::VAE{T,V}, x, z) where {T, V<:Val{:scalarsigma}}
	μx, σ2x = hsplit1softp(m.g(z))
	log_normal(x,μx,collect(σ2x'))
end

jacobian_encoder(m::VAE, x) = Flux.Tracker.jacobian(a -> vec(hsplitsoftp(m.q(a))[1]), x)
jacobian_decoder(m::VAE{T,V}, z) where {T, V<:Val{:scalarsigma}} = Flux.Tracker.jacobian(a -> vec(hsplit1softp(m.g(a))[1]), z) 
jacobian_decoder(m::VAE{T,V}, z) where {T, V<:Val{:unit}} = Flux.Tracker.jacobian(a -> m.g(a), z) 

function log_det_jacobian_encoder(m::VAE, x)
	if size(x, 2) > 1
		xs = [x[:, i] for i in 1:size(x, 2)]
		return map(x -> log_det_jacobian_encoder_singleinstance(m, x), xs)
	else
		return log_det_jacobian_encoder_singleinstance(m, x)
	end
end

function log_det_jacobian_encoder_singleinstance(m::VAE, x)
	@assert size(x, 2) == 1
	s = svd(jacobian_encoder(m, x).data)
	d = reduce(+, log.(abs.(s.S))) * 2
end

function log_det_jacobian_decoder(m::VAE, z)
	if size(z, 2) > 1
		zs = [z[:, i] for i in 1:size(z, 2)]
		return map(z -> log_det_jacobian_decoder_singleinstance(m, z), zs)
	else
		return log_det_jacobian_decoder_singleinstance(m, z)
	end
end

function log_det_jacobian_decoder_singleinstance(m::VAE, z)
	@assert size(z, 2) == 1
	s = svd(jacobian_decoder(m, z).data)
	d = reduce(+, log.(abs.(s.S))) * 2
end
