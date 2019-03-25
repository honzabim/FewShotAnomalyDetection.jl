using Random
using Statistics

"""
		hsplitsoftp(x, ϵ = 1f-5)

		Splits x horizontally into two equal parts and use softplus to the lower part.
		ϵ is added to the soft-plus to prevent underflow and improve the numerical stability

"""
hsplitsoftp(x,ϵ = 1f-5) = x[1:size(x, 1) ÷ 2, :], softplus.(x[size(x, 1) ÷ 2 + 1 : 2 * (size(x, 1) ÷ 2), :] .+ ϵ)
hsplit1softp(x,ϵ = 1f-5) = x[1:size(x, 1) - 1, :], softplus.(x[end, :] .+ ϵ)

samplesoftmax(p) = mapslices(v -> StatsBase.sample(1:size(p,1),StatsBase.Weights(v)),p,1)[:]

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
	# println("x: $(size(x)) μx: $(size(μx)) σ2x: $(size(σ2x'))")
	-mean(log_normal(x,μx,collect(σ2x'))) + m.β * mean(kldiv(μz,σ2z))
end

function loss(m::VAE{T,V},x) where {T,V<:Val{:unit}}
	μz, σ2z, μx = infer(m,x)
	-mean(log_normal(x,μx)) + m.β * mean(kldiv(μz,σ2z))
end

function samplez(m::VAE{T,V}, x) where {T,V}
	z = gaussiansample(zparams(m, x)...)
end

function zparams(m::VAE{T,V}, x) where {T,V}
	μz, σ2z = hsplitsoftp(m.q(x))
	return μz, σ2z
end

function printingloss(m::VAE{T,V},x) where {T,V<:Val{:scalarsigma}}
	μz, σ2z, μx, σ2x = infer(m,x)
	println("loglkl: $(-mean(log_normal(x,μx,collect(σ2x')))) | KL: $(mean(kldiv(μz,σ2z)))")
	-mean(log_normal(x,μx,collect(σ2x'))) + m.β*mean(kldiv(μz,σ2z))
end

function printingloss(m::VAE{T,V},x) where {T,V<:Val{:unit}}
	μz, σ2z, μx = infer(m,x)
	println("MSE:$(-mean(log_normal(x,μx))) Flux.mse: $(Flux.mse(x, μx)) KL: $(mean(kldiv(μz,σ2z)))")
	-mean(log_normal(x,μx)) + m.β*mean(kldiv(μz,σ2z))
end

# """
# 		function px(m::T,x,r::Int = 100) where {T<:Union{VAE,IPMAE}}
#
# 		probability of sample x given the model estimated from `r` samples
# """
# function px(m::T,x,k::Int = 100, σ = 1.0) where {T<:VAE}
# 	xx = FluxExtensions.scatter(x, k)
# 	μz, σ2z = hsplitsoftp(m.q(xx))
# 	z = gaussiansample(μz,σ2z)
# 	μx  = m.g(z)
#
# 	lkl = log_normal(xx, μx, σ)
# 	gather(logsumexp,lkl,k) ./ ((k>1) ? log(k) : 1)
# end

function pxvita(m::T,x, σ = 1.0) where {T<:VAE}
	μz, σ2z = hsplitsoftp(m.q(x))
	log_normal(x,m.g(μz),σ)
end
