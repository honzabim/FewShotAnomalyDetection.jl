abstract type SVAE end

"""
	kldiv(model::SVAE, κ)

	KL divergence between Von Mises-Fisher and Hyperspherical Uniform distributions
"""
kldiv(model::SVAE, κ) = .- vmfentropy(model.zdim, κ) .+ model.hue

function log_pxexpectedz(m::SVAE, x)
	xgivenz = m.g(zparams(m, x)[1])
	Flux.Tracker.data(log_normal(x, xgivenz))
end

jacobian_encoder(m::SVAE, x) = Flux.Tracker.jacobian(a -> Chain(m.q, m.μzfromhidden)(a), x)

function log_pz_jacobian_encoder(m::SVAE, x)
	if size(x, 2) > 1
		xs = [x[:, i] for i in 1:size(x, 2)]
		return map(x -> log_pz_jacobian_encoder_singleinstance(m, x), xs)
	else
		return log_pz_jacobian_encoder_singleinstance(m, x)
	end
end

function log_pz_jacobian_encoder_singleinstance(m::SVAE, x)
	@assert size(x, 2) == 1
	s = svd(jacobian_encoder(m, x).data)
	d = reduce(+, log.(abs.(s.S))) * 2
	d + log_pz(m, x)
end

jacobian_decoder(m::SVAE, z) = Flux.Tracker.jacobian(a -> m.g(a), z)

function log_pz_jacobian_decoder(m::SVAE, z)
	if size(z, 2) > 1
		zs = [z[:, i] for i in 1:size(z, 2)]
		return map(z -> log_pz_jacobian_decoder_singleinstance(m, z), zs)
	else
		return log_pz_jacobian_decoder_singleinstance(m, z)
	end
end

function log_pz_jacobian_decoder_singleinstance(m::SVAE, z)
	@assert size(z, 2) == 1
	s = svd(jacobian_decoder(m, z).data)
	d = reduce(+, log.(abs.(s.S))) * 2
	-d + log_pz_from_z(m, z)
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
	# return model.μzfromhidden(hidden), max.(model.κzfromhidden(hidden), 100)
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
	z = householderrotation(vcat(ω, sqrt.(1 .- ω .^ 2 .+ (10 * eps(Float32))) .* v), μz)
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
	ω = rejectionsampling(m, a, b, d, κ)
	return ω
end

function rejectionsampling(m, a, b, d, κ)
	beta = Beta((m - 1) / 2, (m - 1) / 2)
	T = eltype(Flux.Tracker.data(a))
	ϵ, u = Adapt.adapt(T, rand(beta, size(a)...)), rand(T, size(a)...)

	accepted = isaccepted(ϵ, u, m, Flux.data(a), Flux.Tracker.data(b), Flux.data(d))
	it = 0
	while (!all(accepted)) & (it < 10000)
		mask = .! accepted
		ϵ[mask] = Adapt.adapt(T, rand(beta, sum(mask)))
		u[mask] = rand(T, sum(mask))
		accepted[mask] = isaccepted(mask, ϵ, u, m, Flux.data(a), Flux.data(b), Flux.data(d))
		it += 1
	end
	if it >= 10000
		println("Warning - sampler was stopped by 10000 iterations - it did not accept the sample!")
		println(m)
		println(a)
		println(b)
		println(d)
		println(κ)
		println(accepted)
	end
	return @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
end

# function rejectionsampling(m, a, b, d)
# 	beta = Beta((m - 1) / 2, (m - 1) / 2)
# 	T = eltype(Flux.Tracker.data(a))
# 	ϵ, u = Adapt.adapt(T, rand(beta, size(a)...)), rand(T, size(a)...)
#
# 	accepted = isaccepted(ϵ, u, m, Flux.data(a), Flux.Tracker.data(b), Flux.data(d))
# 	while !all(accepted)
# 		mask = .! accepted
# 		ϵ[mask] = Adapt.adapt(T, rand(beta, sum(mask)))
# 		u[mask] = rand(T, sum(mask))
# 		ia = isaccepted(mask, ϵ, u, m, Flux.data(a), Flux.data(b), Flux.data(d))
# 		accepted[mask] = ia
# 	end
# 	return @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
# end

isaccepted(mask, ϵ, u, m:: Int, a, b, d) = isaccepted(ϵ[mask], u[mask], m, a[mask], b[mask], d[mask]);
function isaccepted(ϵ, u, m:: Int, a, b, d)
	ω = @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
	t = @. 2 * a * b / (1 - (1 - b) * ϵ)
	@. (m - 1) * log(t) - t + d >= log(u)
end
