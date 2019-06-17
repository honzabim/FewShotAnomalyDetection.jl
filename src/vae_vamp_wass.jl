mutable struct VAEvamp{T<:AbstractFloat,V<:Val}
	q  # encoder (inference modul)
    g  # decoder (generator)
    centroids::AbstractMatrix{T}
    labels::Vector{Int}
	variant::V
end

VAEvamp(q, g, c, l, s::Symbol = :unit) = VAEvamp(q, g, c, l, Val(s))

Flux.@treelike(VAEvamp)

function add_labeled_centroid!(m::VAEvamp, c, l)
    m.centroids = hcat(m.centroids, c)
    push!(m.labels, l)
end

function sample_vamp(m::VAEvamp{T,V}, n) where {T,V<:Val{:unit}}
    components = rand(1:size(m.centroids, 2), n)
	μz = m.q(m.centroids[:, components])
	return gaussiansample(μz, 1f0)
end

function sample_vamp(m::VAEvamp{T,V}, n0, n1) where {T,V<:Val{:unit}}
    normal_ids = findall(m.labels .== 0)
    anomalous_ids = findall(m.labels .== 1)
    components_0 = rand(1:length(normal_ids), n0)
    components_1 = rand(1:length(anomalous_ids), n1)
    μz0 = m.q(m.centroids[:, components_0])
    μz1 = m.q(m.centroids[:, components_1])
	return gaussiansample(μz0, 1f0), gaussiansample(μz1, 1f0)
end

function wloss(m::VAEvamp{T,V}, x, d, β) where {T,V<:Val{:unit}}
    μz = m.q(x)
    μx = m.g(μz)
    prior = sample_vamp(m, size(x, 2))
    Ω = d(μz, prior)
	-mean(log_normal(x, μx)) + β * Ω
end

function wloss(m::VAEvamp{T,V}, x, l, d, β, α) where {T,V<:Val{:unit}}
    normal_ids = findall(l .== 0)
    anomalous_ids = findall(l .== 1)
    if length(anomalous_ids) == 0
        return wloss(m, x, d, β)
    end

    μz = m.q(x)
    μx = m.g(μz)
    prior_0, prior_1 = sample_vamp(m, length(normal_ids), length(anomalous_ids))
    Ω0 = d(μz[:, normal_ids], prior_0)
    Ω1 = d(μz[:, anomalous_ids], prior_1)
    -mean(log_normal(x, μx)) + β * ((1f0 - α) * Ω0 + α * Ω1)
end

function log_pxexpectedz(m::VAEvamp{T, V}, x) where {T, V<:Val{:unit}}
	μz = m.q(x)
	log_normal(x, m.g(μz))
end

function pz(m::VAEvamp{T,V}, x) where {T,V<:Val{:unit}}
	if size(x, 2) > 1
		xs = [x[:, i] for i in 1:size(x, 2)]
		return hcat(map(x -> collect(pz_singleinstance(m, x)), xs)...)
	else
		return pz_singleinstance(m, x)
	end
end

function pz_singleinstance(m::VAEvamp{T,V}, x) where {T,V<:Val{:unit}}
	@assert size(x, 2) == 1
	μz = m.q(x).data
    centroids_0 = m.centroids[m.labels .== 0]
    centroids_1 = m.centroids[m.labels .== 1]
	p0 = 0
	for i in 1:size(centroids_0, 2)
		p0 += exp.(log_normal(μz, centroids_0[:, i]))
	end
    p0 /= size(centroids_0, 2)
    p1 = 0
	for i in 1:size(centroids_1, 2)
		p1 += exp.(log_normal(μz, centroids_1[:, i]))
	end
    p1 /= size(centroids_1, 2)
    return p0, p1
end
