# normalizecolumns(m::CuArray{T, 2}) where {T} = m ./ sqrt.(sum(m .^ 2, dims = 1)) .+ eps(T)
# normalizecolumns(m::TrackedArray{T, 2, CuArray{T, 2}}) where {T} = m ./ sqrt.(sum(m .^ 2, dims = 1)) .+ eps(T)

GPUArray{T, N} = Union{CuArray{T, N}, TrackedArray{T, N, CuArray{T, N}}}

samplehsuniform_gpu(size...) = samplehsuniform_gpu(Float32, size...)
function samplehsuniform_gpu(T::Type, size::Int...)
    v = cuzeros(T, size...)
    randn!(v)
	normalizecolumns!(v)
end

function samplez(m::SVAE, μz::GPUArray{T, 2}, κz::GPUArray{T, 2}) where {T}
	ω = sampleω(m, κz)
    v = samplehsuniform_gpu(T, size(μz, 1) - 1, size(μz, 2))
	z = householderrotation(vcat(ω, sqrt.(1 .- ω .^ 2 .+ eps(T)) .* v), μz)
	return z
end

rand_gamma_gpu(k::Number, θ::Number, size::Int...) = rand_gamma_gpu(Float32, k, θ, size...)
rand_gamma_cpu(k::Number, θ::Number, size::Int...) = rand_gamma_cpu(Float32, k, θ, size...)
rand_gamma_gpu(T::Type, k, θ, size...) = rand_gamma(T::Type, k, θ, cufill, cuzeros, size...)
rand_gamma_cpu(T::Type, k, θ, size...) = rand_gamma(T::Type, k, θ, fill, zeros, size...)
function rand_gamma(T::Type, k, θ, fillfun, zerosfun, size...)
    k = T(k)
    θ = T(θ)
    if k < 1
        u = zerosfun(T, size...)
        u .= rand!(u)
        return rand_gamma(T, 1 + k, θ, fillfun, zerosfun, size...) .* (u .^ (1 / k))
    end

    x = zerosfun(T, size...)
    v = zerosfun(T, size...)
    u = zerosfun(T, size...)

    d = k - 1 / T(3)
    c = 1 / (3 * sqrt(d))

    masku = fillfun(true, size...)

    while true
        maskv = copy(masku)
        while any(maskv)
            x[maskv] .= randn!(x[maskv])
            # println("$(x[maskv])")
            v[maskv] .= 1 .+ c .* x[maskv]
            maskv[maskv] .= v[maskv] .<= 0
            # println("maskv: $maskv")
        end

        @. v[masku] = v[masku] * v[masku] * v[masku]
        u[masku] .= rand!(u[masku])
        @. masku[masku] = masku[masku] & !(u[masku] < 1 - 0.0331 * x[masku] * x[masku] * x[masku] * x[masku])
        @. masku[masku] = masku[masku] & !(log(u[masku]) < (0.5 * x[masku] * x[masku] + d * (1 - v[masku] + log(v[masku]))))
        # println("masku: $masku")
        if !any(masku) 
            break
        end
    end
    return θ .* d .* v
end

# rand_gamma_u_sample!(masku, v, u, x, d) = rand_gamma_u_sample!(v[masku], u[masku], x[masku], d)
# function rand_gamma_u_sample!(v::AbstractArray{T, N}, u::AbstractArray{T, N}, x::AbstractArray{T, N}, d) where {T, N}
#     @. v = v * v * v
#     u .= rand!(u)
#     a = @. !(u < 1 - 0.0331 * x * x * x * x) 
#     b = @. !(log(u) < (0.5 * x * x + d * (1 - v + log(v))))
#     a .| b
# end