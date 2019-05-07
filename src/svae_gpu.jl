# normalizecolumns(m::CuArray{T, 2}) where {T} = m ./ sqrt.(sum(m .^ 2, dims = 1)) .+ eps(T)
# normalizecolumns(m::TrackedArray{T, 2, CuArray{T, 2}}) where {T} = m ./ sqrt.(sum(m .^ 2, dims = 1)) .+ eps(T)

using CUDAnative

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


rand_gamma_cpu(k::Number, θ::Number, size::Int...) = rand_gamma_cpu(Float32, k, θ, size...)
rand_gamma_cpu(T::Type, k::Number, θ::Number, size::Int...) = rand_gamma(T::Type, k, θ, fill, zeros, size...)
function rand_gamma(T::Type, k, θ, fillfun, zerosfun, size...)
    k = T(k)
    θ = T(θ)
    if k < 1
        u = zerosfun(T, size...)
        rand!(u)
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
            v[maskv] .= 1 .+ c .* x[maskv]
            maskv[maskv] .= v[maskv] .<= 0
        end

        @. v[masku] = v[masku] * v[masku] * v[masku]
        u[masku] .= rand!(u[masku])
        @. masku[masku] = masku[masku] & !(u[masku] < 1 - 0.0331 * x[masku] * x[masku] * x[masku] * x[masku])
        @. masku[masku] = masku[masku] & !(log(u[masku]) < (0.5 * x[masku] * x[masku] + d * (1 - v[masku] + log(v[masku]))))
        if !any(masku) 
            break
        end
    end
    return θ .* d .* v
end

rand_gamma_gpu(k::Number, θ::Number, size::Int...) = rand_gamma_gpu(Float32, k, θ, size...)
function rand_gamma_gpu(T::Type, k, θ, size...)
    k = T(k)
    θ = T(θ)
    if k < 1
        u = cuzeros(T, size...)
        rand!(u)
        return rand_gamma_gpu(T, 1 + k, θ, size...) .* CUDAnative.pow.(u, (1 / k))
    end

    x = cuzeros(T, size...)
    v = cuzeros(T, size...)
    u = cuzeros(T, size...)

    d = k - 1 / T(3)
    c = 1 / (3 * sqrt(d))

    masku = cufill(true, size...)
    while any(masku)
        maskv = copy(masku)
        while any(maskv)
            v .= map((m, v, nv) -> m ? nv : v, maskv, v, next_gamma_v_sample(x, c))
            maskv .= maskv .& (v .<= 0)
        end
        x .= (v .- 1) ./ c

        v .= map((m, v, nv) -> m ? nv : v, masku, v, v .* v .*v)
        rand!(u)
        @. masku = masku & !(u < 1 - 0.0331 * x * x * x * x)
        @. masku = masku & !(CUDAnative.log(u) < (0.5 * x * x + d * (1 - v + CUDAnative.log(v))))
    end
    return θ .* d .* v
end

function next_gamma_v_sample(x, c)
    randn!(x)
    return 1 .+ c .* x
end

rand_beta_gpu(α::Number, β::Number, size::Int...) = rand_beta_gpu(Float32, α, β, size...)
function rand_beta_gpu(T::Type, α, β, size...)
    α = T(α)
    β = T(β)

    if (α > 1) || (β > 1)
        g1 = rand_gamma_gpu(T, α, 1, size...)
        g2 = rand_gamma_gpu(T, β, 1, size...)
        return g1 ./ (g1 .+ g2)
    end

    u = cuzeros(T, size...)
    v = cuzeros(T, size...)
    x = cuzeros(T, size...)
    y = cuzeros(T, size...)

    mask = cufill(true, size...)
    while any(mask)
        rand!(u)
        rand!(v)
        x .= map((m, x, nx) -> m ? nx : x, mask, x, CUDAnative.pow.(u, (1 / α)))
        y .= map((m, y, ny) -> m ? ny : y, mask, y, CUDAnative.pow.(v, (1 / β)))
        @. mask = mask & ((x + y) > 1)
    end

    return map((x, y) -> (x + y) > 0 ? x / (x + y) : log_beta_expression(CUDAnative.log(x), CUDAnative.log(y)), x, y)
end

function log_beta_expression(logX, logY)
    logM = logX > logY ? logX : logY
    logX -= logM
    logY -= logM
    return CUDAnative.exp(logX - CUDAnative.log(CUDAnative.exp(logX) + CUDAnative.exp(logY)))
end

