# normalizecolumns(m::CuArray{T, 2}) where {T} = m ./ sqrt.(sum(m .^ 2, dims = 1)) .+ eps(T)
# normalizecolumns(m::TrackedArray{T, 2, CuArray{T, 2}}) where {T} = m ./ sqrt.(sum(m .^ 2, dims = 1)) .+ eps(T)

GPUArray{T, N} = Union{CuArray{T, N}, TrackedArray{T, N, CuArray{T, N}}}

samplehsuniform_gpu(size::Int...) = samplehsuniform_gpu(Float32, size...)
function samplehsuniform_gpu(T::Type, size::Int...)
    v = cuzeros(T, size...)
    randn!(v)
	normalizecolumns!(v)
end

function wloss(m::SVAEbase, x::GPUArray{T, 2}, β::T, d) where {T}
	(μz, κz) = zparams(m, x)
	z = samplez(m, μz, κz)
	zp = samplehsuniform_gpu(T, size(z)...)
	Ω = d(z, zp)
	xgivenz = m.g(z)
	return Flux.mse(x, xgivenz) + β * Ω
end

function samplez(m::SVAE, μz::GPUArray{T, 2}, κz::GPUArray{T, 2}) where {T}
	ω = sampleω(m, κz)
    v = samplehsuniform_gpu(T, size(μz, 1) - 1, size(μz, 2))
    ω2 = 1 .- CUDAnative.pow.(ω, 2f0) .+ eps(T)
	z = householderrotation(vcat(ω, CUDAnative.sqrt.(ω2) .* v), μz)
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

function sampleω(model::SVAE, κ::GPUArray{T, N}) where {T, N}
    m = CuArray([T(model.zdim)])
    c2 = @. CUDAnative.pow(4κ, 2f0) + CUDAnative.pow((m - 1), 2f0)
    c = CUDAnative.sqrt.(c2)
	b = @. (-2κ + c) / (m - 1)
    a = @. (m - 1 + 2κ + c) / 4
    logm1 = CUDAnative.log.(m .- 1)
    d = @. (4 * a * b) / (1 + b) - (m - 1) * logm1
	ω = rejectionsampling(m[1], a, b, d, κ)
	return ω
end

function rejectionsampling(m, a, b, d, κ::GPUArray{T, N}) where {T, N}
    nϵ = cuzeros(T, size(a)...)
    ϵ = cuzeros(T, size(a)...)
    u = cuzeros(T, size(a)...)
    # ω = cuzeros(T, size(a)...)
    t = cuzeros(T, size(a)...)
    mask = cufill(true, size(a)...)
    
    while any(mask)
        nϵ .= rand_beta_gpu(T, (m - 1) / 2, (m - 1) / 2, size(a)...)
        ϵ .= map((m, x, nx) -> m ? nx : x, mask, ϵ, nϵ)
        rand!(u)
        t = @. 2 * a * b / (1 - (1 - b) * ϵ)
        mask = @. mask & ((m - 1) * CUDAnative.log(t) - t + d >= CUDAnative.log(u))
	end
	return @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
end

