normalizecolumns(m::AbstractArray{T, 2}) where {T<:Number} = m ./ sqrt.(sum(m .^ 2, dims = 1) .+ eps(T))
function normalizecolumns!(m::AbstractArray{T, 2}) where {T<:Number}
	m .= normalizecolumns(m)
end

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
log_normal(x, μ, σ2::Vector{T}) where {T<:Number} = - sum((x - μ) .^ 2 ./ σ2' .+ log.(σ2' .* 2π), dims = 1) / 2
log_normal(x, μ, Σ::Matrix{T}) where {T<:Number} = - (log(det(Σ)) + (x - μ)' * inv(Σ) * (x - μ) + size(Σ, 1) * log(2π)) / 2

"""
	vmfentropy(m, κ)

	Entropy of Von Mises-Fisher distribution
"""
vmfentropy(m, κ) = .-κ .* besselix(m / 2, κ) ./ besselix(m / 2 - 1, κ) .- ((m ./ 2 .- 1) .* log.(κ) .- (m ./ 2) .* log(2π) .- (κ .+ log.(besselix(m / 2 - 1, κ))))

"""
	huentropy(m)

	Entropy of Hyperspherical Uniform distribution
"""
huentropy(m) = m / 2 * log(π) + log(2) - lgamma(m / 2)

# Likelihood estimation of a sample x under VMF with given parameters taken from https://pdfs.semanticscholar.org/2b5b/724fb175f592c1ff919cc61499adb26996b1.pdf
# normalizing constant for density function of VMF
c(p, κ) = κ ^ (p / 2 - 1) / ((2π) ^ (p / 2) * besseli(p / 2 - 1, κ))

# log likelihood of one sample under the VMF dist with given parameters
log_vmf_c(x, μ, κ) = κ * μ' * x .+ log(c(length(μ), κ))
log_vmf_c(x::AbstractMatrix, μ::AbstractMatrix, κ::T) where {T <: Number} = [log_vmf_c(x[:, i], μ[:, i], κ) for i in size(x, 2)] 
log_vmf_c(x::AbstractMatrix, μ::AbstractMatrix, κ::AbstractVector) = [log_vmf_c(x[:, i], μ[:, i], κ[i]) for i in size(x, 2)] 
log_vmf_wo_c(x, μ, κ) = κ * μ' * x

pairwisecos(x, y) = max.(1 .- (x' * y), 0) # This is a bit of a hack to avoid the distance being negative due to float
pairwisecos(x) = pairwisecos(x, x)

k_imq(x, y, c) = sum( c./ (c .+ pairwisecos(x, y))) / (size(x, 2) * size(y, 2))
k_imq(x::T, c) where {T <: AbstractMatrix} = sum(c ./ (c .+ pairwisecos(x))) / (size(x, 2) * (size(x, 2) - 1))
k_imq(x::T, c) where {T <: AbstractVector} = zero(eltype(x))

mmd_imq(x,y,c) = k_imq(x,c) + k_imq(y,c) - 2 * k_imq(x,y,c)

function gaussiansample(μ, σ2)
	ϵ = randn!(similar(μ,size(μ)))
	μ .+ sqrt.(σ2) .* ϵ
end

function samplehsuniform(size...)
	v = randn(size...)
	v = normalizecolumns(v)
end