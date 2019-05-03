# normalizecolumns(m::CuArray{T, 2}) where {T} = m ./ sqrt.(sum(m .^ 2, dims = 1)) .+ eps(T)
# normalizecolumns(m::TrackedArray{T, 2, CuArray{T, 2}}) where {T} = m ./ sqrt.(sum(m .^ 2, dims = 1)) .+ eps(T)
