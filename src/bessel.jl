using DiffRules
using SpecialFunctions

mybesselix(ν, x) = besselix.(ν, x)

∇mybesselix(ν, x::Flux.Tracker.TrackedMatrix) = ∇mybesselix(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesselix(ν, x::Flux.Tracker.TrackedReal) = ∇mybesselix(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesselix(ν, x::Flux.Tracker.TrackedArray) = ∇mybesselix(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesselix(ν, x) = @. mybesselix(ν - 1, x) - mybesselix(ν, x) * (ν + x) / x

SpecialFunctions.besselix(ν, x::Flux.Tracker.TrackedMatrix) = Flux.Tracker.track(mybesselix, ν, x)
SpecialFunctions.besselix(ν::Real, x::Flux.Tracker.TrackedReal) = Flux.Tracker.track(mybesselix, ν, x)
SpecialFunctions.besselix(ν, x::Flux.Tracker.TrackedArray) = Flux.Tracker.track(mybesselix, ν, x)
SpecialFunctions.besselix(ν, x::AbstractArray) = besselix.(ν, x)

Flux.Tracker.@grad function mybesselix(ν, x)
    return mybesselix(Flux.Tracker.data(ν), Flux.Tracker.data(x)), Δ -> (nothing, ∇mybesselix(ν, x) .* Δ)
end

mybesseli(ν, x) = besseli.(ν, x)

∇mybesseli(ν, x::Flux.Tracker.TrackedMatrix) = ∇mybesseli(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesseli(ν, x::Flux.Tracker.TrackedReal) = ∇mybesseli(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesseli(ν, x::Flux.Tracker.TrackedArray) = ∇mybesseli(Flux.Tracker.data(ν), Flux.Tracker.data(x))
∇mybesseli(ν, x) = @. (besseli(ν - 1, x) + besseli(ν + 1, x)) / 2

SpecialFunctions.besseli(ν, x::Flux.Tracker.TrackedMatrix) = Flux.Tracker.track(mybesseli, ν, x)
SpecialFunctions.besseli(ν::Real, x::Flux.Tracker.TrackedReal) = Flux.Tracker.track(mybesseli, ν, x)
SpecialFunctions.besseli(ν, x::Flux.Tracker.TrackedArray) = Flux.Tracker.track(mybesseli, ν, x)
SpecialFunctions.besseli(ν, x::AbstractArray) = besseli.(ν, x)

Flux.Tracker.@grad function mybesseli(ν, x)
    return mybesseli(Flux.Tracker.data(ν), Flux.Tracker.data(x)), Δ -> (nothing, ∇mybesseli(ν, x) .* Δ)
end
