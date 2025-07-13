no_cache(k, v) = (k, v)
position(::typeof(no_cache)) = 0

struct KVCache{T,A<:AbstractArray{T}}
    k::A
    v::A
    pos::Ref{Int}
end

function Base.show(io::IO, ::MIME"text/plain", cache::KVCache)
    println(io, typeof(cache), ':')
    println(io, "  size: $(size(cache.k))")
    println(io, "  position: $(position(cache)) / $(size(cache.k, 2))")
    print(io, "  batches: $(size(cache.k, 4))")
end

function kv_cache(layer::Attention, len::Int, batch::Int=1)
    k = similar(layer.wq.weight, layer.head_dim, len, layer.n_kv_heads, batch) .= 0
    v = similar(layer.wv.weight, layer.head_dim, len, layer.n_kv_heads, batch) .= 0
    return KVCache(k, v, Ref(0))
end

no_kv_cache(layer::Attention) = no_cache

function extend(cache::KVCache, new_len::Int)
    head_dim, len, kv_heads, batch = size(cache.k)
    @assert new_len > len
    k = similar(cache.k, head_dim, new_len, kv_heads, batch) .= 0
    v = similar(cache.v, head_dim, new_len, kv_heads, batch) .= 0
    k[:, 1:len, :, :] .= cache.k
    v[:, 1:len, :, :] .= cache.v
    return KVCache(k, v, cache.pos)
end

position(cache::KVCache) = cache.pos[]
position!(cache::KVCache, new_pos::Int) = cache.pos[] = new_pos

function (cache::KVCache)(k::AbstractArray, v::AbstractArray)
    cache.k[:, position(cache) .+ axes(k, 2), :, :] .= k
    cache.v[:, position(cache) .+ axes(v, 2), :, :] .= v
    position!(cache, position(cache) + size(k, 2))
    return @views cache.k[:, 1:position(cache), :, :], cache.v[:, 1:position(cache), :, :]
end


struct KVCacheStack{C}
    caches::Vector{C}
end

Base.iterate(cache::KVCacheStack, state...) = iterate(cache.caches, state...)

function Base.show(io::IO, ::MIME"text/plain", cache::KVCacheStack)
    println(io, typeof(cache), ':')
    println(io, "  caches: $(length(cache.caches))")
    println(io, "  position: $(position(cache))")
end

extend(cache::KVCacheStack, new_len::Int) = KVCacheStack([extend(c, new_len) for c in cache.caches])

position(cache::KVCacheStack) = only(unique(position.(cache.caches)))
position!(cache::KVCacheStack, new_pos::Int) = only(unique(position!.(cache.caches, new_pos)))

function kv_cache(model::Transformer, args...; kws...)
    return KVCacheStack([kv_cache(layer.attention, args...; kws...) for layer in model.layers])
end

no_kv_cache(model::Transformer) = KVCacheStack([no_kv_cache(layer.attention) for layer in model.layers])
