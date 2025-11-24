using Onion: zeros_like

@concrete struct KVCache
    k; v
end

Base.length(cache::KVCache) = size(cache.k, 2)

function (cache::KVCache)(k, v, pos)
    num_new_tokens = size(k, 2)
    pos + num_new_tokens > length(cache) && throw(ArgumentError("Cache is full"))
    selectdim(cache.k, 2, pos .+ axes(k, 2)) .= k
    selectdim(cache.v, 2, pos .+ axes(v, 2)) .= v
    new_pos = pos + size(k, 2)
    new_k = selectdim(cache.k, 2, 1:new_pos)
    new_v = selectdim(cache.v, 2, 1:new_pos)
    return new_k, new_v
end

withposition(cache::KVCache, pos) = (k, v) -> cache(k, v, pos)

function kv_cache(layer::Attention, context::Int)
    weight = layer.k_proj.weight
    num_key_value_heads = size(weight, 1) ÷ layer.head_dim
    k = zeros_like(weight, layer.head_dim, context, num_key_value_heads)
    v = zeros_like(weight, layer.head_dim, context, num_key_value_heads)
    return KVCache(k, v)
end

kv_cache(layer::Block, args...) = kv_cache(layer.attention, args...)
kv_cache(layer::Model, args...) = ntuple(index -> kv_cache(layer.blocks[index], args...), length(layer.blocks))

no_kv_cache(::Block, args...) = tuple
no_kv_cache(layer::Model, args...) = ntuple(Returns(tuple), length(layer.blocks))
