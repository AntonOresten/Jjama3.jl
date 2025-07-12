@concrete struct KVCache
    k
    v
    position::Int
end

Base.copy(cache::KVCache) = KVCache(copy(cache.k), copy(cache.v))

Flux.@layer KVCache

head_dim(cache::KVCache) = size(cache.k, 1)
seq_length(cache::KVCache) = size(cache.k, 2)
n_kv_heads(cache::KVCache) = size(cache.k, 3)
batch_size(cache::KVCache) = size(cache.k, 4)

function KVCache(T; head_dim, seq_length=0, n_kv_heads, batch_size=1)
    cache_k = zeros(T, head_dim, seq_length, n_kv_heads, batch_size)
    cache_v = zeros(T, head_dim, seq_length, n_kv_heads, batch_size)
    return KVCache(cache_k, cache_v)
end

function config!(cache::KVCache; seq_length=seq_length(cache), batch_size=batch_size(cache))
    cache.k = similar(cache.k, head_dim(cache), seq_length, n_kv_heads(cache), batch_size) .= 0
    cache.v = similar(cache.v, head_dim(cache), seq_length, n_kv_heads(cache), batch_size) .= 0
end

function extend!(cache::KVCache, new_total_length::Int)
    old_cache = copy(cache)
    config!(cache, seq_length=new_total_length)
    cache.k[:, 1:seq_length(old_cache), :, :] .= old_cache.k
    cache.v[:, 1:seq_length(old_cache), :, :] .= old_cache.v
end

clear!(cache::KVCache) = config!(cache, seq_length=0)

function update!(cache::KVCache, start_pos::Int, k::AbstractArray, v::AbstractArray)
    if iszero(seq_length(cache))
        return k, v
    else
        seqlen = size(k, 2)
        cache.k[:, start_pos+1:start_pos+seqlen, :, :] .= k
        cache.v[:, start_pos+1:start_pos+seqlen, :, :] .= v
        return cache.k[:, 1:start_pos+seqlen, :, :],
            cache.v[:, 1:start_pos+seqlen, :, :]
    end
end
