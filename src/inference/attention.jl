# this file is just a proof of concept
# it avoids some permutations, and calls
# what should be the more efficient batched
# matrix-vector multiplication,
# but it might not be better in practice.
# also, batched_vec can't handle
# non-copying grouped-query attention

using NNlib: batched_transpose, batched_vec

function cached_autoregressive_attention_inference(
    q::AbstractArray{T}, k::AbstractArray{T}, v::AbstractArray{T};
    cache
) where T
    dim = size(q, 1)
    num_attention_heads, num_key_value_heads = size(q, 3), size(k, 3)
    num_q_per_kv = num_attention_heads ÷ num_key_value_heads
    scale = T(inv(√dim))
    q = rearrange(q, einops"d h ... -> d (h ...)")
    k, v = repeat.((k, v), einops"d h ... -> d 1 (r h) ..."; r=num_q_per_kv) # TODO: custom kernel without repeat
    k, v = cache(k, v)
    k, v = rearrange.((k, v), einops"d l h ... -> d l (h ...)")
    score = batched_vec(batched_transpose(k), q) .* scale
    return batched_vec(v, softmax!(score, dims=1))
end

function (layer::Attention)(
    x::AbstractArray, attention::typeof(cached_autoregressive_attention_inference);
    rope = identity, cache, kws...
)
    q, k, v = layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)
    q, k, v = rearrange.((q, k, v), einops"(d h) 1 ... -> d h ..."; d=layer.head_dim)
    q, k = layer.q_norm(q), layer.k_norm(k)
    q, k = rope.((q, k))
    k, v = cache(k, v)
    y = attention(q, k, v; kws...)
    y = rearrange(y, einops"d h ... -> (d h) 1 ...")
    return layer.o_proj(y)
end
