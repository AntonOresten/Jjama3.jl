const AnyDense = Union{Dense, LoRADense}


struct FeedForward{W<:AnyDense}
    w1::W
    w2::W
    w3::W
end

Flux.@layer FeedForward

function FeedForward(dim::Int, ff_hidden_dim::Int)
    FeedForward(
        Dense(dim => ff_hidden_dim, bias=false),
        Dense(ff_hidden_dim => dim, bias=false),
        Dense(dim => ff_hidden_dim, bias=false)
    )
end

(ff::FeedForward)(x) = ff.w2(Flux.swish(ff.w1(x)) .* ff.w3(x))


struct RMSNorm{T<:AbstractFloat,W<:AbstractVector{T}}
    weight::W
    eps::T
end

Flux.@layer RMSNorm

RMSNorm(dim::Int; eps::T=1f-5) where T = RMSNorm(ones(T, dim), eps)

function (norm::RMSNorm)(x)
    rms = sqrt.(sum(abs2, x, dims=1) / size(x, 1) .+ norm.eps)
    return x .* (norm.weight ./ rms)
end


struct RoPE{A<:AbstractArray}
    cos::A
    sin::A
end

Flux.@layer RoPE trainable=()

Base.getindex(rope::RoPE, i) = RoPE(rope.cos[:,i,:,:], rope.sin[:,i,:,:])

function apply_scaling!(freqs::AbstractVector; scale_factor=8)
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    for (i, freq) in enumerate(freqs)
        wavelen = 2π / freq
        if wavelen > low_freq_wavelen
            freqs[i] = freq / scale_factor
        elseif wavelen > high_freq_wavelen
            @assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / 
                    (high_freq_factor - low_freq_factor)
            freqs[i] = (1 - smooth) * freq / scale_factor + smooth * freq
        end
    end
    return freqs
end

function RoPE(
    dim::Int, end_pos::Int; 
    theta::T=10000f0, use_scaled=true, scale_factor=8, start_pos=0
) where T
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    use_scaled && apply_scaling!(freqs; scale_factor)
    freqs_complex = cis.(T.(start_pos:end_pos-1) * freqs')
    cos = permutedims(real(freqs_complex), (2, 1))  # (head_dim/2, seq_len)
    sin = permutedims(imag(freqs_complex), (2, 1))
    cos = reshape(cos, (dim÷2, end_pos - start_pos, 1, 1))
    sin = reshape(sin, (dim÷2, end_pos - start_pos, 1, 1))
    return RoPE(cos, sin)
end

# Note about Huggingface weights and rotary embeddings:
# https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
# Use this one if you're using the Hugging Face weights.
function (rope::RoPE)(x)
    head_dim = size(x, 1)
    x1 = x[1:head_dim÷2, :, :, :]
    x2 = x[head_dim÷2+1:end, :, :, :]
    return vcat(  
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
    )
end

function unrope(rope, x)
    head_dim = size(x, 1)
    x1 = x[1:head_dim÷2, :, :, :]
    x2 = x[head_dim÷2+1:end, :, :, :]
    return vcat(  
        x1 .* rope.cos .+ x2 .* rope.sin,
        x2 .* rope.cos .- x1 .* rope.sin
    )
end

mutable struct Attention
    wq::AnyDense
    wk::AnyDense
    wv::AnyDense
    wo::AnyDense
    dim::Int
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    cache::KVCache
end

Flux.@layer Attention trainable=(wq,wv)

function Attention(dim::Int, n_heads::Int, n_kv_heads=n_heads; qkv_bias=false)
    head_dim = dim ÷ n_heads
    Attention(
        Dense(dim => n_heads * head_dim, bias=qkv_bias),
        Dense(dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(n_heads * head_dim => dim, bias=false),
        dim,
        n_heads,
        n_kv_heads,
        head_dim,
        KVCache(Float32; n_kv_heads, head_dim),
    )
end

repeat_kv(x::AbstractArray, n_rep::Int) = isone(n_rep) ? x : repeat(x, 1, n_rep, 1, 1)

function sdpa(xq::AbstractArray{T}, xk::AbstractArray{T}, xv::AbstractArray{T}, mask::AbstractArray{T}, head_dim::Int) where T
    scores = batched_mul(batched_transpose(xk), xq) / sqrt(T(head_dim))
    scores = scores .+ mask
    sm_scores = softmax(scores; dims=1)
    return batched_mul(xv, sm_scores)
end

function (attn::Attention)(x::AbstractArray{T}, start_pos::Integer, rope=nothing, mask=false) where T
    _, seqlen, batch = size(x)

    xq = attn.wq(x)
    xk = attn.wk(x)
    xv = attn.wv(x)

    xq = reshape(xq, (attn.head_dim, attn.n_heads, seqlen, batch))
    xk = reshape(xk, (attn.head_dim, attn.n_kv_heads, seqlen, batch))
    xv = reshape(xv, (attn.head_dim, attn.n_kv_heads, seqlen, batch))

    xq = permutedims(xq, (1,3,2,4))
    xk = permutedims(xk, (1,3,2,4))
    xv = permutedims(xv, (1,3,2,4))

    if rope isa RoPE
        xq, xk = rope(xq), rope(xk)
    end

    # Update if cache is configured with seq_length > 0
    xk, xv = update!(attn.cache, start_pos, xk, xv)

    xk = repeat_kv(xk, attn.n_heads ÷ attn.n_kv_heads)
    xv = repeat_kv(xv, attn.n_heads ÷ attn.n_kv_heads)

    xq_for_attn = reshape(xq, attn.head_dim, :, attn.n_heads * batch)
    xk_for_attn = reshape(xk, attn.head_dim, :, attn.n_heads * batch)
    xv_for_attn = reshape(xv, attn.head_dim, :, attn.n_heads * batch)

    output = sdpa(xq_for_attn, xk_for_attn, xv_for_attn, mask, attn.head_dim)
    
    e_output = reshape(output, (attn.head_dim, seqlen, attn.n_heads, batch))
    p_output = permutedims(e_output, (1,3,2,4)) 
    r_output = reshape(p_output, (attn.n_heads * attn.head_dim, seqlen, batch))
    proj = attn.wo(r_output)
    return proj
end




struct TransformerBlock{A<:Attention,F<:FeedForward,AN<:RMSNorm,FN<:RMSNorm}
    attention::A
    feed_forward::F
    attention_norm::AN
    ffn_norm::FN
end

function TransformerBlock(
    dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim;
    norm_eps=1f-5, qkv_bias=false,
)
    TransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias),
        FeedForward(dim, ff_hidden_dim),
        RMSNorm(dim, eps=norm_eps),
        RMSNorm(dim, eps=norm_eps)
    )
end

function (block::TransformerBlock)(x, start_pos, rope, mask=nothing)
    h = x + block.attention(block.attention_norm(x), start_pos, rope, mask)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

Flux.@layer TransformerBlock trainable=(attention,)


mutable struct Transformer{E<:Flux.Embedding,B<:Tuple{Vararg{TransformerBlock}},N<:RMSNorm,O<:Dense,R<:RoPE}
    tok_embeddings::E
    layers::B
    norm::N
    output::O
    rope::R
    pos::Int
end

Flux.@layer Transformer trainable=(layers,)

function Transformer(
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps::T=1f-5,
    qkv_bias=false,
    rope_theta::T=500000f0,
    use_scaled_rope=false,
    scale_factor=8,
) where T
    tok_embeddings = Flux.Embedding(vocab_size => dim)
    layers = Tuple(TransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim; norm_eps=norm_eps, qkv_bias=qkv_bias) for _ in 1:n_layers)
    norm = RMSNorm(dim, eps=norm_eps)
    output = Dense(dim => vocab_size, bias=false)
    #This should probably be generated to a sane length, and then extended in the forward pass if needed.
    rope = RoPE(dim ÷ n_heads, max_seq_len * 2; theta=rope_theta, use_scaled=use_scaled_rope, scale_factor=scale_factor)
    Transformer(tok_embeddings, layers, norm, output, rope, 0)
end


function clear_cache!(model::Transformer)
    model.pos = 0
    for layer in model.layers
        clear!(layer.attention.cache)
    end
end

config_cache!(model::Transformer, seq_length) = for layer in model.layers config!(layer.attention.cache, seq_length = seq_length) end

extend_cache!(model::Transformer, seq_length) = for layer in model.layers extend!(layer.attention.cache, seq_length + model.pos) end

function rerope_cache!(model, newstart, rope_theta; range = 1:model.pos)
    dim = model.layers[1].attention.dim ÷ model.layers[1].attention.n_heads
    oldrope = model.rope[range]
    newrope = RoPE(dim, (last(range)-first(range))+newstart+1, theta=rope_theta, start_pos=newstart)
    for l in model.layers
        unroped = unrope(oldrope, l.attention.cache.cache_k[:,range,:,:])
        l.attention.cache.cache_k[:,range,:,:] .= newrope(unroped)
    end
end

function scrape_cache(model::Transformer)    
    cache = (k = [], v = [])
    for l in model.layers
        push!(cache.k, copy(l.attention.cache.cache_k[:,1:model.pos,:,:]))
        push!(cache.v, copy(l.attention.cache.cache_v[:,1:model.pos,:,:]))
    end
    return cache
end

function append_cache!(model, cache)
    if model.pos + size(cache.k[1], 2) > size(model.layers[1].attention.cache.cache_k, 2)
        extend_cache!(model, model.pos + size(cache.k[1], 2))
    end
    for (i, l) in enumerate(model.layers)
        l.attention.cache.cache_k[:, model.pos+1:model.pos+size(cache.k[i], 2), :, :] .= cache.k[i]
        l.attention.cache.cache_v[:, model.pos+1:model.pos+size(cache.v[i], 2), :, :] .= cache.v[i]
    end
    model.pos = model.pos + size(cache.k[1], 2)
end