@concrete struct FeedForward
    w1
    w2
    w3
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


@concrete struct RMSNorm
    weight
    eps
end

Flux.@layer RMSNorm

RMSNorm(dim::Int; eps=1f-5) = RMSNorm(ones(typeof(eps), dim), eps)

function (norm::RMSNorm)(x)
    rms = sqrt.(sum(abs2, x, dims=1) ./ size(x, 1) .+ norm.eps)
    return x .* (norm.weight ./ rms)
end


@concrete struct RoPE
    cos
    sin
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
    n = size(x, 1)
    k = n ÷ 2
    x1 = selectdim(x, 1, 1:k)
    x2 = selectdim(x, 1, 1+k:n)
    return vcat(  
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
    )
end

function unrope(rope, x)
    n = size(x, 1)
    k = n ÷ 2
    x1 = selectdim(x, 1, 1:k)
    x2 = selectdim(x, 1, 1+k:n)
    return vcat(  
        x1 .* rope.cos .+ x2 .* rope.sin,
        x2 .* rope.cos .- x1 .* rope.sin
    )
end

@concrete struct Attention
    wq
    wk
    wv
    wo
    heads::Int
    kv_repeat::Int
end

Flux.@layer Attention

function Attention(embed_dim, heads, kv_heads=heads; qkv_bias=false)
    head_dim = embed_dim ÷ heads
    return Attention(
        Dense(embed_dim => heads * head_dim, bias=qkv_bias),
        Dense(embed_dim => kv_heads * head_dim, bias=qkv_bias),
        Dense(embed_dim => kv_heads * head_dim, bias=qkv_bias),
        Dense(heads * head_dim => embed_dim, bias=false),
        heads, heads ÷ kv_heads)
end

function (layer::Attention)(xq, xk=xq; rope=identity, cache=tuple, dpa=dpa, mask=nothing)
    q, k, v = layer.wq(xq), layer.wk(xk), layer.wv(xk)
    q, k, v = rearrange.((q, k, v), einops"(dim heads) ... -> dim heads ..."; layer.heads)
    q, k = rope(q), rope(k)
    k, v = cache(k, v)
    k, v = repeat.((k, v), einops"dim len heads ... -> dim len (kv_repeat heads) ..."; layer.kv_repeat)
    x = dpa(q, k, v; mask)
    x = rearrange(x, einops"dim heads ... -> (dim heads) ...")
    return layer.wo(x)
end

@concrete struct TransformerBlock
    attention
    feed_forward
    attention_norm
    ffn_norm
end

Flux.@layer TransformerBlock

function TransformerBlock(
    embed_dim, heads, kv_heads=heads, ff_hidden_dim=4*embed_dim;
    norm_eps=1f-5, qkv_bias=false
)
    TransformerBlock(
        Attention(embed_dim, heads, kv_heads; qkv_bias),
        FeedForward(embed_dim, ff_hidden_dim),
        RMSNorm(embed_dim, eps=norm_eps),
        RMSNorm(embed_dim, eps=norm_eps)
    )
end

function (block::TransformerBlock)(x; kwargs...)
    h = x + block.attention(block.attention_norm(x); kwargs...)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

@concrete struct Transformer
    tok_embeddings
    layers
    norm
    output
    rope
end

Flux.@layer Transformer

function Transformer(
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps::T=1f-5,
    qkv_bias=false,
    rope_theta::T=500000f0,
    use_scaled_rope=false,
    scale_factor=8
) where T
    tok_embeddings = Embedding(vocab_size => dim)
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