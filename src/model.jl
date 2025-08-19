@concrete struct Transformer
    embeddings
    layers
    norm
    output
    rope
end

Flux.@layer Transformer

function Transformer(
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps = 1f-5,
    rope_settings = (theta = 500000f0, use_scaled = false, scale_factor = 8),
    head_dim = dim ÷ n_heads,
    kws...
)
    embeddings = Embedding(vocab_size => dim)
    layers = Tuple(TransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim; norm_eps, head_dim, kws...) for _ in 1:n_layers)
    norm = RMSNorm(dim, eps=norm_eps)
    output = Dense(dim => vocab_size, bias=false)
    rope = RoPE(head_dim, max_seq_len * 2; rope_settings...)
    Transformer(embeddings, layers, norm, output, rope)
end

#Note about output layer being tied to embedding: https://github.com/meta-llama/llama-models/issues/172

function masked_agg(ce, mask)
    if mask !== nothing
        ce = ce .* mask
    end
    return sum(ce)/sum(mask)
end

function (model::Transformer)(tokens::AbstractArray{Int}; caches=no_kv_cache(model), kws...)
    h = model.embeddings(tokens)
    for (layer, cache) in zip(model.layers, caches)
        rope = model.rope[pos(cache) .+ (1:size(tokens, 1))]
        h = layer(h; rope, cache, kws...)
    end
    h = model.norm(h)
    output = model.output(h)
    return output
end

function loss(logits, targets::AbstractArray; loss_mask=nothing)
    vocab_size = size(logits,1)
    gt = Flux.onehotbatch(targets, 1:vocab_size)
    if loss_mask !== nothing
        loss = Flux.logitcrossentropy(logits, gt, agg = x -> masked_agg(x, loss_mask))
    else
        loss = Flux.logitcrossentropy(logits, gt)
    end
    return loss
end

# compat
forward_inference(model, args...) = model(args...)
forward_loss(model::Transformer, inputs::AbstractArray, targets::AbstractArray; clear_cache = true, loss_mask = nothing) = loss(model(inputs, clear_cache = clear_cache), targets, loss_mask = loss_mask)

Onion.kv_cache(model::Transformer, args...; kws...) =
    ntuple(i -> kv_cache(model.layers[i].attention, args...; kws...), length(model.layers))

no_cache(k, v) = (k, v)
Onion.pos(::typeof(no_cache)) = 0

no_kv_cache(args...) = no_cache
no_kv_cache(model::Transformer) = ntuple(Returns(no_cache), length(model.layers))
