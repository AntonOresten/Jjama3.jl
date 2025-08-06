### Layers ###

mutable struct ConditionalTransformer{E<:Flux.Embedding,C,B<:Tuple{Vararg{TransformerBlock}},N<:RMSNorm,O<:Dense,R<:RoPE}
    tok_embeddings::E
    cond_embeddings::C
    layers::B
    norm::N
    output::O
    rope::R
    pos::Int
end

Flux.@layer ConditionalTransformer trainable=(layers,)

function ConditionalTransformer(cond_embeddings::Tuple,
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps::T=1f-5,
    qkv_bias=false,
    rope_theta::T=500000f0,
    use_scaled_rope=false,
    scale_factor=8
) where T
    tok_embeddings = Flux.Embedding(vocab_size => dim)
    layers = Tuple(TransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim; norm_eps=norm_eps, qkv_bias=qkv_bias) for _ in 1:n_layers)
    norm = RMSNorm(dim, eps=norm_eps)
    output = Dense(dim => vocab_size, bias=false)
    #This should probably be generated to a sane length, and then extended in the forward pass if needed.
    rope = RoPE(dim ÷ n_heads, max_seq_len * 2; theta=rope_theta, use_scaled=use_scaled_rope, scale_factor=scale_factor)
    ConditionalTransformer(tok_embeddings, cond_embeddings, layers, norm, output, rope, 0)
end

function ConditionalTransformer(cond_embedding,
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps::T=1f-5,
    qkv_bias=false,
    rope_theta::T=500000f0,
    use_scaled_rope=false,
    scale_factor=8
) where T
    ConditionalTransformer((cond_embedding, ), vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len, ff_hidden_dim; norm_eps, qkv_bias, rope_theta, use_scaled_rope, scale_factor)
end


function clear_cache!(model::ConditionalTransformer)
    model.pos = 0
    for layer in model.layers
        clear!(layer.attention.cache)
    end
end

config_cache!(model::ConditionalTransformer, seq_length) = for layer in model.layers config!(layer.attention.cache, seq_length = seq_length) end

extend_cache!(model::ConditionalTransformer, seq_length) = for layer in model.layers extend!(layer.attention.cache, seq_length + model.pos) end

function scrape_cache(model::ConditionalTransformer)    
    cache = (k = [], v = [])
    for l in model.layers
        push!(cache.k, copy(l.attention.cache.cache_k[:,1:model.pos,:,:]))
        push!(cache.v, copy(l.attention.cache.cache_v[:,1:model.pos,:,:]))
    end
    return cache
end

### Model ###

function (model::ConditionalTransformer)(tokens::AbstractArray{Int}, conditionals::Tuple, opt_state; clear_cache = false, checkpoint_func = wrap, sdpa_func = sdpa, conditional_list = 1:length(conditionals))
    if clear_cache
        Flux.ChainRulesCore.ignore_derivatives() do
            Jjama3.clear_cache!(model)
        end
    end
    h = model.tok_embeddings(tokens) # Embedding: (dim, seq_len, batch)
    for (ic, c) in enumerate(conditional_list)
        cond_emb = model.cond_embeddings[c]
        cond = conditionals[ic]
        h = h .+ rearrange(cond_emb(cond), (:dim, :batch) --> (:dim, 1, :batch))
    end
    rope = model.rope[model.pos+1:model.pos+size(tokens, 1)]
    if size(h, 2) == 1 #If there is only one new token, then a 1-by-1 mask = 0 works, via broadcasting (if the attention functions allow it)
        mask = Jjama3.create_mask(h)
    else
        mask = Jjama3.create_mask(h; precached_size = model.pos)
    end
    for i in 1:length(model.layers)
        if !isnothing(opt_state)
            #If checkpoint_func is also just wrap, then this does nothing, but if its Zygote.checkpointed, then this is a checkpointed update
            h = checkpoint_func(wrap, eager_update!(opt_state.layers[i], model.layers[i], Optimisers.update!), h, model.pos, rope, mask, sdpa_func)   
        else
            h = checkpoint_func(wrap, model.layers[i], h, model.pos, rope, mask, sdpa_func)
        end
    end
    h = model.norm(h)
    output = model.output(h)
    model.pos += size(tokens, 1)
    return output
end

(model::ConditionalTransformer)(tokens::AbstractArray{Int}, conditionals; clear_cache = false, checkpoint_func = wrap, sdpa_func = sdpa, conditional_list = 1:ifelse(conditionals isa Tuple, length(conditionals), 1)) = model(tokens, conditionals, nothing; clear_cache, checkpoint_func, sdpa_func, conditional_list)

(model::ConditionalTransformer)(tokens::AbstractArray{Int}, conditional, opt_state; clear_cache = false, checkpoint_func = wrap, sdpa_func = sdpa, conditional_list = 1:1) = model(tokens, (conditional,), opt_state; clear_cache, checkpoint_func, sdpa_func, conditional_list)

(model::ConditionalTransformer)(tokens::AbstractArray{Int}; clear_cache = false, checkpoint_func = wrap, sdpa_func = sdpa) = model(tokens, (); clear_cache, checkpoint_func, sdpa_func)

# compat
forward_inference(model, args...) = model(args...)
forward_loss(model::ConditionalTransformer, inputs::AbstractArray, conditionals, targets::AbstractArray; clear_cache = true, loss_mask = nothing) = loss(model(inputs, conditionals, clear_cache = clear_cache), targets, loss_mask = loss_mask) 

### Sampling ###

function nexttoken!(tokens, model, sampler, logits, tokenizer_for_printing)
    tokens[model.pos+1] = sampler(logits[:, end, 1])
    !isnothing(tokenizer_for_printing) && print(decode(tokenizer_for_printing, [tokens[model.pos+1]], skip_special_tokens = false))
end

"""
    generate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), tokenizer_for_printing=tkn, end_token=128010)

Takes an initial sequence of tokens, and generates new tokens one at a time until the end token is sampled. Uses a KV cache. No batch dim for now.
Runs on CPU by default. If the model is on the GPU (assuming Flux.jl, eg. `model = gpu(model)`), then pass `device = gpu` to `generate` to run on the GPU.

```julia
tkn = llama3_tokenizer()
generate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), tokenizer_for_printing=tkn, end_token=128010)
```
"""
function generate(
    model::ConditionalTransformer, 
    initial_tokens::AbstractArray{<:Integer},
    conditionals;
    max_new_tokens=100,
    sampler::Function=argmax_sampler,
    tokenizer_for_printing = nothing,
    end_token = 128010,
    clear_cache = true,
    pos_offset = 0,
    device = identity,
    sdpa_func = sdpa,
    kvcaching = true
)
    tokens = vcat(initial_tokens, similar(initial_tokens, max_new_tokens))
    if clear_cache
        clear_cache!(model)
        config_cache!(model, ifelse(kvcaching, length(initial_tokens) + max_new_tokens, 0))
    else
        extend_cache!(model, ifelse(kvcaching, length(initial_tokens) + max_new_tokens, 0))
    end
    input_tokens = device(reshape(initial_tokens, :, 1))  # (seq_len, batch=1)
    conditionals = device(conditionals)
    logits = model(input_tokens, conditionals, sdpa_func = sdpa_func)
    if max_new_tokens > 0
        nexttoken!(tokens, model, sampler, logits, tokenizer_for_printing)
        tokens[model.pos+1] == end_token && return tokens[1:model.pos+1]
    else
        return tokens
    end
    for _ in 1:max_new_tokens-1
        input_tokens = device(reshape(tokens[ifelse(kvcaching, model.pos+1:model.pos+1, 1:model.pos+1)], :, 1))  # Just the last token
        logits = model(input_tokens, conditionals, sdpa_func = sdpa_func)
        nexttoken!(tokens, model, sampler, logits, tokenizer_for_printing)
        tokens[model.pos+1] == end_token && break
    end
    return tokens[1:model.pos+1]
end