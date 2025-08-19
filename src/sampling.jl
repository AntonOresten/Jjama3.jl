function nexttoken!(tokens, pos, model, sampler, logits, tokenizer_for_printing)
    tokens[pos+1:pos+1] .= argmax_sampler(logits[:, end, 1])
    !isnothing(tokenizer_for_printing) && print(decode(tokenizer_for_printing, tokens[pos+1:pos+1] |> cpu, skip_special_tokens = false))
end

function handle_token(tokenizer, token, end_token)
    token = Array(token)
    !isnothing(tokenizer) && print(decode(tokenizer, token, skip_special_tokens = false))
    return only(token) == end_token
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
    model::Transformer{T},
    initial_tokens::AbstractArray{<:Integer};
    max_new_tokens=100,
    sampler = x -> logitsample(x, dims=1),
    tokenizer_for_printing = nothing,
    end_token = 128010,
    caches=kv_cache(model, 1024, 1),
    kws...
) where T
    n = size(initial_tokens, 1)
    tokens = initial_tokens
    n > 1 && model(tokens[1:n-1, :]; causal=true, caches, kws...)
    for i in 1:max_new_tokens
        logits = model(tokens[end:end, :]; caches, kws...)
        sampled_token = sampler(logits[:, end])
        tokens = [tokens; sampled_token]
        handle_token(tokenizer_for_printing, sampled_token, end_token) && break
    end
    return tokens
end
