include("cache.jl")
include("attention.jl")
include("prompts.jl")
include("dllm.jl")

using LogitSamplers

function autoregressive_inference(
    model, tokenizer;
    token, cache, pos,
    max_new_tokens = 100,
    transform = Top_nσ(2) ∘ Temperature(0.6),
    io = stdout,
)
    for i in 1:max_new_tokens
        logits = model(token .+ 1; cache=withposition.(cache, pos), pos)[:,end]
        token = logitsample(transform(logits), dims=1) .- 1
        pos += 1
        token_cpu = Array(token)
        only(token_cpu) == model.config.eos_token_id && break
        print(io, Tokenizers.decode(tokenizer, token_cpu))
    end
end

# take prompt template argument
function generate(
    model, tokenizer, input;
    template=get(model.config, :model_type, nothing),
    template_function=assistant_prompt_function(template),
    show_prompt=false,
    kws...
)
    prompt = template_function(input)
    show_prompt && println(prompt)
    tokens = Tokenizers.encode(tokenizer, prompt).ids
    cache = kv_cache(model, 4096)
    model(tokens[1:end-1] .+ 1; causal=true, cache=withposition.(cache, 0))
    autoregressive_inference(model, tokenizer; token=tokens[end:end], cache, pos=length(tokens)-1, kws...)
    println()
end
