module Jjama3

using ConcreteStructs
using Einops
using Flux
using SafeTensors
using LinearAlgebra
using NNlib
using LogitSamplers
using LowRankLayers
using ChainRulesCore

include("cache.jl")
export KVCache

include("layers.jl")
export FeedForward
export RMSNorm
export RoPE
export Attention
export TransformerBlock
export Transformer
export unrope
export rerope_cache!
export scrape_cache
export append_cache!

include("dot-product-attention/dpa.jl")

include("model.jl")
export forward_loss
export forward_inference
export loss

include("sampling.jl")
export top_pk_sampler
export argmax_sampler
export top_nσ_sampler
export min_p_sampler
export generate

include("utils.jl")
export encode
export decode
export load_llama321B_from_safetensors
export load_llama3_from_safetensors
export llama3_instruct_prompt
export llama3_assistant_prompt
export smollm2_instruct_prompt
export smollm2_assistant_prompt
export structured_choice
export pad_and_batch
export export_model

end
