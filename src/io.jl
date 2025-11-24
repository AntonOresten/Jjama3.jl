using JSON
using SafeTensors
import HuggingFaceTokenizers as Tokenizers

# TODO: automatically map config to a canonical form:
# for example, gemma3:
#  - detect if model_type is gemma3
#  - add hidden_act = hidden_activation
#  gemma3 has sliding window vibes?

load_config(path) = NamedTuple(JSON.parsefile(path))

load_model(path::AbstractString) = load_model(joinpath(path, "config.json"), path)
load_model(config_path::AbstractString, path::AbstractString) = load_model(load_config(config_path), path)
load_model(config, path::AbstractString) = load_tensors!(Model(; config...), load_tensors(path))

function load_tensors(path)
    if isdir(path)
        if "model.safetensors.index.json" in readdir(path)
            load_sharded_safetensors(path)
        else
            load_safetensors(joinpath(path, "model.safetensors"))
        end
    else
        load_safetensors(path)
    end
end

# TODO: handle qkv_proj and gate_up_proj names as seen in Phi-3
function load_tensors!(model::Model, tensors)
    model.embed_tokens.weight .= transpose(tensors["model.embed_tokens.weight"])
    model.lm_head.weight .= tensors["model.embed_tokens.weight"]
    model.norm.weight .= tensors["model.norm.weight"]
    for i in 1:model.config.num_hidden_layers
        set! = (weight, key) -> (weight .= tensors["model.layers.$(i-1).$key"])
        block = model.blocks[i]
        set!(block.prenorm1.weight, "input_layernorm.weight")
        set!(block.attention.q_proj.weight, "self_attn.q_proj.weight")
        set!(block.attention.k_proj.weight, "self_attn.k_proj.weight")
        set!(block.attention.v_proj.weight, "self_attn.v_proj.weight")
        set!(block.attention.o_proj.weight, "self_attn.o_proj.weight")
        set!(block.prenorm2.weight, "post_attention_layernorm.weight")
        set!(block.feedforward.up_proj.weight, "mlp.up_proj.weight")
        set!(block.feedforward.gate_proj.weight, "mlp.gate_proj.weight")
        set!(block.feedforward.down_proj.weight, "mlp.down_proj.weight")
        if get(model.config, :attention_bias, true)
            set!(block.attention.q_proj.bias, "self_attn.q_proj.bias")
            set!(block.attention.k_proj.bias, "self_attn.k_proj.bias")
            set!(block.attention.v_proj.bias, "self_attn.v_proj.bias")
        end
        if has_qk_norm(model.config)
            set!(block.attention.q_norm.weight, "self_attn.q_norm.weight")
            set!(block.attention.k_norm.weight, "self_attn.k_norm.weight")
        end
    end
    return model
end

function load_tokenizer(path::AbstractString)
    tokenizer_path = isdir(path) ? joinpath(path, "tokenizer.json") : path
    return Tokenizers.from_file(Tokenizers.Tokenizer, tokenizer_path)
end


function save_model(output_path, model; type_convert=identity)
    weights = Dict{String,AbstractArray}()
    weights["model.embed_tokens.weight"] = type_convert(transpose(model.embed_tokens.weight))
    weights["lm_head.weight"] = type_convert(model.lm_head.weight)
    weights["model.norm.weight"] = type_convert(model.norm.weight)
    for (i, layer) in enumerate(model.layers)
        set! = (key, weight) -> (weights["model.layers.$(i-1).$key"] = type_convert(weight))
        set!("self_attn.q_proj.weight", layer.attention.q_proj.weight)
        set!("self_attn.k_proj.weight", layer.attention.k_proj.weight)
        set!("self_attn.v_proj.weight", layer.attention.v_proj.weight)
        set!("self_attn.o_proj.weight", layer.attention.o_proj.weight)
        if get(model.config, :attention_bias, true)
            set!("self_attn.q_proj.bias", layer.attention.q_proj.bias)
            set!("self_attn.k_proj.bias", layer.attention.k_proj.bias)
            set!("self_attn.v_proj.bias", layer.attention.v_proj.bias)
        end
        set!("mlp.gate_proj.weight", layer.feed_forward.up.weight)
        set!("mlp.down_proj.weight", layer.feed_forward.w2.weight)
        set!("mlp.up_proj.weight", layer.feed_forward.w3.weight)
        set!("input_layernorm.weight", layer.attention_norm.weight)
        set!("post_attention_layernorm.weight", layer.ffn_norm.weight)
    end
    SafeTensors.serialize(output_path, weights)
    return nothing
end
