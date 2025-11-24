using ConcreteStructs
using Einops
using Flux: Flux, Dense, Embedding
using Onion: Onion, Layer, RMSNorm, RoPE
import Optimisers: trainable


const ACTIVATIONS = Dict("silu" => Flux.swish)
get_activation(name::String) = ACTIVATIONS[name]
get_activation(act::Function) = act

model_type(cfg) = Symbol(get(cfg, :model_type, "unknown"))

const HAS_QK_NORM = Set([:qwen3])
has_qk_norm(cfg) = model_type(cfg) in HAS_QK_NORM


@concrete struct Attention <: Layer
    q_proj; k_proj; v_proj; o_proj
    q_norm; k_norm
    head_dim
end

function Attention(;
    hidden_size::Int,
    num_attention_heads::Int,
    num_key_value_heads::Int,
    head_dim::Int,
    attention_bias::Bool = true,
    cfg...
)
    qk_norm = has_qk_norm(cfg)
    return Attention(
        Dense(hidden_size => num_attention_heads * head_dim; bias=attention_bias),
        Dense(hidden_size => num_key_value_heads * head_dim; bias=attention_bias),
        Dense(hidden_size => num_key_value_heads * head_dim; bias=attention_bias),
        Dense(num_attention_heads * head_dim => hidden_size; bias=false),
        qk_norm ? RMSNorm(head_dim; eps=cfg[:rms_norm_eps]) : identity,
        qk_norm ? RMSNorm(head_dim; eps=cfg[:rms_norm_eps]) : identity,
        head_dim)
end

function (layer::Attention)(
    x::AbstractArray, attention = Onion.Ops.sdpa;
    rope = identity, cache = tuple, kws...
)
    q, k, v = layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)
    q, k, v = rearrange.((q, k, v), einops"(d h) l ... -> d l h ..."; d=layer.head_dim)
    q, k = layer.q_norm(q), layer.k_norm(k)
    q, k = rope.((q, k))
    k, v = cache(k, v)
    y = attention(q, k, v; kws...)
    y = rearrange(y, einops"d l h ... -> (d h) l ...")
    o = layer.o_proj(y)
    return o
end


@concrete struct StarGLU <: Layer
    up_proj; gate_proj; down_proj; act
end

function StarGLU(; cfg...)
    hidden_size = cfg[:hidden_size]
    intermediate_size = cfg[:intermediate_size]
    hidden_act = get_activation(cfg[:hidden_act])
    return StarGLU(
        Dense(hidden_size => intermediate_size, bias=false),
        Dense(hidden_size => intermediate_size, bias=false),
        Dense(intermediate_size => hidden_size, bias=false),
        hidden_act)
end

function (layer::StarGLU{W,W,W})(
    x, chunk_size::Int = 1024
) where W<:Dense{typeof(identity),<:DenseMatrix,Bool}
    chunk_size > 0 || throw(ArgumentError("Chunk size must be greater than 0."))
    weights = (;
        up=layer.up_proj.weight, gate=layer.gate_proj.weight,
        down=layer.down_proj.weight, act=layer.act)
    return Onion.Ops.star_glu(weights, x; chunk_size)
end

function (layer::StarGLU)(x, chunked::Bool = false)
    !chunked || throw(ArgumentError("For chunked StarGLU, pass the chunk size as an `Int`."))
    return layer.down_proj(layer.act.(layer.gate_proj(x)) .* layer.up_proj(x))
end


@concrete struct Block <: Layer
    prenorm1; attention
    prenorm2; feedforward
end

function Block(; cfg...)
    return Block(
        RMSNorm(cfg[:hidden_size]; eps=cfg[:rms_norm_eps]), Attention(; cfg...),
        RMSNorm(cfg[:hidden_size]; eps=cfg[:rms_norm_eps]), StarGLU(; cfg...))
end

function (layer::Block)(x; attention=Onion.Ops.sdpa, kws...)
    x = x + layer.attention(layer.prenorm1(x), attention; kws...)
    x = x + layer.feedforward(layer.prenorm2(x))
    return x
end


struct Model <: Layer
    embed_tokens; rope; blocks; norm; lm_head; config
end

trainable(m::Model) = (; m.embed_tokens, m.rope, m.blocks, m.norm, m.lm_head)

function Model(; cfg...)
    cfg = (; head_dim=cfg[:hidden_size] ÷ cfg[:num_attention_heads], cfg...)
    return Model(
        Embedding(cfg[:vocab_size] => cfg[:hidden_size]),
        RoPE(cfg[:head_dim], cfg[:max_position_embeddings]; theta=cfg[:rope_theta]),
            # TODO: fix scaling
            # use_scaled=!isnothing(cfg[:rope_scaling]), scale_factor=cfg[:rope_scaling]),
        ntuple(
            index -> Block(; index, cfg...),
            cfg[:num_hidden_layers]),
        RMSNorm(cfg[:hidden_size]; eps=cfg[:rms_norm_eps]),
        Dense(cfg[:hidden_size] => cfg[:vocab_size], bias=false),
        cfg)
end

function (model::Model)(tokens; cache=no_kv_cache(model), pos=0, kws...)
    x = model.embed_tokens(tokens)
    rope = model.rope[pos .+ axes(x, 2)]
    for (block, cache) in zip(model.blocks, cache)
        x = block(x; rope, cache, kws...)
    end
    return model.lm_head(model.norm(x))
end
