using Functors

function fmap_fields(f::Function, model; fields)
    partial_paths = map(x -> x isa Symbol ? (x,) : x, fields)

    function pred(kp)
        isempty(kp) && return false
        t = Tuple(kp)
        any(partial_paths) do pp
            n = length(pp)
            length(t) >= n && t[end-n+1:end] == pp
        end
    end

    exclude(kp, x) = Functors.isleaf(x) || pred(kp)

    fmap_with_path(model; exclude) do kp, x
        pred(kp) ? f(x) : x
    end
end


# TODO: chunked forward with custom backward like we do with StarGLU in Onion
@concrete struct LoRADense <: Layer
    primary
    lora_A
    lora_B
end

trainable(layer::LoRADense) = (; layer.lora_A, layer.lora_B)

function LoRADense(primary::Dense, hidden_dim::Int; init=Flux.kaiming_uniform)
    dim2, dim1 = size(primary.weight)
    layer = LoRADense(
        primary,
        Dense(dim1 => hidden_dim; bias=false, init),
        Dense(hidden_dim => dim2; bias=false)
    )
    layer.lora_B.weight .= 0
    return layer
end

(layer::LoRADense)(x) = layer.primary(x) .+ layer.lora_B(layer.lora_A(x))

islora(layer) = layer isa LoRADense
function load_lora_tensors!(layer::LoRADense, lora_A_weight, lora_B_weight)
    layer.lora_A.weight .= lora_A_weight
    layer.lora_B.weight .= lora_B_weight
end
 

# NOTE: this doesn't automatically freeze other parameters
function add_lora_to(
    model::Model,
    dim::Int,
    fields;
    freeze_non_lora=true
)
    model_lora = fmap_fields(model; fields) do layer
        @assert layer isa Dense
        LoRADense(layer, dim)
    end
    model_lora = fmap_fields(model_lora; fields=[:config]) do config
        merge(config, (; adapter_config=(; r=dim, target_modules=String.(fields))))
    end
    if freeze_non_lora
        model_lora = lora_only_trainable(model_lora)
    end
    return model_lora
end

function load_lora_tensors!(
    model::Model, tensors;
    prefix="base_model.model."
)
    for (i, block) in enumerate(model.blocks)
        layer = "$(prefix)model.layers.$(i-1)"
        set! = (l, key) -> if islora(l)
            load_lora_tensors!(l, tensors["$layer.$key.lora_A.weight"], tensors["$layer.$key.lora_B.weight"])
        end
        for key in [:q_proj, :k_proj, :v_proj, :o_proj]
            set!(getproperty(block.attention, key), "self_attn.$key")
        end
        for key in [:up_proj, :gate_proj, :down_proj]
            set!(getproperty(block.feedforward, key), "mlp.$key")
        end
    end
end

function load_lora(model::Model, path::AbstractString; kws...)
    config = load_config(joinpath(path, "adapter_config.json"))
    tensors = load_safetensors(joinpath(path, "adapter_model.safetensors"))
    model_lora = add_lora_to(model, config.r, Symbol.(config.target_modules); kws...)
    load_lora_tensors!(model_lora, tensors)
    return model_lora
end

function save_lora(
    model::Model, path::AbstractString;
    adapter_config=get(model.config, :adapter_config, nothing),
    prefix="base_model.model."
)
    @assert isdir(path)
    @assert !isnothing(adapter_config)
    JSON.json(joinpath(path, "adapter_config.json"), adapter_config, pretty=true)
    tensors = Dict{String,AbstractArray}()
    for (i, block) in enumerate(model.blocks)
        layer = "$(prefix)model.layers.$(i-1)"
        set! = (key, l) -> if islora(l)
            tensors["$layer.$(key).lora_A.weight"] = l.lora_A.weight
            tensors["$layer.$(key).lora_B.weight"] = l.lora_B.weight
        end
        for key in [:q_proj, :k_proj, :v_proj, :o_proj]
            set!("self_attn.$key", getproperty(block.attention, key))
        end
        for key in [:up_proj, :gate_proj, :down_proj]
            set!("mlp.$key", getproperty(block.feedforward, key))
        end
    end
    SafeTensors.serialize(joinpath(path, "adapter_model.safetensors"), tensors)
    return nothing
end


has_direct_params(x) = any(v -> v isa AbstractArray{<:AbstractFloat}, trainable(x))

function contains_lora(x)
    exclude(z) = Functors.isleaf(z) || z isa LoRADense
    any(z -> z isa LoRADense, Functors.fleaves(x; exclude))
end


using Onion: Untrainable

function lora_only_trainable(model)
    function exclude(x)
        Functors.isleaf(x) && return true
        islora(x) && return true
        has_direct_params(x) && !contains_lora(x) && return true
        return false
    end
    fmap(model; exclude) do x
        if x isa LoRADense
            x
        elseif has_direct_params(x)
            if contains_lora(x)
                @warn "direct params + LoRA descendants; direct params won't be frozen"
                x
            elseif !contains_lora(x)
                Untrainable(x)
            end
        else
            x
        end
    end
end
