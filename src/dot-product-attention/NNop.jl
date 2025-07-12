using NNop: NNop
using GPUArraysCore: AbstractGPUArray

function flash_attention(
    q::AbstractGPUArray, k::AbstractGPUArray, v::AbstractGPUArray;
    mask=nothing,
)
    causal = (mask === :causal)
    if causal || isnothing(mask)
        return NNop.flash_attention(q, k, v; causal)
    else
        @debug "mask of type $(typeof(mask)) not compatible with `NNop.flash_attention`"
        return sdpa(q, k, v; mask)
    end
end

function flash_attention(
    q::AbstractArray, k::AbstractArray, v::AbstractArray;
    mask=nothing,
)
    @debug "`NNop.flash_attention` only implemented for GPU arrays"
    return sdpa(q, k, v; mask)
end
