function sample_tokens(logits, transform)
    new_logits = transform(logits)
    indices = logitsample(new_logits, dims=1)
    tokens = first.(Tuple.(indices)) .- 1
    log_probs = Flux.logsoftmax(new_logits, dims=1)
    confidence = log_probs[indices]
    return vec(confidence), vec(tokens)
end

function diffusion_generate(
    model, tokenizer, input;
    mask_token_id = Tokenizers.encode(tokenizer, "<M>").ids[1],
    max_new_tokens = 150,
    steps = 256,
    kws...
)
    input_ids = Tokenizers.encode(tokenizer, input).ids
    x = [input_ids; fill(mask_token_id, max_new_tokens)]

    fix_mask = x .!= mask_token_id

    for i in 1:steps
        mask_index = x .== mask_token_id
        any(mask_index) || break

        # is_causal=false is crucial for bidirectional attention
        logits = model(x .+ 1; causal=false) |> Flux.cpu
        
        # CRITICAL: Shift logits to predict the next token, aligning with training
        logits = [logits[:, 1:1];; logits[:, 1:end-1]]

        # p2 algorithm
        kappa_t = (i + 1) / steps

        # Compute confidence and sampled tokens for the entire sequence
        conf_full, x0_full = sample_tokens(logits, Top_nσ(2))

        # Construct full_conf matrix and mask out fixed positions
        # Only positions in (~fix_mask) are candidates for masking/unmasking
        full_conf = copy(conf_full)
        full_conf[fix_mask] .= Inf

        # Calculate how many positions to re-mask
        num_positions = sum(.!fix_mask)
        num_to_mask = floor(Int, num_positions * (1 - kappa_t))
        num_to_mask = clamp(num_to_mask, 0, num_positions)

        # Select the lowest-confidence positions for re-masking
        sorted_idx = sortperm(full_conf)

        if num_to_mask > 0
            topk_idx = sorted_idx[1:num_to_mask]
            to_mask = zeros(Bool, size(x))
            to_mask[topk_idx] .= true
        else
            to_mask = zeros(Bool, size(x))
        end

        # Apply re-masking: set selected positions back to mask_token_id
        x[to_mask] .= mask_token_id

        # For positions that started as mask and were not re-masked, unmask them with sampled tokens
        keep_unmask = mask_index .& .!to_mask
        x[keep_unmask] .= x0_full[keep_unmask]

        y = copy(x)
        y[y .== mask_token_id] .= 93
        println(Tokenizers.decode(tokenizer, y))
    end
    
    return x
end
