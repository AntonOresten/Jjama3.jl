using Jjama3

using JSON3
import HuggingFaceTokenizers

dir = "models/Qwen3-0.6B"#ARGS[1]

config = JSON3.read(read(joinpath(dir, "config.json"), String));
model = Jjama3.load_qwen3_from_safetensors(filter(endswith(".safetensors"), readdir(dir, join=true)), config);
tkn = HuggingFaceTokenizers.from_file(HuggingFaceTokenizers.Tokenizer, joinpath(dir, "tokenizer.json"));

using CUDA, Flux

model = model |> gpu;
prompt = Jjama3.qwen3_assistant_prompt(tkn,"Tell me the two worst things about Python.") |> gpu;

CUDA.@time generate(model, prompt,
    max_new_tokens=300,
    tokenizer_for_printing=tkn,
    end_token = encode(tkn, "<|im_end|>")[end]);
