chatml_prompt(role) = "<|im_start|>$role\n"
chatml_message(role, content) = chatml_prompt(role) * "$content<|im_end|>\n"

function chatml_instruct_prompt(system, user; include_system=true)
    (include_system ? chatml_message("system", system) : "") *
    chatml_message("user", user) *
    chatml_prompt("assistant")
end

llama3_prompt(role) = "<|start_header_id|>$role<|end_header_id|>\n"
llama3_message(role, content) = llama3_prompt(role) * "$content\n<|eot_id|>\n"

function llama3_instruct_prompt(system, user)
    llama3_message("system", system) *
    llama3_message("user", user) *
    llama3_prompt("assistant")
end

const DEFAULT_INSTRUCT_PROMPT_FUNCTION = chatml_instruct_prompt
const INSTRUCT_PROMPT_FUNCTIONS = Dict(
    "llama3" => llama3_instruct_prompt,
    "monad" => (args...) -> chatml_instruct_prompt(args...; include_system=false)*"<think>\n"
)

instruct_prompt_function(name) = get(INSTRUCT_PROMPT_FUNCTIONS, name, DEFAULT_INSTRUCT_PROMPT_FUNCTION)

const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
const SYSTEM_PROMPTS = Dict(
    "smollm" => "You are a helpful AI assistant named SmolLM, trained by Hugging Face.",
)

default_assistant_prompt(name) = get(SYSTEM_PROMPTS, name, DEFAULT_SYSTEM_PROMPT)

assistant_prompt_function(name) =
    input -> instruct_prompt_function(name)(default_assistant_prompt(name), input)
