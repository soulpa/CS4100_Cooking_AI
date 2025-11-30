# 1. We will load a language model model from huggingface (Qwen 0.5B Instruct)
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"    # swap if you prefer another instruct model
LOAD_8BIT    = False                           # set True if you installed bitsandbytes and want 8-bit loading
DTYPE        = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ====== TODO ======
# Load model with AutoModelForCausalLM.from_pretrained() from huggingface with the above MODEL_NAME, LOAD_8BIT, DTYPE
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=DTYPE,
    trust_remote_code=True,
    attn_implementation="eager",
    **({"load_in_8bit": True} if LOAD_8BIT else {})
    )

# Generation configuration: use GenerationConfig to define the generation parameters
gen_cfg = GenerationConfig(
    max_new_tokens=100,
    temperature=0.3,
    do_sample=True
)
# ====== TODO ======

# ====== Helper function: Enforce two-line schema in the decoding ======
T_PATTERN = re.compile(r"Thought:\s*(.+)")
A_PATTERN = re.compile(r"Action:\s*(.+)")

def _postprocess_to_two_lines(text: str) -> str:
    """
    Extract the first 'Thought:' and 'Action:' lines from the model output.
    If the model drifts, fall back to a conservative default Action.
    """
    # Stop at first Observation if present (model shouldn't produce it, but just in case)
    text = text.split("\nObservation:")[0]
    # Keep only the assistant's new tokens (strip any trailing commentary)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    # Try to find explicit Thought/Action anywhere in the output
    thought = None
    action  = None
    for ln in lines:
        if thought is None:
            m = T_PATTERN.match(ln)
            if m:
                thought = m.group(1).strip()
                continue
        if action is None:
            m = A_PATTERN.match(ln)
            if m:
                action = m.group(1).strip()
                continue

    # Fallbacks if the model didn’t comply perfectly
    if thought is None:
        thought = "I should search for key facts related to the question."
    if action is None:
        # Default to a generic search; your controller will parse it.
        action = 'search[query="(auto) refine the user question", k=3]'

    return f"Thought: {thought}\nAction: {action}"
# ====== Helper function: Enforce two-line schema in the decoding ======



# 2. We define the LLM function. This will be plugged into the agent without changing the controller ---
def hf_llm(prompt: str) -> str:
    """
    Completes from your existing ReAct prompt and returns exactly two lines:
    'Thought: ...' and 'Action: ...'
    """
    # We add a strong instruction to the prompt to improve compliance with the format
    format_guard = (
        "\n\nIMPORTANT: Respond with EXACTLY two lines in this format:\n"
        "Thought: <one concise sentence>\n"
        "Action: <either search[query=\"...\"] or finish[answer=\"...\"]>\n"
        "Do NOT include Observation."
    )
    full_prompt = prompt + format_guard

    # ====== TODO ======
    #     Here, let's write the code to use language model to generate the response given the full_prompt
    #     First, we need to use the tokenizer to tokenize the prompt into pytorch tensors
    #     Second, we need to use model.generate() to generate the model response (which includes the Thought and Action)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, generation_config=gen_cfg)
    # ====== TODO ======


    # Slice off the prompt tokens to get only the completion
    completion = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return _postprocess_to_two_lines(completion)

# We will wire it into the agent system
LLM = hf_llm