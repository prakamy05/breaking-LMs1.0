
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# ============ CONFIGURATION ============
GEN_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
PROMPT_DIR = "generated_prompts_with_catalysts"
RESPONSE_DIR = "generated_responses_with_catalysts"
os.makedirs(RESPONSE_DIR, exist_ok=True)

# ============ LOAD MODEL ============
print("Loading LLaMA 2 model for catalytic response generation...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ============ RESPONSE GENERATION ============
def generate_response(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# ============ PROCESS FILES ============
for file in os.listdir(PROMPT_DIR):
    if file.endswith(".csv"):
        prompt_path = os.path.join(PROMPT_DIR, file)
        response_path = os.path.join(RESPONSE_DIR, file)

        df = pd.read_csv(prompt_path)
        prompts = df["Prompt"].dropna().tolist()

        responses = []
        for prompt in tqdm(prompts, desc=f"Generating responses for {file}"):
            response = generate_response(prompt)
            responses.append({"Prompt": prompt, "Response": response})

        pd.DataFrame(responses).to_csv(response_path, index=False)
        print(f"✅ Saved: {response_path}")
