import os
import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# ============ CONFIGURATION ============
GEN_MODEL_ID = "microsoft/phi-2"  # Phi-2 model
PROMPT_DIR = "generated_prompts"
RESPONSE_DIR = "generated_responses_phi2"
os.makedirs(RESPONSE_DIR, exist_ok=True)

# ============ DEVICE SETUP ============
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ============ LOAD MODEL ============
print(f"🔄 Loading Phi-2 model on {device.upper()}...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None
)
print(f"✅ Model loaded using {device.upper()}")

# ============ GENERATION FUNCTION ============
def generate_response(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
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

# ============ PROCESS EACH FILE ============
for file in os.listdir(PROMPT_DIR):
    if file.endswith(".csv"):
        prompt_path = os.path.join(PROMPT_DIR, file)
        response_path = os.path.join(RESPONSE_DIR, file)

        df = pd.read_csv(prompt_path)
        prompts = df["Prompt"].dropna().tolist()

        responses = []
        for prompt in tqdm(prompts, desc=f"Generating for {file}"):
            response = generate_response(prompt)
            responses.append({"Prompt": prompt, "Response": response})

        # Save to CSV
        output_df = pd.DataFrame(responses)
        output_df.to_csv(response_path, index=False)
        print(f"📁 Saved: {response_path}")
