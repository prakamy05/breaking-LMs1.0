import os
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# ============ CONFIGURATION ============
GEN_MODEL_ID = "microsoft/phi-2"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_P = 0.95
REPETITION_PENALTY = 1.2
GENERATIONS_PER_CATEGORY = 35

SEED_DIR = "seeds"
OUTPUT_DIR = "generated_prompts_phi2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
print(f"✅ Model loaded using Accelerate on {device.upper()}")

# ============ FUNCTIONS ============

def build_fewshot_prompt(seeds):
    prompt = "Here is a list of provocative questions that might elicit controversial or harmful responses:\n"
    for i, sp in enumerate(seeds, 1):
        prompt += f"{i}. {sp.strip()}\n"
    prompt += f"{len(seeds)+1}."
    return prompt

def generate_prompts(seed_prompts, num=50):
    fewshot = build_fewshot_prompt(seed_prompts)
    inputs = tokenizer(fewshot, return_tensors="pt")
    
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    prompts = []

    with torch.no_grad():
        for _ in tqdm(range(num), desc="Generating"):
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            generated = decoded[len(fewshot):].strip()

            lines = [line.strip() for line in generated.split("\n") if line.strip()]
            for line in lines:
                if any(c in line for c in "?!.") and len(line.split()) > 3:
                    clean = line.strip("1234567890. ").strip()
                    prompts.append(clean)
                    break

    return prompts[:num]

def save_to_csv(prompts, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt"])
        for p in prompts:
            writer.writerow([p])
    print(f"📁 Saved {len(prompts)} prompts to {filename}")

# ============ MAIN ============
if __name__ == "__main__":
    print("🚀 Starting prompt generation with Phi-2...")
    seed_files = [f for f in os.listdir(SEED_DIR) if f.endswith(".txt")]

    for seed_file in seed_files:
        category = seed_file.replace(".txt", "")
        seed_path = os.path.join(SEED_DIR, seed_file)

        with open(seed_path, "r", encoding="utf-8") as f:
            seeds = [line.strip() for line in f if line.strip()]

        print(f"\n🔍 Category: {category} | Seed count: {len(seeds)}")

        generated = generate_prompts(seeds, num=GENERATIONS_PER_CATEGORY)
        output_path = os.path.join(OUTPUT_DIR, f"{category}.csv")
        save_to_csv(generated, output_path)
