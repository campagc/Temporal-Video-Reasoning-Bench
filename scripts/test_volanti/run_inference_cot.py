import os
import json
import random
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURAZIONE ---
MANIFEST_PATH = "data/processed_frames/dataset_manifest.json"
OUTPUT_DIR = "output"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_TEST_CLIPS = 3  # Teniamo 3 per ora, se funziona alzi a 5
# ----------------------

def main():
    print(f"--- FASE 2: INFERENZA CHAIN-OF-THOUGHT (CoT) ---")
    
    # 1. Carica e Seleziona Dati
    with open(MANIFEST_PATH, "r") as f:
        ground_truth_steps = json.load(f)
    
    if len(ground_truth_steps) > NUM_TEST_CLIPS:
        subset_steps = random.sample(ground_truth_steps, NUM_TEST_CLIPS)
    else:
        subset_steps = ground_truth_steps
    
    # Shuffle
    shuffled_steps = subset_steps.copy()
    random.seed(42)
    random.shuffle(shuffled_steps)
    
    shuffled_mapping = []
    print("--- INPUT ---")
    for idx, step in enumerate(shuffled_steps):
        clip_label = f"Clip {idx+1}"
        print(f"{clip_label} -> Realtà: {step['label']} (ID: {step['step_id']})")
        shuffled_mapping.append({
            "clip_label": clip_label,
            "frames": step['frames'],
            "original_id": step['step_id']
        })

    # 2. Carica Modello
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
    except:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
        )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 3. Costruzione Prompt CoT (Step-by-Step)
    content_list = []
    
    # Introduzione
    content_list.append({"type": "text", "text": "You are a video analyst helping to reconstruct a recipe timeline.\n"})
    content_list.append({"type": "text", "text": f"Here are {len(shuffled_mapping)} unordered video clips.\n\n"})

    # Inseriamo le clip
    for item in shuffled_mapping:
        content_list.append({"type": "text", "text": f"--- {item['clip_label']} ---\n"})
        for frame_path in item['frames']:
            content_list.append({"type": "image", "image": frame_path})
        content_list.append({"type": "text", "text": "\n"})

    # LA PARTE IMPORTANTE: Le istruzioni "Obbligatorie"
    prompt_text = (
        "TASK:\n"
        "Step 1: Analyze each clip individually. Describe exactly what action is happening (e.g. 'cutting carrots', 'washing hands').\n"
        "Step 2: Determine logical dependencies (e.g. you must cut before cooking).\n"
        "Step 3: Provide the correct chronological order.\n\n"
        "OUTPUT FORMAT (Strictly follow this):\n"
        "Analysis Clip 1: [Description]\n"
        "Analysis Clip 2: [Description]\n"
        "... (for all clips)\n"
        "Reasoning: [Why you chose this order]\n"
        "Final Sequence: Clip X -> Clip Y -> Clip Z"
    )
    content_list.append({"type": "text", "text": prompt_text})

    # 4. Inferenza
    messages = [{"role": "user", "content": content_list}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    print("\nGenerazione...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.1, # Bassa temperatura = Più logica
            do_sample=True
        )
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"\n--- RISPOSTA ---\n{output_text}\n----------------")

    # 5. Salvataggio
    result_data = {
        "model_name": MODEL_PATH,
        "input_map": shuffled_mapping,
        "raw_output": output_text,
        "ground_truth_ids": [s['step_id'] for s in subset_steps] # Questi sono ordinati per natura del subset? No, occhio.
        # Correggiamo: La GT è l'ordine degli ID dal più piccolo al più grande
    }
    # La GT vera è la lista degli ID ordinati
    gt_sorted = sorted([s['step_id'] for s in shuffled_steps])
    result_data["ground_truth_linear"] = gt_sorted

    with open(os.path.join(OUTPUT_DIR, "result_cot.json"), "w") as f:
        json.dump(result_data, f, indent=4)

if __name__ == "__main__":
    main()