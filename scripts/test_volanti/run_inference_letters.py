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
# ----------------------

def main():
    print(f"--- FASE 2: TEST LETTERE (Anti-Bias) ---")
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERRORE: {MANIFEST_PATH} non trovato.")
        return

    with open(MANIFEST_PATH, "r") as f:
        ground_truth_steps = json.load(f)
    
    # 1. PRENDIAMO SOLO 5 STEP (Subset)
    if len(ground_truth_steps) > 5:
        subset_steps = random.sample(ground_truth_steps, 5)
    else:
        subset_steps = ground_truth_steps
    
    # Mischiamo
    shuffled_steps = subset_steps.copy()
    random.seed(42)
    random.shuffle(shuffled_steps)
    
    # 2. ASSEGNIAMO LETTERE (A, B, C, D, E)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G'][:len(shuffled_steps)]
    shuffled_mapping = []
    
    print("Mappa data al modello:")
    for idx, step in enumerate(shuffled_steps):
        letter = letters[idx]
        print(f"Clip {letter} -> Step reale: {step['label']} (ID: {step['step_id']})")
        shuffled_mapping.append({
            "clip_id": letter,
            "original_step_id": step['step_id'],
            "frames": step['frames']
        })

    # 3. Carica Modello
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
    except:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
        )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 4. Prompt "Describe then Sort"
    content_list = []
    
    text_prompt = (
        "You are a video analyst. You are given 5 unordered video clips labeled A, B, C, D, E.\n"
        "Your goal is to arrange them in chronological order to form a coherent recipe procedure.\n\n"
    )
    
    for item in shuffled_mapping:
        letter = item['clip_id']
        text_prompt += f"--- Clip {letter} ---\n"
        for frame_path in item['frames']:
            content_list.append({"type": "image", "image": frame_path})
        text_prompt += "\n"

    text_prompt += (
        "TASK:\n"
        "1. For EACH clip (A, B, C, D, E), write one short sentence describing the action visible in the frames.\n"
        "2. Identify which action logically starts the process (e.g. taking ingredients) and which ends it (e.g. serving).\n"
        "3. Provide the final chronological sequence using the letters.\n\n"
        "OUTPUT FORMAT:\n"
        "Description A: [action]\n"
        "Description B: [action]\n"
        "... (for all clips)\n"
        "Reasoning: [short explanation]\n"
        "SEQUENCE: ClipX->ClipY->ClipZ->..."
    )
    
    content_list.append({"type": "text", "text": text_prompt})
    messages = [{"role": "user", "content": content_list}]

    # Pre-processing
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # 5. Generazione (con do_sample=True per fixare il warning)
    print("\nGenerazione...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.1, # Bassissima: vogliamo ragionamento logico, non creatività
            do_sample=True   # Necessario per usare la temperatura
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"\n--- RISPOSTA ---\n{output_text}\n----------------")

    # 6. Salvataggio
    result_data = {
        "model_name": MODEL_PATH,
        "shuffled_input_map": shuffled_mapping,
        "model_raw_output": output_text,
        "ground_truth_linear": sorted([s['step_id'] for s in shuffled_steps]) 
    }
    
    with open(os.path.join(OUTPUT_DIR, "result_letters.json"), "w") as f:
        json.dump(result_data, f, indent=4)

if __name__ == "__main__":
    main()