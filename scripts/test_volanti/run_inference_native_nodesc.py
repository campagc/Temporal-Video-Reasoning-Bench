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
NUM_TEST_CLIPS = 5  # Proviamo con 5 clip per vedere se regge
# ----------------------

def main():
    print(f"--- FASE 2: INFERENZA DIRETTA (No CoT) ---")
    
    # 1. Carica Dataset
    with open(MANIFEST_PATH, "r") as f:
        ground_truth_steps = json.load(f)
    
    if len(ground_truth_steps) > NUM_TEST_CLIPS:
        subset_steps = random.sample(ground_truth_steps, NUM_TEST_CLIPS)
    else:
        subset_steps = ground_truth_steps
    
    # 2. Shuffle e ID Casuali (MANTENIAMO QUESTO TRUCCO)
    shuffled_steps = subset_steps.copy()
    random.seed(42)
    random.shuffle(shuffled_steps)
    
    shuffled_mapping = []
    # Generiamo ID casuali per evitare bias numerico
    random_ids = random.sample(range(100, 999), len(shuffled_steps))

    print("--- INPUT ---")
    for idx, step in enumerate(shuffled_steps):
        assigned_id = str(random_ids[idx])
        clip_label = f"Clip {assigned_id}"
        print(f"{clip_label} -> Realtà: {step['label']} (ID: {step['step_id']})")
        
        frame_paths = [str(p) for p in step['frames']]
        
        shuffled_mapping.append({
            "clip_label": clip_label,
            "clip_id_num": assigned_id,
            "frames": frame_paths,
            "original_id": step['step_id']
        })

    # 3. Carica Modello
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
    except:
        print("Flash Attention non disponibile.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
        )
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 4. Prompt DIRETTO (Senza descrizioni)
    content_list = []
    
    content_list.append({"type": "text", "text": "You are a video analyst reconstructing a recipe timeline.\n"})
    content_list.append({"type": "text", "text": f"Here are {len(shuffled_mapping)} unordered video clips.\n\n"})

    for item in shuffled_mapping:
        content_list.append({"type": "text", "text": f"--- {item['clip_label']} ---\n"})
        content_list.append({
            "type": "video",
            "video": item['frames'],
            "fps": 1.0, 
        })
        content_list.append({"type": "text", "text": "\n"})

    # --- MODIFICA PROMPT ---
    prompt_text = (
        "TASK: Reconstruct the correct chronological order of these clips based on visual logic.\n"
        "The Clip IDs (e.g. 402, 199) are RANDOM and have NO meaning.\n\n"
        "Do not provide descriptions or reasoning steps.\n"
        "Output ONLY the final sequence in this format:\n"
        "Final Sequence: Clip [ID] -> Clip [ID] -> Clip [ID]"
    )
    content_list.append({"type": "text", "text": prompt_text})

    # 5. Processamento
    messages = [{"role": "user", "content": content_list}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    # Fix FPS list error
    if "fps" in video_kwargs: _ = video_kwargs.pop("fps")

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        fps=1.0,
        **video_kwargs
    ).to("cuda")

    print("\nGenerazione (Diretta)...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128, # Meno token perché non deve descrivere
            temperature=0.1,
            do_sample=True
        )
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"\n--- RISPOSTA ---\n{output_text}\n----------------")

    # 6. Salvataggio e Soluzione
    result_data = {
        "model_name": MODEL_PATH,
        "mode": "direct_inference_no_cot",
        "input_map": shuffled_mapping,
        "raw_output": output_text,
        "ground_truth_linear": sorted([s['step_id'] for s in shuffled_steps]),
        "solution_cheat_sheet": " -> ".join([
            m['clip_label'] for m in sorted(shuffled_mapping, key=lambda x: x['original_id'])
        ])
    }
    
    print(f"Soluzione Reale: {result_data['solution_cheat_sheet']}")
    
    with open(os.path.join(OUTPUT_DIR, "result_direct.json"), "w") as f:
        json.dump(result_data, f, indent=4)

if __name__ == "__main__":
    main()