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
    print(f"--- FASE 2: INFERENZA MULTI-SEQUENZA CON {MODEL_PATH} ---")
    
    # 1. Carica il Dataset
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERRORE: Non trovo {MANIFEST_PATH}.")
        return

    with open(MANIFEST_PATH, "r") as f:
        ground_truth_steps = json.load(f)
    
# ...
    # 2. Mischiamo gli step (Shuffle)
    # --- MODIFICA FONDAMENTALE: RIDUZIONE DATASET ---
    # Prendiamo solo 5 step casuali dalla lista totale
    if len(ground_truth_steps) > 5:
        print(f"⚠️ TEST RIDOTTO: Seleziono solo 3 clip su {len(ground_truth_steps)} per testare il ragionamento.")
        subset_steps = random.sample(ground_truth_steps, 3)
    else:
        subset_steps = ground_truth_steps
    
    shuffled_steps = subset_steps.copy()
    random.seed(42) 
    random.shuffle(shuffled_steps)
    # -----------------------------------------------
    
    shuffled_mapping = []
    # ... (continua col resto del codice uguale)
    
    print("Ordine mischiato dato al modello:")
    for idx, step in enumerate(shuffled_steps):
        print(f"Clip {idx+1} -> Step originale: {step['label']} (ID: {step['step_id']})")
        shuffled_mapping.append({
            "model_position_idx": idx + 1,
            "original_step_id": step['step_id'],
            "frames": step['frames'],
            "label": step['label'] # Ci serve per capire noi umani cosa sta succedendo
        })

    # 3. Carica il Modello
    print("\nCaricamento modello...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Flash Attention non disponibile, uso standard: {e}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 4. Preparazione Input (Prompt Avanzato)
    content_list = []
    
    ### NUOVO: Prompt aggiornato per chiedere ragionamento parallelo
    text_prompt = (
        f"You are analyzing {len(shuffled_mapping)} scrambled video clips of a recipe. "
        "The current order (1, 2, 3...) is COMPLETELY RANDOM and WRONG.\n\n"
    )
    
    for item in shuffled_mapping:
        clip_num = item['model_position_idx']
        text_prompt += f"Clip {clip_num}:\n"
        for frame_path in item['frames']:
            content_list.append({"type": "image", "image": frame_path})
        text_prompt += "\n"

    ### NUOVO: Istruzioni specifiche per sequenze multiple
    text_prompt += (
        "TASK:\n"
        "1. First, analyze the visual content of each clip to understand the action.\n"
        "2. Identify the logical start (e.g., preparing ingredients) and end (e.g., serving).\n"
        "3. Reconstruct the chronological timeline.\n\n"
        "OUTPUT FORMAT:\n"
        "Briefly describe the first step and the last step you found.\n"
        "Then, provide the sequence in this exact format:\n"
        "SEQUENCE: ClipID->ClipID->ClipID\n"
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
    )
    inputs = inputs.to("cuda")

    # 5. Generazione
    print("\nAvvio ragionamento e generazione...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,  # ### NUOVO: Aumentato per permettere output più lunghi (più sequenze)
            temperature=0.2,     # ### NUOVO: Bassa temperatura per essere precisi ma non robotici
            do_sample=True       # Abilita un minimo di variabilità
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"\n--- RISPOSTA DEL MODELLO ---\n{output_text}\n----------------------------")

    # 6. Salvataggio
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_data = {
        "model_name": MODEL_PATH,
        "prompt_type": "multi_sequence_reasoning",
        "shuffled_input_map": shuffled_mapping,
        "model_raw_output": output_text,
        # Salviamo la GT lineare originale come riferimento base
        "ground_truth_linear": [s['step_id'] for s in ground_truth_steps] 
    }
    
    save_path = os.path.join(OUTPUT_DIR, "inference_result_multi.json")
    with open(save_path, "w") as f:
        json.dump(result_data, f, indent=4)
        
    print(f"Risultato salvato in {save_path}")

if __name__ == "__main__":
    main()