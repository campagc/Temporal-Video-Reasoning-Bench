import os
import json
import random
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import itertools

# --- CONFIGURAZIONE ---
MANIFEST_PATH = "data/processed_frames/dataset_manifest.json"
OUTPUT_DIR = "output"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_CLIPS_SUBSET = 5 # Lavoriamo su un sottoinsieme di 5 clip
# ----------------------

def main():
    print(f"--- FASE 2: INFERENZA A COPPIE (Pairwise) ---")
    
    # 1. Carica e Seleziona Subset
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERRORE: Non trovo {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r") as f:
        ground_truth_steps = json.load(f)
    
    # Ordiniamoli per ID temporale reale
    ground_truth_steps.sort(key=lambda x: x['step_id'])

    # Prendiamo 5 step casuali
    if len(ground_truth_steps) > NUM_CLIPS_SUBSET:
        selected_steps = random.sample(ground_truth_steps, NUM_CLIPS_SUBSET)
    else:
        selected_steps = ground_truth_steps
    
    # Assegniamo ID fittizi (Clip A, Clip B...)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G'][:len(selected_steps)]
    clip_map = {}
    
    print("Clip selezionate per il torneo:")
    for idx, step in enumerate(selected_steps):
        lid = letters[idx]
        clip_map[lid] = step
        print(f"ID: {lid} -> Step Reale: {step['step_id']} ({step['label']})")

    # 2. Generiamo le coppie
    pairs = list(itertools.combinations(letters, 2))
    print(f"\nGenereremo {len(pairs)} confronti a coppie.")

    # 3. Carica Modello (CORRETTO PER EVITARE ERRORE FLASH ATTENTION)
    print("Caricamento modello...")
    try:
        # Proviamo prima con Flash Attention (se c'è)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"Flash Attention non disponibile ({e}), uso attenzione standard.")
        # Fallback standard
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    results = []

    # 4. Loop sulle coppie
    for id1, id2 in pairs:
        print(f"\nConfronto {id1} vs {id2}...", end=" ")
        
        step1 = clip_map[id1]
        step2 = clip_map[id2]
        
        # Mischiamo l'ordine di presentazione
        show_order = [id1, id2]
        random.shuffle(show_order)
        
        first_shown_id = show_order[0]
        second_shown_id = show_order[1]
        
        content_list = []
        
        # Prompt Visivo
        text_prompt = (
            "You are a video editor. Look at these two video clips from the same activity.\n"
            f"--- Clip {first_shown_id} ---\n"
        )
        
        for f in clip_map[first_shown_id]['frames']:
            content_list.append({"type": "image", "image": f})
        
        text_prompt += f"\n--- Clip {second_shown_id} ---\n"
        
        for f in clip_map[second_shown_id]['frames']:
            content_list.append({"type": "image", "image": f})

        text_prompt += (
            "\nTASK: Which of these two actions happens FIRST in the procedure?\n"
            "Think about the logic (e.g., you must peel before cutting).\n"
            f"Output ONLY the ID of the clip that happens first: '{first_shown_id}' or '{second_shown_id}'."
        )

        content_list.append({"type": "text", "text": text_prompt})
        messages = [{"role": "user", "content": content_list}]

        # Inference
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=16, 
                temperature=0.1,  # Bassa temperatura
                do_sample=True    # Necessario se usi temperature
            )
        
        output = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parsing risposta
        response = output.split(text_prompt)[-1].strip()
        
        predicted_first = None
        # Logica robusta per capire chi ha vinto
        if id1 in response and id2 not in response: predicted_first = id1
        elif id2 in response and id1 not in response: predicted_first = id2
        else: 
            # Se il modello scrive "Clip A comes before Clip B"
            pos1 = response.find(id1)
            pos2 = response.find(id2)
            if pos1 != -1 and pos2 == -1: predicted_first = id1
            elif pos2 != -1 and pos1 == -1: predicted_first = id2
            elif pos1 != -1 and pos2 != -1: 
                # Se li cita entrambi, speriamo il primo citato sia la risposta (non sempre vero ma ok per test)
                predicted_first = id1 if pos1 < pos2 else id2
            else:
                predicted_first = "UNKNOWN"

        print(f"-> Vince {predicted_first}")

        results.append({
            "pair": [id1, id2],
            "shown_order": [first_shown_id, second_shown_id],
            "model_response": response,
            "predicted_winner": predicted_first,
            "truth_winner": id1 if step1['step_id'] < step2['step_id'] else id2
        })

    # 5. Calcolo Score
    correct = 0
    valid = 0
    for r in results:
        if r['predicted_winner'] != "UNKNOWN":
            valid += 1
            if r['predicted_winner'] == r['truth_winner']:
                correct += 1
    
    acc = (correct / valid * 100) if valid > 0 else 0
    print(f"\n--- RISULTATI ---")
    print(f"Accuracy Pairwise: {correct}/{valid} ({acc:.1f}%)")

    # 6. Salva
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "result_pairwise.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_PATH,
            "accuracy": acc,
            "results": results
        }, f, indent=4)
    print(f"Salvato in {out_path}")

if __name__ == "__main__":
    main()