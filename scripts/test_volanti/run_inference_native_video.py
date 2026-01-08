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
NUM_TEST_CLIPS = 11  # Proviamo con 5 clip
# ----------------------

def main():
    print(f"--- FASE 2: INFERENZA VIDEO CON ID CASUALI ---")
    
    # 1. Carica e Seleziona Dati
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERRORE: Non trovo {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r") as f:
        ground_truth_steps = json.load(f)
    
    # Subset (prendiamo un campione casuale se sono troppi)
    if len(ground_truth_steps) > NUM_TEST_CLIPS:
        # PER FARE RANDOMICO 
        subset_steps = random.sample(ground_truth_steps, NUM_TEST_CLIPS)
        # PER TESTARE IN MODO DETERMINISTICO
        # ricordarsi modificare numero se si vuole di diverso numero
        # subset_steps = [ground_truth_steps[i] for i in [12, 11, 15, 2, 8]]
    else:
        subset_steps = ground_truth_steps
    
    # Shuffle (Mischiamo l'ordine in cui li presentiamo)
    shuffled_steps = subset_steps.copy()
    random.seed(42)
    random.shuffle(shuffled_steps)
    
    # Generiamo ID casuali numerici (es. 104, 892) per evitare "1, 2, 3"
    random_ids = random.sample(range(100, 999), len(shuffled_steps))
    
    shuffled_mapping = []
    print("--- INPUT GENERATO ---")
    
    for idx, step in enumerate(shuffled_steps):
        assigned_id = str(random_ids[idx])
        clip_label = f"Clip {assigned_id}"
        
        print(f"{clip_label} -> Contenuto Reale: {step['label']} (ID Originale: {step['step_id']})")
        
        # Assicuriamoci che i path siano stringhe
        frame_paths = [str(p) for p in step['frames']]
        
        shuffled_mapping.append({
            "clip_label": clip_label,
            "clip_id_num": assigned_id,
            "frames": frame_paths,
            "original_id": step['step_id'],
            "real_label": step['label']
        })

    # 2. Carica Modello
    try:
        print("Caricamento modello con Flash Attention 2...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"Flash Attention non disponibile ({e}), uso standard.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
        )
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 3. Costruzione Prompt (FORMATO VIDEO)
    content_list = []
    
    content_list.append({"type": "text", "text": "You are a video analyst reconstructing a recipe timeline.\n"})
    content_list.append({"type": "text", "text": f"Here are {len(shuffled_mapping)} unordered video clips.\n\n"})

    for item in shuffled_mapping:
        content_list.append({"type": "text", "text": f"--- {item['clip_label']} ---\n"})
        
        # Inserimento Video Nativo
        content_list.append({
            "type": "video",
            "video": item['frames'], 
            "fps": 1.0, 
        })
        content_list.append({"type": "text", "text": "\n"})

    # Prompt Chain of Thought
    prompt_text = (
        "TASK:\n"
        "1. Analyze each clip individually. Describe exactly what action is happening (objects, hands, food).\n"
        "2. Note that the Clip IDs (e.g. 105, 802) are RANDOM and DO NOT indicate time.\n"
        "3. Identify logical dependencies (e.g. cleaning happens AFTER dirtying).\n"
        "4. Provide the chronological order.\n\n"
        "OUTPUT FORMAT:\n"
        "Analysis Clip [ID]: [Description]\n"
        "... (for all clips)\n"
        "Reasoning: [Explain the logical flow]\n"
        "Final Sequence: Clip [ID] -> Clip [ID] -> Clip [ID]"
    )
    content_list.append({"type": "text", "text": prompt_text})

    # 4. Processamento
    messages = [{"role": "user", "content": content_list}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Estrazione input video
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    # FIX PER ERRORE FPS (Rimuove fps dalla lista kwargs e lo passa esplicitamente)
    if "fps" in video_kwargs:
        _ = video_kwargs.pop("fps")

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        fps=1.0,  # Passato come float singolo
        **video_kwargs 
    ).to("cuda")

    print("\nGenerazione in corso...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1024, # Aumentato per permettere descrizioni lunghe
            temperature=0.1,
            do_sample=True
        )
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"\n--- RISPOSTA DEL MODELLO ---\n{output_text}\n----------------------------")

    # 5. Salvataggio Risultati
    
    # Calcoliamo la soluzione corretta per scriverla nel JSON (Cheat Sheet)
    # Ordiniamo gli elementi in base all'ID originale (temporale)
    correct_order = sorted(shuffled_mapping, key=lambda x: x['original_id'])
    cheat_sheet_string = " -> ".join([x['clip_label'] for x in correct_order])

    result_data = {
        "model_name": MODEL_PATH,
        "input_map": shuffled_mapping,
        "model_raw_output": output_text,
        "ground_truth_linear_ids": sorted([s['step_id'] for s in subset_steps]),
        "SOLUTION_CHEAT_SHEET": cheat_sheet_string
    }
    
    output_file = os.path.join(OUTPUT_DIR, "result_native_video.json")
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=4)
    
    print(f"Risultato salvato in: {output_file}")
    print(f"SOLUZIONE REALE (per tuo controllo): {cheat_sheet_string}")

if __name__ == "__main__":
    main()