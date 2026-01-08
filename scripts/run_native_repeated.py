import os
import json
import random
import torch
import copy
import argparse
import sys
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURAZIONE ---
DEFAULT_NUM_TEST_CLIPS = 5 # Precisabile su Terminale come --num_clips=
DEFAULT_NUM_REPEATED_RUNS = 3 # Precisabile su Terminale come --num_runs=

MANIFEST_PATH = "data/processed_frames/dataset_manifest.json"
OUTPUT_DIR = "output"
OUTPUT_FILE = "results_repeated.json"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
# ----------------------

def main():
    # --- GESTIONE ARGOMENTI ---
    parser = argparse.ArgumentParser(description="Script per inferenza ripetuta su clip video.")
    
    parser.add_argument(
        "--num_clips", 
        type=int, 
        default=DEFAULT_NUM_TEST_CLIPS,
        help=f"Numero di clip nel subset da testare (Default: {DEFAULT_NUM_TEST_CLIPS})"
    )
    
    parser.add_argument(
        "--num_runs", 
        type=int, 
        default=DEFAULT_NUM_REPEATED_RUNS,
        help=f"Quante volte ripetere il test sullo stesso subset (Default: {DEFAULT_NUM_REPEATED_RUNS})"
    )

    args = parser.parse_args()

    NUM_TEST_CLIPS = args.num_clips
    NUM_REPEATED_RUNS = args.num_runs

    print(f"--- INFERENZA RIPETUTA ---")
    print(f"Configurazione: {NUM_TEST_CLIPS} clip | {NUM_REPEATED_RUNS} run totali")
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERRORE: Non trovo {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r") as f:
        ground_truth_steps = json.load(f)
    
    # --- SELEZIONE SUBSET (Fissa per tutte le run) ---

    # La subset di clip è fissa per tutte le run per poter confrontare i risultati
    if len(ground_truth_steps) > NUM_TEST_CLIPS:
        # seed fisso
        random.seed(999) 
        fixed_subset_steps = random.sample(ground_truth_steps, NUM_TEST_CLIPS)

    else:
        print(f"Attenzione: Il dataset ha solo {len(ground_truth_steps)} clip, meno delle {NUM_TEST_CLIPS} richieste. Le userò tutte.")
        fixed_subset_steps = ground_truth_steps

    print(f"Subset selezionato di {len(fixed_subset_steps)} clip.")
    print(f"Steps originali: {[s['step_id'] for s in fixed_subset_steps]}")

    # Carica Modello (Una volta sola per efficienza)
    try:
        print("Caricamento modello...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
    except:
        print("⚠️ Flash Attention 2 non disponibile o fallita, fallback su standard attention.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
        )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Lista per salvare tutti i risultati
    all_runs_results = []

    # --- CICLO DI RUN ---
    for run_idx in range(NUM_REPEATED_RUNS):
        print(f"\n>>> AVVIO RUN {run_idx + 1}/{NUM_REPEATED_RUNS}")
        
        # Resetta il seed per ogni run così da avere mischiamenti diversi di nomi di clip
        random.seed() 
        
        # Copia e mischia
        current_shuffled_steps = copy.deepcopy(fixed_subset_steps)
        random.shuffle(current_shuffled_steps)
        
        # Genera nuovi ID casuali per questa run
        random_ids = random.sample(range(100, 999), len(current_shuffled_steps))
        
        run_mapping = []
        
        for idx, step in enumerate(current_shuffled_steps):
            assigned_id = str(random_ids[idx])
            clip_label = f"Clip {assigned_id}"
            
            # Converti frame paths in stringhe
            frame_paths = [str(p) for p in step['frames']]
            
            run_mapping.append({
                "clip_label": clip_label,
                "clip_id_num": assigned_id,
                "frames": frame_paths,
                "original_id": step['step_id'],
                "real_label": step['label']
            })

        # Costruzione Prompt
        content_list = []
        content_list.append({"type": "text", "text": "You are a video analyst reconstructing a recipe timeline.\n"})
        content_list.append({"type": "text", "text": f"Here are {len(run_mapping)} unordered video clips.\n\n"})

        for item in run_mapping:
            content_list.append({"type": "text", "text": f"--- {item['clip_label']} ---\n"})
            content_list.append({"type": "video", "video": item['frames'], "fps": 1.0})
            content_list.append({"type": "text", "text": "\n"})

        prompt_text = (
            "TASK:\n"
            "1. Analyze each clip individually.\n"
            "2. Note that the Clip IDs are RANDOM and DO NOT indicate time.\n"
            "3. Identify logical dependencies.\n"
            "4. Provide the chronological order.\n\n"
            "OUTPUT FORMAT:\n"
            "Analysis Clip [ID]: [Description]\n"
            "Reasoning: ...\n"
            "Final Sequence: Clip [ID] -> Clip [ID] -> Clip [ID]"
        )
        content_list.append({"type": "text", "text": prompt_text})

        # Inferenza
        messages = [{"role": "user", "content": content_list}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        if "fps" in video_kwargs: _ = video_kwargs.pop("fps")

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True,
            return_tensors="pt", fps=1.0, **video_kwargs 
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=1024, temperature=0.1, do_sample=True
            )
        
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Salva risultato parziale in memoria
        run_data = {
            "run_index": run_idx + 1,
            "input_map": run_mapping,
            "model_raw_output": output_text
        }
        all_runs_results.append(run_data)
        print("Run completata.")

    #Salvo su file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Aggiungiamo meta-info sul subset usato
    full_dump = {
        "model_name": MODEL_PATH,
        "parameters": {
            "num_clips": NUM_TEST_CLIPS,
            "num_runs": NUM_REPEATED_RUNS
        },
        "subset_original_ids": sorted([s['step_id'] for s in fixed_subset_steps]),
        "runs": all_runs_results
    }
    
    with open(final_output_path, "w") as f:
        json.dump(full_dump, f, indent=4)
    
    print(f"\n✅ Tutte le {NUM_REPEATED_RUNS} run salvate in: {final_output_path}")

if __name__ == "__main__":
    main()