import json
import os
import random
import torch
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Ottimizzazione allocatore PyTorch per evitare frammentazione VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# PREPARAZIONE DATI (CPU BOUND)
# ==============================================================================

def prepare_batch_on_cpu(video_data, processor, num_clips, seed_offset):
    """
    Questa funzione gira su un thread secondario (CPU).
    Si occupa di I/O (immagini) e tokenizzazione.
    """
    video_id, group = video_data
    current_activity_name = group.iloc[0]['activity_name']
    
    clips_data = []
    for _, row in group.iterrows():
        clips_data.append({
            "original_step_id": row['step_id'],
            "taskgraph_id": row['taskgraph_id'],
            "frames": row['frames'],
            "activity_name": row['activity_name'],
            "label": row['label']
        })
        
    # Randomizzazione deterministica
    rng = random.Random(f"{video_id}_{seed_offset}") 
    rng.shuffle(clips_data)
    
    if len(clips_data) < num_clips:
        return None # Skip video troppo corti
        
    selected_clips = clips_data[:num_clips]
    random_ids = rng.sample(range(100, 1000), len(selected_clips))
    
    input_map = []
    model_input_clips = []
    
    # Preparazione prompt
    ids_list = []
    for idx, clip in enumerate(selected_clips):
        sid = str(random_ids[idx])
        ids_list.append(sid)
        input_map.append({
            "shuffled_id": sid,
            "original_step_id": clip['original_step_id'],
            "taskgraph_id": clip['taskgraph_id'],
            "label": clip['label']
        })
        model_input_clips.append({"shuffled_idx": sid, "frames": clip['frames']})

    ids_display = ", ".join(ids_list)
    clean_activity = current_activity_name.replace("_", " ")

    content = []
    text_prompt = (
        f"You are given {len(selected_clips)} shuffled segments of a video demonstrating the activity: '{clean_activity}'.\n"
        f"Each segment has a unique ID ({ids_display}).\n"
    )
    
    for clip in model_input_clips:
        idx = clip['shuffled_idx']
        text_prompt += f"Segment ID {idx}:\n"
        for f_path in clip['frames']:
            content.append({"type": "image", "image": f_path})
            
    text_prompt += (
        f"\nYour Goal: Reorder these segments into the correct chronological sequence for '{clean_activity}'.\n"
        f"Instructions:\n"
        f"1. ANALYZE: Briefly describe what is happening in each Segment ID.\n"
        f"2. REASON: Explain the logical order based on the cooking procedure.\n"
        f"3. SOLVE: Output the final sequence.\n\n"
        "STRICT OUTPUT FORMAT:\n"
        "Analysis: [Brief descriptions]\n"
        "Reasoning: [Step-by-step logic]\n"
        f"Final Sequence: {ids_list[-1]}->{ids_list[0]}->... (Use actual IDs)\n"
    )
    
    content.append({"type": "text", "text": text_prompt})
    messages = [{"role": "user", "content": content}]
    
    # Preprocessing immagini e testo
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    return {
        "video_id": video_id,
        "activity_name": current_activity_name,
        "subset_size": len(selected_clips),
        "seed_offset": seed_offset,
        "input_map": input_map,
        "model_inputs": inputs
    }

# ==============================================================================
# CORE INFERENCE CON PREFETCH
# ==============================================================================

def run_inference_with_loaded_model(
    model, processor, manifest_path, output_path, 
    num_clips, seed_offset=0, max_new_tokens=512, temperature=0
):
    device = model.device
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    df = pd.DataFrame(manifest)
    # Raggruppa e converti in lista per poter iterare
    grouped = list(df.groupby('video_id'))
    if len(grouped) == 0:
        print(f"Manifest vuoto: nessun video da processare.")
        return
    
    results_data = []
    desc_str = f"Inference (Clips={num_clips}, Seed={seed_offset})"
    
    # ThreadPool per preparare i dati in background mentre la GPU lavora
    # Max workers = 2 è sufficiente per saturare la GPU
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        
        # Sottometti il primo batch
        future_data = executor.submit(prepare_batch_on_cpu, grouped[0], processor, num_clips, seed_offset)
        
        for i in tqdm(range(len(grouped)), desc=desc_str):
            # Sottometti il prossimo lavoro in anticipo (se esiste)
            if i + 1 < len(grouped):
                next_future = executor.submit(prepare_batch_on_cpu, grouped[i+1], processor, num_clips, seed_offset)
            
            # Attendi il risultato corrente (dovrebbe essere quasi pronto se il prefetch funziona)
            data = future_data.result()
            
            # Aggiorna il puntatore al futuro per il prossimo giro
            if i + 1 < len(grouped):
                future_data = next_future
            
            if data is None: continue # Skip video troppo corti o errori
            
            # --- SEZIONE GPU (CRITICA) ---
            try:
                inputs = data['model_inputs'].to(device)
                
                generation_kwargs = {"max_new_tokens": max_new_tokens}
                if temperature > 0:
                    generation_kwargs["do_sample"] = True; generation_kwargs["temperature"] = temperature
                else:
                    generation_kwargs["do_sample"] = False
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **generation_kwargs)
                    
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                results_data.append({
                    "video_id": data['video_id'],
                    "activity_name": data['activity_name'],
                    "subset_size": data['subset_size'],
                    "seed_offset": data['seed_offset'],
                    "input_map": data['input_map'],
                    "model_raw_output": output_text
                })
                
            except Exception as e:
                print(f"\n Errore su video {data['video_id']}: {str(e)}")
                torch.cuda.empty_cache() # Svuota cache solo su errore
                continue

            # Pulizia VRAM periodica (meno frequente per velocità)
            if i % 50 == 0:
                torch.cuda.empty_cache()

    # Salvataggio finale
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=4)