import json
import os
import re
import math
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# 🧠 FUNZIONI CORE DI ANALISI
# ==============================================================================

def parse_llm_sequence(raw_text, valid_ids_set):
    """Estrae la sequenza predetta dall'output del modello."""
    text = raw_text.lower()
    markers = ["final sequence:", "final answer:", "sequence:", "order:"]
    target_text = text
    for marker in markers:
        if marker in text:
            target_text = text.split(marker)[-1]
            break
            
    found_tokens = re.findall(r'\b\d+\b', target_text)
    sequence = []
    seen = set()
    for token in found_tokens:
        if token in valid_ids_set and token not in seen:
            sequence.append(token)
            seen.add(token)
    return sequence

# ==============================================================================
#  FUNZIONE CORE PER LA VALUTAZIONE EXACT MATCH
# ==============================================================================

def evaluate_exact_match(json_input_path, csv_output_path):
    """
    Legge i risultati JSON e calcola se la sequenza predetta combacia al 100% 
    con la sequenza cronologica originale (Exact Match).
    """
    print(f"🎯 Calcolo Exact Match per: {os.path.basename(json_input_path)}")
    
    if not os.path.exists(json_input_path):
        print(f"File input non trovato: {json_input_path}")
        return None

    with open(json_input_path, 'r') as f:
        results_data = json.load(f)

    final_csv_rows = []

    for entry in tqdm(results_data, desc="Calculating Exact Match"):
        video_id = entry['video_id']
        activity_name = entry['activity_name']
        input_map = entry['input_map']
        raw_output = entry['model_raw_output']

        # 1. Ricostruzione Ground Truth cronologico
        # Ordiniamo l'input_map in base all'id cronologico originale
        sorted_input = sorted(input_map, key=lambda x: float(x['original_step_id']))
        true_shuffled_ids = [str(item['shuffled_id']) for item in sorted_input]
        
        valid_random_ids = set(true_shuffled_ids)
        
        # 2. Estrazione predizione
        pred_shuffled_ids = parse_llm_sequence(raw_output, valid_random_ids)

        # 3. Calcolo Exact Match (1.0 se perfettamente uguale, altrimenti 0.0)
        is_exact_match = 1.0 if pred_shuffled_ids == true_shuffled_ids else 0.0
        
        # 4. Baseline Random
        # La probabilità di azzeccare l'ordine esatto tirando a caso è 1 diviso il fattoriale del numero di clip
        num_clips = len(true_shuffled_ids)
        rnd_exact_match = 1.0 / math.factorial(num_clips) if num_clips > 0 else 0.0

        row = {
            "video_id": video_id,
            "activity": activity_name,
            "num_clips": num_clips,
            "LLM_Exact_Match": is_exact_match,
            "RND_Exact_Match": round(rnd_exact_match, 6),
            "Delta_Exact_Match": round(is_exact_match - rnd_exact_match, 6)
        }
        final_csv_rows.append(row)

    # Salvataggio su CSV
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df_res = pd.DataFrame(final_csv_rows)
    df_res.to_csv(csv_output_path, index=False)
    
    if not df_res.empty:
        return df_res[["LLM_Exact_Match"]].mean().to_dict()
    else:
        return {}

if __name__ == "__main__":
    # Esempio di utilizzo locale (se lo lanci da solo per un file)
    # evaluate_exact_match("path/to/run_0_results.json", "path/to/run_0_exact_match.csv")
    pass