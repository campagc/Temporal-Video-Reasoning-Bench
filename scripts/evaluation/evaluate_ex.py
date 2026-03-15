import os
import json
import re
import itertools
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# FUNZIONI DI SUPPORTO
# ==============================================================================

def parse_llm_sequence(raw_text, valid_ids_set):
    """Estrae la sequenza numerica dall'output del modello."""
    text = str(raw_text).lower()
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
    return list(sequence)

def get_linear_constraints(gt_sequence):
    """
    Costruisce tutte le coppie possibili mantenendo l'ordine cronologico.
    Equivale alle dipendenze di un grafo lineare e completo.
    Esempio: [1, 2, 3] -> [(1, 2), (1, 3), (2, 3)]
    """
    return list(itertools.combinations(gt_sequence, 2))

def calculate_pairwise_accuracy(predicted_sequence, constraints):
    """
    Calcola l'accuratezza Pairwise.
    Identica a quella che usi per i DAG, ma applicata ai constraints lineari.
    """
    valid_preds = [x for x in predicted_sequence if x is not None]
    correct = 0
    total = len(constraints)
    
    if total == 0: 
        return 1.0 
    
    for (u, v) in constraints:
        # Verifica se entrambi gli elementi sono stati predetti
        if u in valid_preds and v in valid_preds:
            # Se l'indice di u è minore dell'indice di v, l'ordine relativo è corretto
            if valid_preds.index(u) < valid_preds.index(v):
                correct += 1
                
    # Nota: se un nodo non viene predetto (viene saltato), la coppia conta come errore
    return correct / total

# ==============================================================================
# CALCOLO METRICHE E SCANSIONE CARTELLE
# ==============================================================================

def process_linear_pairwise(base_dir):
    """
    Scansiona ricorsivamente base_dir alla ricerca di '*_results.json'.
    Genera un file '*_linear_match.csv' per ogni esperimento.
    """
    print(f"Avvio scansione nella directory: {base_dir}")
    
    processed_count = 0

    for root, dirs, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith('results.json')]
        
        for json_file in json_files:
            input_path = os.path.join(root, json_file)
            output_name = json_file.replace('results.json', 'linear_match.csv')
            output_path = os.path.join(root, output_name)
            
            try:
                with open(input_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Errore nella lettura di {input_path}: {e}")
                continue

            rows = []
            for entry in data:
                # 1. Costruiamo la Sequenza Corretta Temporale (Ground Truth)
                input_map = entry['input_map']
                sorted_map = sorted(input_map, key=lambda x: int(x['original_step_id']))
                gt_sequence = [str(item['shuffled_id']) for item in sorted_map]
                valid_ids = set(gt_sequence)
                
                # 2. Estraiamo la Sequenza Predetta
                raw_output = entry.get('model_raw_output', '')
                pred_sequence = parse_llm_sequence(raw_output, valid_ids)
                
                # 3. Costruiamo i constraints (Tutte le coppie ordinate)
                linear_constraints = get_linear_constraints(gt_sequence)
                
                # 4. Calcoliamo la Linear Pairwise Accuracy (Equivalente al Kendall Tau normalizzato)
                linear_acc = calculate_pairwise_accuracy(pred_sequence, linear_constraints)
                
                # 5. Baseline Randomica (in una pairwise accuracy temporale pura, è matematicamente 0.50)
                rnd_linear_acc = 0.50
                
                rows.append({
                    "video_id": entry.get('video_id', 'unknown'),
                    "activity": entry.get('activity_name', 'unknown'),
                    "num_clips": len(gt_sequence),
                    "LLM_Linear_Acc": round(linear_acc, 4),
                    "RND_Linear_Acc": rnd_linear_acc,
                    "Delta_Linear_Acc": round(linear_acc - rnd_linear_acc, 4)
                })
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
                processed_count += 1
                
                avg_acc = df['LLM_Linear_Acc'].mean()
                print(f"Salvato: {output_name} | Media Linear Pairwise: {avg_acc:.1%}")

    print(f"\n✨ Operazione completata. Processati {processed_count} file JSON.")

if __name__ == "__main__":
    TARGET_DIRECTORY = "output_grid_experiment_com" 
    
    if os.path.exists(TARGET_DIRECTORY):
        process_linear_pairwise(TARGET_DIRECTORY)
    else:
        print(f"Cartella '{TARGET_DIRECTORY}' non trovata. Verifica il percorso corretto.")