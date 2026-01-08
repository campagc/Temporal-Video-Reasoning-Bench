import json
import os
import re
import random
import networkx as nx
import itertools

# --- CONFIGURAZIONE ---
MANIFEST_PATH = "data/processed_frames/dataset_manifest.json"
RESULT_JSON_PATH = "output/result_native_video.json"
TASK_GRAPHS_DIR = "data/annotations_repo/task_graphs"
NUM_RANDOM_RUNS = 5  # Quante volte generiamo una sequenza random

def parse_llm_sequence(raw_text):
    text = raw_text.lower()
    if "final sequence" in text:
        text = text.split("final sequence")[-1]
    ids = re.findall(r'\b\d{3}\b', text)
    seen = set()
    cleaned_ids = [x for x in ids if not (x in seen or seen.add(x))]
    return cleaned_ids

def build_full_graph(activity_name):
    json_path = os.path.join(TASK_GRAPHS_DIR, f"{activity_name}.json")
    if not os.path.exists(json_path):
        print(f"❌ ERRORE: Task Graph non trovato: {json_path}")
        return None
    with open(json_path, 'r') as f:
        tg_data = json.load(f)
    G = nx.DiGraph()
    for edge in tg_data.get('edges', []):
        G.add_edge(edge[0], edge[1])
    for step_id in tg_data.get('steps', {}).keys():
        G.add_node(int(step_id))
    return G

def get_subset_constraints(G_full, subset_task_ids):
    constraints = []
    valid_ids = [x for x in subset_task_ids if x is not None]
    for u in valid_ids:
        for v in valid_ids:
            if u == v: continue
            if nx.has_path(G_full, u, v):
                constraints.append((u, v))
    return constraints

def calculate_metrics(predicted_sequence, subset_constraints, subset_input_ids):
    valid_preds = [x for x in predicted_sequence if x is not None]
    
    # 1. Completeness
    input_set = set(x for x in subset_input_ids if x is not None)
    output_set = set(valid_preds)
    completeness_score = len(input_set.intersection(output_set)) / len(input_set) if len(input_set) > 0 else 0

    # 2. Pairwise Accuracy
    correct_constraints = 0
    total_constraints = len(subset_constraints)
    
    for (u, v) in subset_constraints:
        if u in valid_preds and v in valid_preds:
            idx_u = valid_preds.index(u)
            idx_v = valid_preds.index(v)
            if idx_u < idx_v:
                correct_constraints += 1
    
    pairwise_acc = correct_constraints / total_constraints if total_constraints > 0 else 1.0

    # 3. Topological Violations
    violations = 0
    for i in range(len(valid_preds)):
        for j in range(i + 1, len(valid_preds)):
            pred_u = valid_preds[i]
            pred_v = valid_preds[j]
            if (pred_v, pred_u) in subset_constraints:
                violations += 1

    return {
        "completeness": completeness_score,
        "pairwise_accuracy": pairwise_acc,
        "correct_pairs": correct_constraints,
        "total_subset_constraints": total_constraints,
        "topological_violations": violations
    }

def main():
    print("--- FASE 3: EVALUATION (LLM vs Random Baseline Dettagliato) ---\n")
    
    if not os.path.exists(RESULT_JSON_PATH):
        print(f"File non trovato: {RESULT_JSON_PATH}")
        return

    # 1. Caricamento Dati
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    with open(RESULT_JSON_PATH, 'r') as f:
        result_data = json.load(f)

    # 2. Creazione Mappe
    step_to_task = {m['step_id']: m['taskgraph_id'] for m in manifest}
    activity_name = manifest[0]['activity_name']
    
    input_clip_map = {}
    input_task_ids = []
    
    for item in result_data['input_map']:
        clip_num = item['clip_id_num']
        orig_step = item['original_id']
        tg_id = step_to_task.get(orig_step)
        input_clip_map[clip_num] = tg_id
        input_task_ids.append(tg_id)

    clean_input_task_ids = [x for x in input_task_ids if x is not None]

    # 3. Grafo e Vincoli
    G = build_full_graph(activity_name)
    if not G: return
    subset_constraints = get_subset_constraints(G, clean_input_task_ids)
    
    print(f"Attività: {activity_name}")
    print(f"Numero Clip nel subset: {len(clean_input_task_ids)}")
    print(f"Vincoli totali da rispettare: {len(subset_constraints)}")

    # --- VALUTAZIONE LLM ---
    raw_output = result_data['model_raw_output']
    pred_clip_ids = parse_llm_sequence(raw_output)
    pred_task_ids_llm = []
    for cid in pred_clip_ids:
        tid = input_clip_map.get(cid)
        if tid is not None:
            pred_task_ids_llm.append(tid)
            
    llm_metrics = calculate_metrics(pred_task_ids_llm, subset_constraints, clean_input_task_ids)

    # --- VALUTAZIONE RANDOM BASELINE ---
    print(f"\n--- Generazione {NUM_RANDOM_RUNS} Baseline Randomiche ---")
    print(f"{'Iterazione':<12} | {'Accuracy':<10} | {'Violations'}")
    print("-" * 40)
    
    random_acc_sum = 0
    random_viol_sum = 0
    random_individual_scores = [] # Lista per salvare i singoli punteggi
    
    for i in range(NUM_RANDOM_RUNS):
        # Mischia gli ID di input
        random_pred = clean_input_task_ids[:] 
        random.shuffle(random_pred)
        
        r_metrics = calculate_metrics(random_pred, subset_constraints, clean_input_task_ids)
        
        acc = r_metrics['pairwise_accuracy']
        viol = r_metrics['topological_violations']
        
        random_acc_sum += acc
        random_viol_sum += viol
        random_individual_scores.append(acc)
        
        # Stampa riga per riga
        print(f"Run {i+1:<8} | {acc:.2%}    | {viol}")

    avg_random_acc = random_acc_sum / NUM_RANDOM_RUNS
    avg_random_viol = random_viol_sum / NUM_RANDOM_RUNS

    # --- RISULTATI COMPARATIVI ---
    print("\n" + "="*55)
    print(f"{'METRICA':<25} | {'LLM (Tuo Modello)':<20} | {'RANDOM (Avg)':<20}")
    print("-" * 55)
    
    llm_acc_str = f"{llm_metrics['pairwise_accuracy']:.2%}"
    rnd_acc_str = f"{avg_random_acc:.2%}"
    
    print(f"{'Pairwise Accuracy':<25} | {llm_acc_str:<20} | {rnd_acc_str:<20}")
    print(f"{'Topological Violations':<25} | {llm_metrics['topological_violations']:<20} | {avg_random_viol:<20.1f}")
    
    llm_comp_str = f"{llm_metrics['completeness']:.0%}"
    print(f"{'Completeness':<25} | {llm_comp_str:<20} | {'100%':<20}")
    print("="*55 + "\n")

    # Salvataggio JSON combinato
    final_output = {
        "llm_metrics": llm_metrics,
        "random_baseline": {
            "runs": NUM_RANDOM_RUNS,
            "individual_accuracies": random_individual_scores,
            "avg_pairwise_accuracy": avg_random_acc,
            "avg_violations": avg_random_viol
        }
    }
    
    with open(os.path.join("output", "evaluation_comparison.json"), "w") as f:
        json.dump(final_output, f, indent=4)
        print("Report completo salvato in output/evaluation_comparison.json")

if __name__ == "__main__":
    main()