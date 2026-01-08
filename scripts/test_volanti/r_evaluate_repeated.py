import json
import os
import re
import random
import networkx as nx
import statistics # Per calcolare mean e stdev

# --- CONFIGURAZIONE ---
MANIFEST_PATH = "data/processed_frames/dataset_manifest.json"
REPEATED_RESULTS_PATH = "output/results_repeated.json"
TASK_GRAPHS_DIR = "data/annotations_repo/task_graphs"
NUM_RANDOM_BASELINE_RUNS = 100 # Facciamo una baseline molto solida (veloce da calcolare)

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
    if not os.path.exists(json_path): return None
    with open(json_path, 'r') as f: tg_data = json.load(f)
    G = nx.DiGraph()
    for edge in tg_data.get('edges', []): G.add_edge(edge[0], edge[1])
    for step_id in tg_data.get('steps', {}).keys(): G.add_node(int(step_id))
    return G

def get_subset_constraints(G_full, subset_task_ids):
    constraints = []
    valid_ids = [x for x in subset_task_ids if x is not None]
    for u in valid_ids:
        for v in valid_ids:
            if u == v: continue
            if nx.has_path(G_full, u, v): constraints.append((u, v))
    return constraints

def calculate_metrics(predicted_sequence, subset_constraints, subset_input_ids):
    valid_preds = [x for x in predicted_sequence if x is not None]
    input_set = set(x for x in subset_input_ids if x is not None)
    output_set = set(valid_preds)
    
    completeness = len(input_set.intersection(output_set)) / len(input_set) if len(input_set) > 0 else 0
    
    correct_constraints = 0
    total_constraints = len(subset_constraints)
    for (u, v) in subset_constraints:
        if u in valid_preds and v in valid_preds:
            if valid_preds.index(u) < valid_preds.index(v):
                correct_constraints += 1
    
    pairwise_acc = correct_constraints / total_constraints if total_constraints > 0 else 1.0

    violations = 0
    for i in range(len(valid_preds)):
        for j in range(i + 1, len(valid_preds)):
            u, v = valid_preds[i], valid_preds[j]
            if (v, u) in subset_constraints: violations += 1
            
    return {"pairwise_acc": pairwise_acc, "violations": violations, "completeness": completeness}

def main():
    print("--- EVALUATION: Repeated Runs Analysis ---\n")
    
    if not os.path.exists(REPEATED_RESULTS_PATH):
        print(f"File non trovato: {REPEATED_RESULTS_PATH}")
        return

    with open(MANIFEST_PATH, 'r') as f: manifest = json.load(f)
    with open(REPEATED_RESULTS_PATH, 'r') as f: results_data = json.load(f)

    # Setup Grafo
    step_to_task = {m['step_id']: m['taskgraph_id'] for m in manifest}
    activity_name = manifest[0]['activity_name']
    G = build_full_graph(activity_name)
    if not G: return

    # --- 1. Analisi delle Run dell'LLM ---
    llm_scores_acc = []
    llm_scores_viol = []
    llm_scores_comp = []

    # Recuperiamo gli ID task coinvolti (sono uguali per tutte le run, basta la prima)
    # Ma per sicurezza li ricalcoliamo per ogni run
    first_run_input_task_ids = [] 

    print(f"{'Run ID':<8} | {'Accuracy':<10} | {'Violations':<10} | {'Completeness'}")
    print("-" * 50)

    for run in results_data['runs']:
        # Mappa ClipID -> TaskID per QUESTA specifica run
        run_map = {}
        run_input_task_ids = []
        
        for item in run['input_map']:
            clip_id = item['clip_id_num']
            tg_id = step_to_task.get(item['original_id'])
            run_map[clip_id] = tg_id
            run_input_task_ids.append(tg_id)
        
        # Salviamo per la baseline (usiamo l'ultimo valido)
        first_run_input_task_ids = [x for x in run_input_task_ids if x is not None]

        # Vincoli del subset
        subset_constraints = get_subset_constraints(G, first_run_input_task_ids)

        # Parsing Output
        pred_clip_ids = parse_llm_sequence(run['model_raw_output'])
        pred_task_ids = [run_map.get(cid) for cid in pred_clip_ids if run_map.get(cid) is not None]

        # Metriche
        m = calculate_metrics(pred_task_ids, subset_constraints, first_run_input_task_ids)
        
        llm_scores_acc.append(m['pairwise_acc'])
        llm_scores_viol.append(m['violations'])
        llm_scores_comp.append(m['completeness'])

        print(f"Run {run['run_index']:<4} | {m['pairwise_acc']:.2%}    | {m['violations']:<10} | {m['completeness']:.0%}")

    # Statistiche LLM
    avg_llm_acc = statistics.mean(llm_scores_acc)
    std_llm_acc = statistics.stdev(llm_scores_acc) if len(llm_scores_acc) > 1 else 0
    avg_llm_viol = statistics.mean(llm_scores_viol)

    # --- 2. Baseline Randomica (Massiccia) ---
    # Usiamo gli stessi task ID (il subset è fisso)
    subset_constraints = get_subset_constraints(G, first_run_input_task_ids)
    
    rnd_accs = []
    for _ in range(NUM_RANDOM_BASELINE_RUNS):
        rnd_pred = first_run_input_task_ids[:]
        random.shuffle(rnd_pred)
        m = calculate_metrics(rnd_pred, subset_constraints, first_run_input_task_ids)
        rnd_accs.append(m['pairwise_acc'])
    
    avg_rnd_acc = statistics.mean(rnd_accs)
    
    # --- REPORT FINALE ---
    print("\n" + "="*60)
    print(f"REPORT FINALE (Media su {len(results_data['runs'])} Run LLM)")
    print("="*60)
    print(f"LLM Avg Accuracy:      {avg_llm_acc:.2%}  (± {std_llm_acc:.2%})")
    print(f"LLM Avg Violations:    {avg_llm_viol:.1f}")
    print(f"LLM Avg Completeness:  {statistics.mean(llm_scores_comp):.0%}")
    print("-" * 60)
    print(f"Random Baseline (Avg): {avg_rnd_acc:.2%}")
    print("-" * 60)
    
    delta = avg_llm_acc - avg_rnd_acc
    print(f"Miglioramento su Random: {delta:+.2%}")
    if delta > 0.15:
        print("✅ RISULTATO: Il modello dimostra un ragionamento robusto.")
    elif delta > 0:
        print("⚠️ RISULTATO: Il modello è leggermente meglio del caso, ma instabile.")
    else:
        print("❌ RISULTATO: Il modello non sta ragionando (livello casuale).")

if __name__ == "__main__":
    main()