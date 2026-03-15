import json
import os
import re
import random
import networkx as nx
import itertools
import statistics
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# CONFIGURAZIONE DEFAULT (Fallback per retrocompatibilità)
# ==============================================================================
DEFAULT_TASK_GRAPHS_DIR = "data/annotations_repo/task_graphs"
NUM_RANDOM_BASELINE_RUNS = 100

# ==============================================================================
# FUNZIONI CORE DI ANALISI
# ==============================================================================

def build_full_graph(activity_name, task_graphs_dir):
    """
    Costruisce il grafo completo cercando nel path specificato.
    Accetta task_graphs_dir come argomento dinamico.
    """
    candidates = [
        os.path.join(task_graphs_dir, f"{activity_name}.json"),
        os.path.join(task_graphs_dir, f"{activity_name.replace(' ', '_')}.json"),
        os.path.join(task_graphs_dir, f"{activity_name.lower().replace(' ', '')}.json")
    ]
    
    tg_data = None
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r') as f:
                tg_data = json.load(f)
            break
            
    if tg_data is None:
        return None

    G = nx.DiGraph()
    for edge in tg_data.get('edges', []): 
        G.add_edge(int(edge[0]), int(edge[1]))
    # Aggiungi nodi isolati
    for step_id in tg_data.get('steps', {}).keys(): 
        G.add_node(int(step_id))
    return G

def parse_llm_sequence(raw_text, valid_ids_set):
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

def get_constraints_closure(G_full, subset_task_ids):
    constraints = []
    valid_ids = [x for x in subset_task_ids if x is not None]
    for u in valid_ids:
        for v in valid_ids:
            if u == v: continue
            if nx.has_path(G_full, u, v):
                constraints.append((u, v))
    return constraints

def get_constraints_reduction(G_full, subset_task_ids):
    valid_ids = [x for x in subset_task_ids if x is not None]
    subset_G = nx.DiGraph()
    subset_G.add_nodes_from(valid_ids)
    for u in valid_ids:
        for v in valid_ids:
            if u == v: continue
            if nx.has_path(G_full, u, v):
                subset_G.add_edge(u, v)
    try:
        reduced_G = nx.transitive_reduction(subset_G)
        return list(reduced_G.edges())
    except:
        return list(subset_G.edges())

def calculate_pairwise_accuracy(predicted_sequence, constraints):
    valid_preds = [x for x in predicted_sequence if x is not None]
    correct = 0
    total = len(constraints)
    if total == 0: return 1.0 
    
    for (u, v) in constraints:
        if u in valid_preds and v in valid_preds:
            if valid_preds.index(u) < valid_preds.index(v):
                correct += 1
    return correct / total

def calculate_kendalls_tau_dag(predicted_sequence, constraints, subset_ids):
    valid_preds = [x for x in predicted_sequence if x is not None]
    concordant = 0
    discordant = 0
    valid_input_ids = [x for x in subset_ids if x is not None]
    
    relevant_pairs = 0
    for u, v in itertools.permutations(valid_input_ids, 2):
        if (u, v) in constraints:
            relevant_pairs += 1
            try:
                idx_u = valid_preds.index(u)
                idx_v = valid_preds.index(v)
                if idx_u < idx_v: concordant += 1
                else: discordant += 1
            except ValueError:
                discordant += 1 
                
    total = concordant + discordant
    if total == 0: return 1.0
    tau = (concordant - discordant) / total
    return (tau + 1) / 2 

# ==============================================================================
# FUNZIONE CORE PER LA VALUTAZIONE
# ==============================================================================

def evaluate_file(json_input_path, csv_output_path, task_graphs_path=None):
    """
    Funzione entry point per valutare un file JSON di risultati.
    task_graphs_path: percorso opzionale della cartella grafi.
    """
    
    # Se task_graphs_path è None (vecchie chiamate), usa il default (CaptainCook)
    # Se è passato (nuove chiamate da grid experiment), usa quello.
    graphs_dir = task_graphs_path if task_graphs_path else DEFAULT_TASK_GRAPHS_DIR
    
    print(f"Analisi file: {os.path.basename(json_input_path)}")
    print(f"Using Graphs Dir: {graphs_dir}")
    
    if not os.path.exists(json_input_path):
        print(f"File input non trovato: {json_input_path}")
        return None

    with open(json_input_path, 'r') as f:
        results_data = json.load(f)

    final_csv_rows = []
    graph_cache = {} 

    for entry in tqdm(results_data, desc="Calculating Metrics"):
        video_id = entry['video_id']
        activity_name = entry['activity_name']
        input_map = entry['input_map']
        raw_output = entry['model_raw_output']

        # 1. Carica Grafo (usando la directory dinamica)
        if activity_name not in graph_cache:
            graph_cache[activity_name] = build_full_graph(activity_name, graphs_dir)
        G_full = graph_cache[activity_name]
        
        if G_full is None:
            # print(f"Warning: Grafo non trovato per {activity_name}")
            continue

        shuffled_to_task = {item['shuffled_id']: item['taskgraph_id'] for item in input_map}
        valid_random_ids = set(shuffled_to_task.keys())
        subset_task_ids = [item['taskgraph_id'] for item in input_map]
        
        pred_shuffled_ids = parse_llm_sequence(raw_output, valid_random_ids)
        pred_task_sequence = [shuffled_to_task.get(sid) for sid in pred_shuffled_ids if sid in shuffled_to_task]

        const_closure = get_constraints_closure(G_full, subset_task_ids)
        const_reduction = get_constraints_reduction(G_full, subset_task_ids)
        
        llm_clos_acc = calculate_pairwise_accuracy(pred_task_sequence, const_closure)
        llm_clos_tau = calculate_kendalls_tau_dag(pred_task_sequence, const_closure, subset_task_ids)
        llm_red_acc = calculate_pairwise_accuracy(pred_task_sequence, const_reduction)
        llm_red_tau = calculate_kendalls_tau_dag(pred_task_sequence, const_reduction, subset_task_ids)
        
        # Baseline Random
        rnd_clos_accs, rnd_red_accs = [], []
        for _ in range(NUM_RANDOM_BASELINE_RUNS):
            rnd_seq = subset_task_ids[:]
            random.shuffle(rnd_seq)
            rnd_clos_accs.append(calculate_pairwise_accuracy(rnd_seq, const_closure))
            rnd_red_accs.append(calculate_pairwise_accuracy(rnd_seq, const_reduction))
            
        rnd_clos_acc_mean = statistics.mean(rnd_clos_accs)
        rnd_red_acc_mean = statistics.mean(rnd_red_accs)

        row = {
            "video_id": video_id,
            "activity": activity_name,
            "num_clips": len(subset_task_ids),
            "LLM_Closure_Acc": round(llm_clos_acc, 4),
            "LLM_Closure_Tau": round(llm_clos_tau, 4),
            "LLM_Reduction_Acc": round(llm_red_acc, 4),
            "LLM_Reduction_Tau": round(llm_red_tau, 4),
            "RND_Closure_Acc": round(rnd_clos_acc_mean, 4),
            "RND_Reduction_Acc": round(rnd_red_acc_mean, 4),
            "Delta_Closure_Acc": round(llm_clos_acc - rnd_clos_acc_mean, 4),
            "Delta_Reduction_Acc": round(llm_red_acc - rnd_red_acc_mean, 4)
        }
        final_csv_rows.append(row)

    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df_res = pd.DataFrame(final_csv_rows)
    df_res.to_csv(csv_output_path, index=False)
    
    if not df_res.empty:
        return df_res[["LLM_Closure_Acc", "LLM_Reduction_Acc"]].mean().to_dict()
    else:
        return {}

if __name__ == "__main__":
    pass