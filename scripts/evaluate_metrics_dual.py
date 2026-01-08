import json
import os
import re
import random
import networkx as nx
import itertools
import statistics

# --- CONFIGURAZIONE ---
MANIFEST_PATH = "data/processed_frames/dataset_manifest.json"
REPEATED_RESULTS_PATH = "output/results_repeated.json"
TASK_GRAPHS_DIR = "data/annotations_repo/task_graphs"
NUM_RANDOM_BASELINE_RUNS = 200 # Numero alto per avere una media random stabile

# --- UTILS ---

def parse_llm_sequence(raw_text):
    text = raw_text.lower()
    if "final sequence" in text:
        text = text.split("final sequence")[-1]
    ids = re.findall(r'\b\d{3}\b', text)
    seen = set()
    return [x for x in ids if not (x in seen or seen.add(x))]

def build_full_graph(activity_name):
    json_path = os.path.join(TASK_GRAPHS_DIR, f"{activity_name}.json")
    if not os.path.exists(json_path): return None
    with open(json_path, 'r') as f: tg_data = json.load(f)
    G = nx.DiGraph()
    for edge in tg_data.get('edges', []): G.add_edge(edge[0], edge[1])
    for step_id in tg_data.get('steps', {}).keys(): G.add_node(int(step_id))
    return G

# --- VINCOLI TAK GRAPH ---

# Transitive Closure: Tutti i percorsi possibili (nonni inclusi).
def get_constraints_closure(G_full, subset_task_ids):
    constraints = []
    valid_ids = [x for x in subset_task_ids if x is not None]
    for u in valid_ids:
        for v in valid_ids:
            if u == v: continue
            if nx.has_path(G_full, u, v):
                constraints.append((u, v))
    return constraints

# Transitive Reduction: Solo i collegamenti diretti essenziali nel subset.
def get_constraints_reduction(G_full, subset_task_ids):
    valid_ids = [x for x in subset_task_ids if x is not None]
    subset_G = nx.DiGraph()
    subset_G.add_nodes_from(valid_ids)
    for u in valid_ids:
        for v in valid_ids:
            if u == v: continue
            if nx.has_path(G_full, u, v):
                subset_G.add_edge(u, v)
    reduced_G = nx.transitive_reduction(subset_G)
    return list(reduced_G.edges())

# --- METRICHE ---

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

# Kendall's Tau DAG
def calculate_kendalls_tau_dag(predicted_sequence, constraints, subset_ids):
    valid_preds = [x for x in predicted_sequence if x is not None]
    concordant = 0
    discordant = 0
    valid_input_ids = [x for x in subset_ids if x is not None]
    
    for u, v in itertools.permutations(valid_input_ids, 2):
        is_constraint = (u, v) in constraints
        if is_constraint:
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

def calculate_exact_match(pred_clip_ids, input_map):
    sorted_clips = sorted(input_map, key=lambda x: x['original_id'])
    original_seq = [c['clip_id_num'] for c in sorted_clips]
    matches = 0
    min_len = min(len(pred_clip_ids), len(original_seq))
    for i in range(min_len):
        if pred_clip_ids[i] == original_seq[i]: matches += 1
    return matches / min_len if min_len > 0 else 0

# Kendall's Tau standard (non DAG)
def calculate_standard_kendall_tau(pred_clip_ids, input_map):
    sorted_clips = sorted(input_map, key=lambda x: x['original_id'])
    gt_order = [c['clip_id_num'] for c in sorted_clips]
    
    gt_ranks = {cid: i for i, cid in enumerate(gt_order)}
    
    valid_preds = [c for c in pred_clip_ids if c in gt_ranks]
    
    n = len(valid_preds)
    if n < 2:
        return 0.0
        
    pred_ranks = [gt_ranks[c] for c in valid_preds]
    
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if pred_ranks[i] < pred_ranks[j]:
                concordant += 1
            else:
                discordant += 1
                
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0: return 0.0
    
    tau = (concordant - discordant) / total_pairs
    return tau

# --- MAIN ---

def main():
    print("--- EVALUATION BATCH: Comparison LLM vs Random ---\n")
    
    if not os.path.exists(REPEATED_RESULTS_PATH):
        print("File risultati non trovato.")
        return

    with open(MANIFEST_PATH, 'r') as f: manifest = json.load(f)
    with open(REPEATED_RESULTS_PATH, 'r') as f: results_data = json.load(f)

    step_to_task = {m['step_id']: m['taskgraph_id'] for m in manifest}
    activity_name = manifest[0]['activity_name']
    G_full = build_full_graph(activity_name)
    
    stats = {
        "closure_acc": [], "closure_tau": [],
        "reduction_acc": [], "reduction_tau": [],
        "exact_acc": [], "exact_tau": []
    }

    print(f"Attività: {activity_name}")
    print(f"Numero Run LLM: {len(results_data['runs'])}")
    print("-" * 120)
    print(f"{'Run':<4} | {'Clos.Acc':<9} {'Clos.Tau':<9} | {'Red.Acc':<9} {'Red.Tau':<9} | {'Exact.Acc':<9} {'Ex.Tau':<9}")
    print("-" * 120)

    baseline_subset_ids = None
    baseline_closure_const = None
    baseline_reduction_const = None
    baseline_input_map_template = None

    # --- 1. CALCOLO METRICHE LLM ---
    for run in results_data['runs']:
        run_map = {item['clip_id_num']: step_to_task.get(item['original_id']) for item in run['input_map']}
        subset_task_ids = [step_to_task.get(item['original_id']) for item in run['input_map']]
        
        pred_clips = parse_llm_sequence(run['model_raw_output'])
        pred_tasks = [run_map.get(c) for c in pred_clips if run_map.get(c) is not None]
        
        const_closure = get_constraints_closure(G_full, subset_task_ids)
        const_reduction = get_constraints_reduction(G_full, subset_task_ids)
        
        if baseline_subset_ids is None:
            baseline_subset_ids = subset_task_ids
            baseline_closure_const = const_closure
            baseline_reduction_const = const_reduction
            baseline_input_map_template = run['input_map']

        # DAG Metrics
        c_acc = calculate_pairwise_accuracy(pred_tasks, const_closure)
        c_tau = calculate_kendalls_tau_dag(pred_tasks, const_closure, subset_task_ids)
        r_acc = calculate_pairwise_accuracy(pred_tasks, const_reduction)
        r_tau = calculate_kendalls_tau_dag(pred_tasks, const_reduction, subset_task_ids)
        
        # Exact Metrics
        e_acc = calculate_exact_match(pred_clips, run['input_map'])
        e_tau = calculate_standard_kendall_tau(pred_clips, run['input_map'])
        
        print(f"{run['run_index']:<4} | {c_acc:.2%}    {c_tau:.2f}      | {r_acc:.2%}    {r_tau:.2f}      | {e_acc:.2%}    {e_tau:.2f}")
        
        stats["closure_acc"].append(c_acc)
        stats["closure_tau"].append(c_tau)
        stats["reduction_acc"].append(r_acc)
        stats["reduction_tau"].append(r_tau)
        stats["exact_acc"].append(e_acc)
        stats["exact_tau"].append(e_tau)

    # --- 2. CALCOLO BASELINE RANDOMICA ---
    # Simulo 200 volte un ordinamento casuale
    rnd_stats = {
        "closure_acc": [], "closure_tau": [],
        "reduction_acc": [], "reduction_tau": [],
        "exact_tau": [] 
    }
    
    # Recupero gli ID delle clip reali per mischiarli
    gt_clip_ids = [c['clip_id_num'] for c in sorted(baseline_input_map_template, key=lambda x: x['original_id'])]
    
    for _ in range(NUM_RANDOM_BASELINE_RUNS):
        # 1. Random Shuffle per DAG metrics
        rnd_preds_task = baseline_subset_ids[:]
        random.shuffle(rnd_preds_task)
        
        # 2. Random Shuffle per Exact metrics
        rnd_preds_clips = gt_clip_ids[:]
        random.shuffle(rnd_preds_clips)
        
        # Calcolo DAG metrics Random
        rnd_stats["closure_acc"].append(calculate_pairwise_accuracy(rnd_preds_task, baseline_closure_const))
        rnd_stats["closure_tau"].append(calculate_kendalls_tau_dag(rnd_preds_task, baseline_closure_const, baseline_subset_ids))
        rnd_stats["reduction_acc"].append(calculate_pairwise_accuracy(rnd_preds_task, baseline_reduction_const))
        rnd_stats["reduction_tau"].append(calculate_kendalls_tau_dag(rnd_preds_task, baseline_reduction_const, baseline_subset_ids))
        
        # Calcolo Exact Tau Random
        # Nota: baseline_input_map_template mi serve per le associazioni corrette
        rnd_stats["exact_tau"].append(calculate_standard_kendall_tau(rnd_preds_clips, baseline_input_map_template))

    # --- 3. REPORT DI COMPARAZIONE ---
    print("-" * 120)
    print("\n" + "="*85)
    print(f"REPORT FINALE DI COMPARAZIONE (LLM vs RANDOM)")
    print("="*85)
    print(f"{'METRICA':<25} | {'LLM Mean':<12} {'(StDev)':<10} | {'RANDOM Mean':<12} | {'DELTA':<10}")
    print("-" * 85)

    def print_comparison(label, llm_data, rnd_data):
        llm_mean = statistics.mean(llm_data)
        llm_std = statistics.stdev(llm_data) if len(llm_data) > 1 else 0
        rnd_mean = statistics.mean(rnd_data) if rnd_data else 0.0
        delta = llm_mean - rnd_mean
        
        # Formattazione corretta
        if "Acc" in label:
            llm_str = f"{llm_mean:.2%}"
            std_str = f"(\u00b1{llm_std:.2%})" # Ora anche la StDev è in %
            rnd_str = f"{rnd_mean:.2%}"
            delta_str = f"{delta:+.2%}"        # Anche il Delta è in %
        else:
            llm_str = f"{llm_mean:.2f}"
            std_str = f"(\u00b1{llm_std:.2f})"
            rnd_str = f"{rnd_mean:.2f}"
            delta_str = f"{delta:+.2f}"
        
        print(f"{label:<25} | {llm_str:<12} {std_str:<10} | {rnd_str:<12} | {delta_str:<10}")
        return delta

    delta_c_acc = print_comparison("Closure Accuracy", stats["closure_acc"], rnd_stats["closure_acc"])
    print_comparison("Closure Tau (DAG)", stats["closure_tau"], rnd_stats["closure_tau"])
    print("-" * 85)
    delta_r_acc = print_comparison("Reduction Accuracy", stats["reduction_acc"], rnd_stats["reduction_acc"])
    print_comparison("Reduction Tau (DAG)", stats["reduction_tau"], rnd_stats["reduction_tau"])
    print("=" * 85)
    
    # Aggiunta comparazione Exact Match Tau
    print_comparison("Exact Match Tau", stats["exact_tau"], rnd_stats["exact_tau"])
    print("=" * 85)

    # --- VERDETTO ---
    print("\n--- VERDETTO FINALE ---")
    
    if delta_r_acc > 0.15:
        print("✅ ECCELLENTE: Il modello ha batte il random (Reduction Delta > 15%)")
    elif delta_r_acc > 0.05:
        print("⚠️ BUONO MA INSTABILE: Il modello batte il random ma di poco (Reduction Delta > 5%)")
    elif delta_c_acc > 0.10 and delta_r_acc <= 0:
        print("⚠️ PARZIALE: Il modello capisce il flusso globale (Closure) ma sbaglia i dettagli locali (Reduction)")
    else:
        print("❌ FALLIMENTO: Il modello performa come (o peggio) del random")

if __name__ == "__main__":
    main()