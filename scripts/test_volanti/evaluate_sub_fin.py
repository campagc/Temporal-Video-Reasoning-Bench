import json
import os
import re
import networkx as nx
import itertools

# --- CONFIGURAZIONE ---
MANIFEST_PATH = "data/processed_frames/dataset_manifest.json"
RESULT_JSON_PATH = "output/result_native_video.json"
TASK_GRAPHS_DIR = "data/annotations_repo/task_graphs"

def parse_llm_sequence(raw_text):
    """
    Estrae i numeri (Clip ID) dall'output dell'LLM.
    Cerchiamo di essere tolleranti: prende tutti i numeri a 3 cifre trovati.
    """
    text = raw_text.lower()
    # Se c'è una sezione specifica, taglia tutto quello che c'è prima
    if "final sequence" in text:
        text = text.split("final sequence")[-1]
    
    # Trova tutti i numeri di 3 cifre (es. 104, 892...)
    ids = re.findall(r'\b\d{3}\b', text)
    
    # Rimuovi duplicati preservando l'ordine
    seen = set()
    cleaned_ids = [x for x in ids if not (x in seen or seen.add(x))]
    return cleaned_ids

def build_full_graph(activity_name):
    """
    Carica il grafo completo (Ground Truth globale) dal file JSON.
    """
    json_path = os.path.join(TASK_GRAPHS_DIR, f"{activity_name}.json")
    if not os.path.exists(json_path):
        print(f"❌ ERRORE: Task Graph non trovato: {json_path}")
        return None

    with open(json_path, 'r') as f:
        tg_data = json.load(f)

    G = nx.DiGraph()
    for edge in tg_data.get('edges', []):
        G.add_edge(edge[0], edge[1])
    
    # Aggiungi anche i nodi isolati per sicurezza
    for step_id in tg_data.get('steps', {}).keys():
        G.add_node(int(step_id))
        
    return G

def get_subset_constraints(G_full, subset_task_ids):
    """
    Dato il grafo completo e la lista degli ID presenti nel sottoinsieme (input),
    restituisce TUTTE le coppie (u, v) tali che u deve precedere v.
    Sfrutta la transitività: se A->B->C nel grafo e abbiamo {A, C}, rileva A->C.
    """
    constraints = []
    # Filtra None se ce ne sono
    valid_ids = [x for x in subset_task_ids if x is not None]
    
    # Controlla ogni possibile coppia nel sottoinsieme
    for u in valid_ids:
        for v in valid_ids:
            if u == v: continue
            
            # Se nel grafo completo esiste un percorso da u a v, 
            # allora u DEVE venire prima di v anche nel sottoinsieme.
            if nx.has_path(G_full, u, v):
                constraints.append((u, v))
                
    return constraints

def calculate_metrics(predicted_sequence, subset_constraints, subset_input_ids):
    """
    predicted_sequence: lista di TaskGraph ID nell'ordine dato dall'LLM
    subset_constraints: lista di tuple (u, v) che SONO VERE nel sottoinsieme
    subset_input_ids: lista degli ID che erano in input (per controllare se mancano pezzi)
    """
    
    valid_preds = [x for x in predicted_sequence if x is not None]
    
    # 1. Quante clip di input sono state effettivamente incluse nell'output?
    input_set = set(x for x in subset_input_ids if x is not None)
    output_set = set(valid_preds)
    missing_items = input_set - output_set
    hallucinated_items = output_set - input_set
    
    completeness_score = len(input_set.intersection(output_set)) / len(input_set) if len(input_set) > 0 else 0

    # 2. Pairwise Accuracy (Sui vincoli noti del subset)
    # Controlliamo quante delle regole (u deve stare prima di v) sono state rispettate.
    correct_constraints = 0
    total_constraints = len(subset_constraints)
    
    for (u, v) in subset_constraints:
        # Se uno dei due manca nell'output, la relazione non può essere valutata (o è errata)
        if u in valid_preds and v in valid_preds:
            idx_u = valid_preds.index(u)
            idx_v = valid_preds.index(v)
            if idx_u < idx_v:
                correct_constraints += 1
        else:
            # Penalità: se manca un pezzo necessario per la relazione, conta come errore?
            # In genere sì per la pairwise accuracy "assoluta".
            pass

    pairwise_acc = correct_constraints / total_constraints if total_constraints > 0 else 1.0

    # 3. Topological Validity (Errori gravi)
    # C'è qualche inversione esplicita? (Es. ho messo B prima di A, ma doveva essere A->B)
    violations = 0
    for i in range(len(valid_preds)):
        for j in range(i + 1, len(valid_preds)):
            pred_u = valid_preds[i]
            pred_v = valid_preds[j]
            
            # Se esiste una regola inversa (pred_v dovrebbe stare prima di pred_u)
            # allora abbiamo commesso una violazione topologica
            if (pred_v, pred_u) in subset_constraints:
                violations += 1

    return {
        "completeness": completeness_score,
        "missing_clips": list(missing_items),
        "pairwise_accuracy": pairwise_acc,
        "correct_pairs": correct_constraints,
        "total_subset_constraints": total_constraints,
        "topological_violations": violations
    }

def main():
    print("--- FASE 3: EVALUATION (Subset-Aware) ---\n")
    
    if not os.path.exists(RESULT_JSON_PATH):
        print(f"File non trovato: {RESULT_JSON_PATH}")
        return

    # 1. Caricamento Dati
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    with open(RESULT_JSON_PATH, 'r') as f:
        result_data = json.load(f)

    # 2. Creazione Mappe
    # Step ID -> TaskGraph ID
    step_to_task = {m['step_id']: m['taskgraph_id'] for m in manifest}
    activity_name = manifest[0]['activity_name']
    
    # Clip ID (Random) -> TaskGraph ID (GT)
    # Qui recuperiamo ESATTAMENTE quali ID erano nel sottoinsieme di input
    input_clip_map = {} # "104" -> 5 (TaskID)
    input_task_ids = [] # [5, 2, 8, 1...]
    
    print("Mapping Input Subset:")
    for item in result_data['input_map']:
        clip_num = item['clip_id_num']
        orig_step = item['original_id']
        tg_id = step_to_task.get(orig_step)
        
        input_clip_map[clip_num] = tg_id
        input_task_ids.append(tg_id)
        print(f"  Clip {clip_num} == Step {orig_step} (TaskID: {tg_id})")

    # 3. Costruzione Grafo e Estrazione Vincoli del Subset
    G = build_full_graph(activity_name)
    if not G: return

    # Troviamo tutte le relazioni vere SOLO tra questi ID
    subset_constraints = get_subset_constraints(G, input_task_ids)
    print(f"\nVincoli trovati nel sottoinsieme ({len(subset_constraints)}):")
    for c in subset_constraints:
        print(f"  {c[0]} deve precedere {c[1]}")

    # 4. Parsing Predizione LLM
    raw_output = result_data['model_raw_output']
    pred_clip_ids = parse_llm_sequence(raw_output)
    
    # Converti Clip ID -> Task ID
    pred_task_ids = []
    for cid in pred_clip_ids:
        tid = input_clip_map.get(cid)
        if tid is not None:
            pred_task_ids.append(tid)
    
    print(f"\nPredizione LLM (Task IDs): {pred_task_ids}")

    # 5. Calcolo Metriche
    metrics = calculate_metrics(pred_task_ids, subset_constraints, input_task_ids)

    print("\n--- RISULTATI VALUTAZIONE ---")
    print(f"Completeness: {metrics['completeness']:.0%} (Mancanti: {metrics['missing_clips']})")
    print(f"Pairwise Accuracy: {metrics['pairwise_accuracy']:.2%} ({metrics['correct_pairs']}/{metrics['total_subset_constraints']} coppie corrette)")
    print(f"Topological Violations: {metrics['topological_violations']} (Inversioni logiche)")
    print("-----------------------------")

    # Salva
    with open(os.path.join("output", "evaluation_metrics_subset.json"), "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()