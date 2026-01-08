import os
import json
import re
import networkx as nx
import numpy as np
from scipy.stats import kendalltau

# --- CONFIGURAZIONE ---
RESULT_PATH = "output/result_native_video.json"
MANIFEST_PATH = "data/processed_frames/dataset_manifest.json"
TASK_GRAPHS_DIR = "data/annotations_repo/task_graphs"
# ----------------------

def load_json(path):
    if not os.path.exists(path):
        print(f"ERRORE: File non trovato: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def parse_llm_sequence(raw_text):
    """
    Estrae la sequenza finale dall'output dell'LLM.
    Cerca pattern come 'Final Sequence: 104 -> 890 -> ...'
    """
    # 1. Cerca la sezione "Final Sequence"
    match = re.search(r"Final Sequence:?\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
    if not match:
        print("⚠️  Warning: Non ho trovato 'Final Sequence:' nel testo. Provo a cercare l'ultima riga con frecce.")
        lines = raw_text.strip().split('\n')
        candidate = [l for l in lines if "->" in l]
        if candidate:
            sequence_str = candidate[-1]
        else:
            return []
    else:
        sequence_str = match.group(1)

    # 2. Pulisci e splitta
    # Rimuove testo extra, newlines, ecc. tiene solo numeri e frecce
    clean_str = sequence_str.replace("Clip", "").strip()
    
    # Split per '->' o '>'
    parts = re.split(r"-?>", clean_str)
    
    # Estrai solo i numeri
    extracted_ids = []
    for p in parts:
        nums = re.findall(r"\d+", p)
        if nums:
            extracted_ids.append(nums[0]) # Prende il primo numero trovato nel chunk
            
    return extracted_ids

def build_gt_graph(graph_data, relevant_steps):
    """
    Costruisce un DAG NetworkX dal file JSON del task graph.
    Filtra solo i nodi che sono presenti effettivamente nella clip video (relevant_steps).
    """
    G = nx.DiGraph()
    
    # I nodi nel JSON sono stringhe "1", "2", ecc.
    # Convertiamo tutto in interi per coerenza con step_idx
    
    valid_nodes = set(relevant_steps)
    
    # Aggiungi nodi
    for node_id, desc in graph_data['steps'].items():
        if node_id.isdigit():
            n_id = int(node_id)
            if n_id in valid_nodes:
                G.add_node(n_id, label=desc)
    
    # Aggiungi archi
    for edge in graph_data['edges']:
        u, v = int(edge[0]), int(edge[1])
        # Aggiungi l'arco solo se entrambi i nodi sono nel nostro subset video
        if u in valid_nodes and v in valid_nodes:
            G.add_edge(u, v)
            
    return G

def compute_metrics(predicted_order, G_gt):
    """
    Confronta l'ordine predetto (lista di int) con il Grafo GT.
    """
    if not predicted_order:
        return {"validity": 0.0, "violations": [], "kendall_tau": 0.0}

    # 1. Check Violazioni Topologiche (Validity)
    # Per ogni coppia (u, v) nella predizione dove u viene prima di v:
    # C'è una violazione se nel grafo GT esiste un percorso v -> u (cioè v doveva essere prima)
    violations = 0
    total_pairs = 0
    violation_details = []

    for i in range(len(predicted_order)):
        for j in range(i + 1, len(predicted_order)):
            u = predicted_order[i]
            v = predicted_order[j]
            
            # Controlla se entrambi i nodi sono nel grafo (potrebbe aver allucinato ID)
            if u not in G_gt.nodes or v not in G_gt.nodes:
                continue
                
            total_pairs += 1
            
            # Se nel grafo reale V deve venire prima di U (v -> ... -> u), allora l'ordine u, v è sbagliato
            if nx.has_path(G_gt, v, u):
                violations += 1
                violation_details.append(f"Ha messo {u} prima di {v}, ma il grafo richiede {v} -> ... -> {u}")

    validity_score = 1.0 - (violations / total_pairs) if total_pairs > 0 else 0.0

    # 2. Kendall's Tau (Correlazione di rango)
    # Confrontiamo la posizione predetta con l'ordine 'canonico' (step_idx crescente)
    # Nota: Questo penalizza anche varianti valide se il grafo è lasco, ma è una buona baseline.
    canonical_order = sorted(predicted_order) # [1, 2, 3, 4...]
    
    # Creiamo due liste di rank per calcolare la correlazione
    rank_pred = [predicted_order.index(x) for x in canonical_order]
    rank_true = [canonical_order.index(x) for x in canonical_order] # sarà [0, 1, 2...]
    
    tau, _ = kendalltau(rank_pred, rank_true)

    return {
        "validity_score": validity_score,
        "n_violations": violations,
        "violation_details": violation_details,
        "kendall_tau": tau
    }

def main():
    print("--- FASE 3: EVALUATION ---")
    
    # 1. Carica i risultati
    result_data = load_json(RESULT_PATH)
    manifest_data = load_json(MANIFEST_PATH)
    
    if not result_data or not manifest_data:
        return

    # 2. Recupera activity_name per trovare il grafo giusto
    # Prendiamo il primo elemento del manifest per capire l'attività
    activity_name = manifest_data[0].get("activity_name")
    if not activity_name:
        print("ERRORE: activity_name non trovato nel manifest.")
        return
    
    graph_filename = f"{activity_name}.json"
    graph_path = os.path.join(TASK_GRAPHS_DIR, graph_filename)
    print(f"Loading Graph: {graph_path}")
    
    gt_graph_data = load_json(graph_path)
    if not gt_graph_data:
        return

    # 3. Decodifica Output LLM
    print("\nParsing output LLM...")
    raw_output = result_data.get("model_raw_output", "")
    pred_random_ids = parse_llm_sequence(raw_output)
    
    print(f"Sequenza estratta (Random IDs): {pred_random_ids}")
    
    # 4. Mappa Random ID -> Original Step ID
    input_map = result_data.get("input_map", [])
    # Crea dizionario: "105" -> 3
    id_map = {str(item['clip_id_num']): int(item['original_id']) for item in input_map}
    
    pred_step_ids = []
    for rid in pred_random_ids:
        if rid in id_map:
            pred_step_ids.append(id_map[rid])
        else:
            print(f"⚠️  ID allucinato dal modello: {rid} (non era nell'input)")

    print(f"Sequenza Mappata (Step IDs): {pred_step_ids}")
    
    # Gli step ID reali presenti nel video
    gt_step_ids = sorted([int(item['original_id']) for item in input_map])
    print(f"Sequenza Ground Truth Lineare: {gt_step_ids}")

    # 5. Costruisci Grafo e Calcola Metriche
    G = build_gt_graph(gt_graph_data, gt_step_ids)
    
    metrics = compute_metrics(pred_step_ids, G)
    
    # 6. Stampa Report
    print("\n" + "="*40)
    print("       REPORT VALUTAZIONE")
    print("="*40)
    print(f"Modello: {result_data.get('model_name')}")
    print(f"Attività: {activity_name}")
    print("-" * 20)
    print(f"Predicted Order: {pred_step_ids}")
    print(f"Nodes in GT:     {len(G.nodes)}")
    print(f"Edges in GT:     {len(G.edges)}")
    print("-" * 20)
    print(f"✅ Validity Score: {metrics['validity_score']:.2f} (1.0 = Perfetto rispetto al Grafo)")
    print(f"📉 Kendall's Tau:  {metrics['kendall_tau']:.2f} (1.0 = Perfetta correlazione rango)")
    print(f"❌ Violazioni:     {metrics['n_violations']}")
    
    if metrics['violation_details']:
        print("\nDettaglio Errori:")
        for v in metrics['violation_details']:
            print(f"  - {v}")
            
    print("="*40)

if __name__ == "__main__":
    main()