import os
import json
import re
import networkx as nx
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
    """ Estrae la sequenza numerica (ID casuali) dall'output LLM """
    match = re.search(r"Final Sequence:?\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
    if not match:
        # Fallback: cerca l'ultima riga che contiene frecce
        lines = [l for l in raw_text.strip().split('\n') if "->" in l]
        if lines:
            sequence_str = lines[-1]
        else:
            return []
    else:
        sequence_str = match.group(1)

    clean_str = sequence_str.replace("Clip", "").strip()
    parts = re.split(r"-?>", clean_str)
    
    extracted_ids = []
    for p in parts:
        nums = re.findall(r"\d+", p)
        if nums:
            extracted_ids.append(nums[0])
    return extracted_ids

def build_transitive_subgraph(graph_data, subset_ids):
    """
    1. Costruisce il grafo COMPLETO dell'attività.
    2. Calcola la Chiusura Transitiva (se A->B->C, aggiunge arco A->C).
    3. Restituisce solo il sottografo contenente le clip del subset.
    """
    # 1. Grafo Completo
    G_full = nx.DiGraph()
    
    # Aggiunge tutti i nodi possibili dal JSON
    for node_id, desc in graph_data['steps'].items():
        if node_id.isdigit():
            G_full.add_node(int(node_id), label=desc)
            
    # Aggiunge tutti gli archi
    for edge in graph_data['edges']:
        u, v = int(edge[0]), int(edge[1])
        G_full.add_edge(u, v)
        
    # 2. Chiusura Transitiva
    # Questo è vitale: se ho clip 3 e clip 10, e il grafo era 3->...->10,
    # ora avrò un arco diretto 3->10.
    G_transitive = nx.transitive_closure(G_full)
    
    # 3. Estrai Sottografo (Subset)
    # Teniamo solo i nodi che sono stati effettivamente dati al modello
    # Verifica che i nodi esistano nel grafo (per evitare crash su ID strani)
    valid_subset_ids = [nid for nid in subset_ids if nid in G_transitive.nodes]
    
    G_sub = G_transitive.subgraph(valid_subset_ids).copy()
    
    return G_sub

def compute_metrics(predicted_order, G_sub):
    """
    Confronta l'ordine predetto con il Sottografo Transitivo.
    """
    if not predicted_order:
        return {"validity": 0.0, "violations": [], "kendall_tau": 0.0}

    # Filtra la predizione: considera solo gli ID che erano nel subset di input
    # (Se il modello allucina ID esterni, li ignoriamo per la metrica di validità,
    # ma potresti volerli penalizzare in futuro)
    clean_pred = [x for x in predicted_order if x in G_sub.nodes]
    
    violations = 0
    total_pairs = 0
    violation_details = []

    # Validity Score (Pairwise)
    for i in range(len(clean_pred)):
        for j in range(i + 1, len(clean_pred)):
            u = clean_pred[i] # Step messo prima
            v = clean_pred[j] # Step messo dopo
            
            total_pairs += 1
            
            # Se nel grafo (chiusura transitiva) esiste V -> U
            # significa che V DOVEVA avvenire prima di U.
            # Il modello ha sbagliato l'ordine relativo.
            if G_sub.has_edge(v, u):
                violations += 1
                violation_details.append(f"Errore: {u} messo prima di {v} (Relazione nel grafo: {v} -> {u})")

    validity_score = 1.0 - (violations / total_pairs) if total_pairs > 0 else 0.0

    # Kendall's Tau
    # Ordine 'ideale' è semplicemente gli ID crescenti (approssimazione valida per cooking)
    canonical_order = sorted(clean_pred)
    
    if len(clean_pred) > 1:
        rank_pred = [clean_pred.index(x) for x in canonical_order]
        rank_true = [canonical_order.index(x) for x in canonical_order]
        tau, _ = kendalltau(rank_pred, rank_true)
    else:
        tau = 1.0 # Con 0 o 1 elemento è ordinato per definizione

    return {
        "validity_score": validity_score,
        "n_violations": violations,
        "violation_details": violation_details,
        "kendall_tau": tau,
        "clean_pred": clean_pred
    }

def main():
    print("--- EVALUATION SU SUBSET (5 CLIP) ---")
    
    # 1. Carica i risultati
    result_data = load_json(RESULT_PATH)
    manifest_data = load_json(MANIFEST_PATH)
    
    if not result_data: 
        return

    # 2. Trova il Task Graph corretto
    # Usiamo il manifest per recuperare l'activity name, oppure lo prendiamo dal json result se lo salvassimo
    # Qui assumiamo che il manifest corrisponda al video processato
    activity_name = manifest_data[0].get("activity_name")
    graph_path = os.path.join(TASK_GRAPHS_DIR, f"{activity_name}.json")
    
    print(f"Activity: {activity_name}")
    gt_graph_data = load_json(graph_path)

    # 3. Recupera gli Step ID reali usati nell'Input
    # Questa è la lista dei 5 step reali (es. 3, 15, 22, 25, 30)
    input_map = result_data.get("input_map", [])
    subset_real_ids = sorted([int(item['original_id']) for item in input_map])
    print(f"Subset Ground Truth (Original IDs): {subset_real_ids}")

    # 4. Costruisci il Sottografo Transitivo
    G_sub = build_transitive_subgraph(gt_graph_data, subset_real_ids)
    print(f"Costruito grafo subset: {len(G_sub.nodes)} nodi, {len(G_sub.edges)} dipendenze trovate.")

    # ... (codice precedente)

    # 4. Costruisci il Sottografo Transitivo
    G_sub = build_transitive_subgraph(gt_graph_data, subset_real_ids)
    
    print("\n--- CONTROLLO GRAFO DI VALUTAZIONE ---")
    print(f"Sto valutando SOresult_native_video.jsonLO su queste clip (Nodi): {list(G_sub.nodes)}")
    print("Regole di dipendenza valide per questo subset (Archi):")
    for u, v in G_sub.edges:
        print(f"  - Clip {u} deve venire PRIMA di Clip {v}")
    print("--------------------------------------\n")

    # 5. Parsing Output LLM
    raw_output = result_data.get("model_raw_output", "")
    pred_random_ids = parse_llm_sequence(raw_output)
    
    # Mapping Random ID -> Real ID
    id_map = {str(item['clip_id_num']): int(item['original_id']) for item in input_map}
    
    pred_step_ids = []
    for rid in pred_random_ids:
        if rid in id_map:
            pred_step_ids.append(id_map[rid])
    
    print(f"Predizione Modello (Real IDs): {pred_step_ids}")

    # 6. Calcolo Metriche
    metrics = compute_metrics(pred_step_ids, G_sub)

    # 7. Report
    print("\n" + "="*40)
    print("       REPORT VALUTAZIONE (SUBSET)")
    print("="*40)
    print(f"Clip testate: {len(subset_real_ids)}")
    print(f"Clip riconosciute nel'output: {len(metrics['clean_pred'])}")
    print("-" * 20)
    print(f"✅ Validity Score: {metrics['validity_score']:.2f}")
    print(f"📊 Kendall's Tau:  {metrics['kendall_tau']:.2f}")
    print(f"❌ Errori Logici:  {metrics['n_violations']}")
    
    if metrics['violation_details']:
        print("\nDettaglio:")
        for v in metrics['violation_details']:
            print(f"  - {v}")
            
    print("="*40)

if __name__ == "__main__":
    main()