import os
import json
import re
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau

# --- CONFIGURAZIONE ---
RESULT_PATH = "output/result_native_video.json"
MANIFEST_PATH = "data/processed_frames/dataset_manifest.json"
TASK_GRAPHS_DIR = "data/annotations_repo/task_graphs"
ANNOTATIONS_CSV = "data/annotations_repo/annotation_csv/step_annotations.csv"
# ----------------------

def load_json(path):
    if not os.path.exists(path):
        print(f"ERRORE: File non trovato: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def get_id_mapping(csv_path, subset_ids):
    """
    Legge il CSV e mappa l'INDICE DI RIGA (1640) nello STEP_ID (1, 2, 3...)
    """
    if not os.path.exists(csv_path):
        print(f"ERRORE: CSV annotazioni non trovato in {csv_path}")
        return {}

    # Carica CSV
    # Sep=None permette al motore python di indovinare se è virgola o altro
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip() # Rimuove spazi dai nomi colonne
    
    # 1. subset_ids sono gli indici di riga globali (es. 1640)
    subset_ids_int = [int(x) for x in subset_ids]
    
    # 2. Filtriamo il dataframe usando l'INDICE (non una colonna)
    # df.index contiene 0, 1, 2... 1640...
    filtered = df.loc[df.index.isin(subset_ids_int)]
    
    mapping = {}
    for global_idx, row in filtered.iterrows():
        # global_idx è 1640
        # row['step_id'] è l'ID locale (es. 1, 2, 3) usato nel task graph
        
        # Gestione sicurezza se 'step_id' non esiste (dal tuo readme dovrebbe esserci)
        if 'step_id' in row:
            local_id = int(row['step_id'])
            mapping[global_idx] = local_id
        else:
            print(f"⚠️ Riga {global_idx} non ha 'step_id'. Colonne disponibili: {df.columns}")

    return mapping

def parse_llm_sequence(raw_text):
    match = re.search(r"Final Sequence:?\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
    if not match:
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

def build_transitive_subgraph(graph_data, local_ids):
    """ Costruisce il grafo usando gli STEP_IDX (Locali) e applica chiusura transitiva. """
    G_full = nx.DiGraph()
    
    # Aggiungi nodi dal JSON (es. "1": "slicing...")
    if 'steps' not in graph_data:
        print(f"❌ ERRORE: JSON non ha chiave 'steps'. Chiavi disponibili: {graph_data.keys()}")
        return None
        
    for node_id, desc in graph_data['steps'].items():
        if node_id.isdigit():
            G_full.add_node(int(node_id), label=desc)
    
    print(f"✓ Caricati {len(G_full.nodes)} nodi dal grafo")
    
    if 'edges' not in graph_data:
        print(f"❌ ERRORE: JSON non ha chiave 'edges'. Chiavi disponibili: {graph_data.keys()}")
        return None
            
    # Aggiungi archi dal JSON
    for edge in graph_data['edges']:
        u, v = int(edge[0]), int(edge[1])
        G_full.add_edge(u, v)
        
    print(f"✓ Caricati {len(G_full.edges)} archi dal grafo")
    
    # Calcola chiusura transitiva su tutto il grafo
    G_transitive = nx.transitive_closure(G_full)
    
    # Estrai solo i nodi che ci interessano
    # (Verifica che esistano nel grafo per evitare errori se il CSV ha ID strani)
    valid_ids = [nid for nid in local_ids if nid in G_transitive.nodes]
    
    if len(valid_ids) < len(local_ids):
        missing = [nid for nid in local_ids if nid not in G_transitive.nodes]
        print(f"⚠️  Attenzione: Alcuni ID locali {missing} non sono nel file JSON del grafo!")
    
    G_sub = G_transitive.subgraph(valid_ids).copy()
    return G_sub

def compute_metrics(predicted_local_order, G_sub):
    if not predicted_local_order:
        return {"validity": 0.0, "violations": [], "kendall_tau": 0.0, "clean_pred": []}

    # Filtriamo su ciò che è nel grafo
    clean_pred = [x for x in predicted_local_order if x in G_sub.nodes]
    
    violations = 0
    total_pairs = 0
    violation_details = []

    for i in range(len(clean_pred)):
        for j in range(i + 1, len(clean_pred)):
            u = clean_pred[i] 
            v = clean_pred[j]
            total_pairs += 1
            
            # Se esiste V->U nel grafo, significa che U doveva stare DOPO V.
            # Ma qui abbiamo messo U prima di V. Errore.
            if G_sub.has_edge(v, u):
                violations += 1
                violation_details.append(f"Errore: Step {u} messo prima di {v} (Richiesto: {v} -> {u})")

    validity_score = 1.0 - (violations / total_pairs) if total_pairs > 0 else 0.0

    canonical_order = sorted(clean_pred)
    if len(clean_pred) > 1:
        rank_pred = [clean_pred.index(x) for x in canonical_order]
        rank_true = [canonical_order.index(x) for x in canonical_order]
        tau, _ = kendalltau(rank_pred, rank_true)
    else:
        tau = 1.0

    return {
        "validity_score": validity_score,
        "n_violations": violations,
        "violation_details": violation_details,
        "kendall_tau": tau,
        "clean_pred": clean_pred
    }

def main():
    print("--- EVALUATION SU SUBSET CON MAPPING ID (CORRETTO) ---")
    
    result_data = load_json(RESULT_PATH)
    manifest_data = load_json(MANIFEST_PATH)
    
    if not result_data: return

    # 1. Recupera Info base
    # Prova prima dal manifest, poi dal result_data (per backward compatibility)
    activity_name = manifest_data[0].get("activity_name") if manifest_data else None
    
    if not activity_name:
        # Fallback: prova a estrarre dal path del primo file o da metadati del result
        print("⚠️ activity_name non trovato nel manifest. Cercando alternative...")
        # Se il manifest non ha l'activity_name, potresti specificarlo manualmente
        activity_name = input("Inserisci il nome dell'attività (es. 'pinwheels', 'ramen'): ").strip().lower()
    
    graph_path = os.path.join(TASK_GRAPHS_DIR, f"{activity_name}.json")
    
    if not os.path.exists(graph_path):
        print(f"ERRORE: Grafo non trovato in {graph_path}")
        return
        
    gt_graph_data = load_json(graph_path)

    # 2. Recupera ID Globali (Input del modello, es. 1640)
    input_map = result_data.get("input_map", [])
    subset_global_ids = [int(item['original_id']) for item in input_map]
    print(f"ID Globali (Indici CSV): {subset_global_ids}")

    # 3. CREA MAPPING (Global Index -> Step ID)
    print("Lettura CSV per mappare Indici Riga -> Step ID...")
    mapping = get_id_mapping(ANNOTATIONS_CSV, subset_global_ids)
    
    subset_local_ids = []
    for gid in subset_global_ids:
        if gid in mapping:
            subset_local_ids.append(mapping[gid])
        else:
            print(f"⚠️ Warning: Riga {gid} non trovata nel mapping!")
            
    print(f"ID Locali Mappati (Step Graph): {subset_local_ids}")

    if not subset_local_ids:
        print("ERRORE CRITICO: Nessun ID locale recuperato. Impossibile costruire il grafo.")
        return

    # 4. Costruisci Grafo usando ID Locali
    G_sub = build_transitive_subgraph(gt_graph_data, subset_local_ids)
    
    if G_sub is None or len(G_sub.nodes) == 0:
        print("❌ ERRORE CRITICO: Impossibile costruire il grafo.")
        print(f"   - Activity Name: {activity_name}")
        print(f"   - Graph Path: {graph_path}")
        print(f"   - ID Locali Cercati: {subset_local_ids}")
        return
    
    print(f"\n✓ Grafo Subset Costruito: {len(G_sub.nodes)} nodi, {len(G_sub.edges)} archi.")
    print("Regole di Dipendenza (Step Index):")
    for u, v in G_sub.edges:
        print(f"  - Step {u} deve precedere Step {v}")

    # 5. Parsing Output LLM
    raw_output = result_data.get("model_raw_output", "")
    pred_random_ids = parse_llm_sequence(raw_output)
    
    # Catena di mapping: Random ID (LLM) -> Global ID (CSV Index) -> Local ID (Graph Step)
    random_to_global = {str(item['clip_id_num']): int(item['original_id']) for item in input_map}
    
    pred_local_ids = []
    for rid in pred_random_ids:
        if rid in random_to_global:
            g_id = random_to_global[rid]
            if g_id in mapping:
                pred_local_ids.append(mapping[g_id])
    
    print(f"\nPredizione Modello (Step Index): {pred_local_ids}")

    # 6. Calcolo Metriche
    metrics = compute_metrics(pred_local_ids, G_sub)

    # 7. Report
    print("\n" + "="*40)
    print("       REPORT VALUTAZIONE")
    print("="*40)
    print(f"Clip testate: {len(subset_local_ids)}")
    print(f"Clip trovate nella risposta: {len(metrics['clean_pred'])}")
    print("-" * 20)
    print(f"✅ Validity Score: {metrics['validity_score']:.2f}")
    print(f"📊 Kendall's Tau:  {metrics['kendall_tau']:.2f}")
    print(f"❌ Errori Logici:  {metrics['n_violations']}")
    if metrics['violation_details']:
        print("Dettagli:")
        for v in metrics['violation_details']:
            print(f"  - {v}")
    print("="*40)

if __name__ == "__main__":
    main()