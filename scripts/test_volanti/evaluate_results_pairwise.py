import json
import networkx as nx
import os

# --- CONFIGURAZIONE ---
INPUT_FILE = "output/result_pairwise.json"
# ----------------------

def main():
    print("--- FASE 3: VALUTAZIONE E RICOSTRUZIONE ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} non trovato.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    
    results = data.get("results", [])
    
    # --- FIX ROBUSTEZZA ---
    # Se "clip_map" non c'è, ricaviamo le etichette (A, B, C...) dai risultati stessi
    if "clip_map" in data:
        clip_labels = list(data["clip_map"].keys())
    else:
        print("⚠️ Chiave 'clip_map' non trovata nel JSON. Estraggo gli ID dai risultati...")
        labels_set = set()
        for r in results:
            # Ogni risultato ha "pair": ["A", "B"]
            labels_set.add(r["pair"][0])
            labels_set.add(r["pair"][1])
        clip_labels = sorted(list(labels_set))
        print(f"ID trovati: {clip_labels}")
    # ----------------------

    # 1. Costruiamo il Grafo delle Predizioni
    G_pred = nx.DiGraph()
    G_pred.add_nodes_from(clip_labels)
    
    correct_edges = 0
    total_edges = 0
    
    print("\nAnalisi delle coppie:")
    for r in results:
        winner = r["predicted_winner"]
        # Se A e B giocano, e vince A, il perdente è B
        p1, p2 = r["pair"]
        loser = p1 if p2 == winner else p2
        
        # Ignoriamo i pareggi o errori
        if winner != "UNKNOWN":
            G_pred.add_edge(winner, loser)
            
            # Calcolo accuracy al volo sugli archi
            total_edges += 1
            if winner == r["truth_winner"]:
                correct_edges += 1
            else:
                # print(f"❌ Errore: Ha detto {winner} prima di {loser}, ma era il contrario.")
                pass # Decommenta per vedere gli errori uno per uno

    edge_acc = (correct_edges / total_edges * 100) if total_edges > 0 else 0
    print(f"\nAccuracy sulle coppie (Edge Accuracy): {edge_acc:.1f}%")

    # 2. Cerchiamo di ricostruire l'ordine (Topological Sort)
    try:
        predicted_order = list(nx.topological_sort(G_pred))
        print(f"\n✅ Ordine ricostruito (senza cicli): {predicted_order}")
    except nx.NetworkXUnfeasible:
        print("\n⚠️ Il modello ha creato un 'paradosso' (Ciclo nel grafo).")
        print("Esempio: A > B, B > C, ma C > A.")
        print("Applico strategia di recupero: Ordino per numero di vittorie.")
        
        # Metodo approssimativo: chi ha vinto più sfide sta prima
        wins = {node: 0 for node in clip_labels}
        for u, v in G_pred.edges():
            wins[u] += 1
        # Ordiniamo per chi ha più vittorie (decrescente)
        predicted_order = sorted(wins, key=wins.get, reverse=True)
        print(f"Ordine stimato: {predicted_order}")

    # 3. Confronto con la verità
    # Assumiamo ordine alfabetico crescente A->B->C... come ground truth
    # (perché nel codice inference avevamo ordinato gli step per ID prima di assegnare le lettere)
    ground_truth_order = sorted(clip_labels) 
    
    print(f"\nORDINE REALE:    {ground_truth_order}")
    print(f"ORDINE PREDETTO: {predicted_order}")
    
    # Calcoliamo Kendall Tau (approssimata) o simple match
    matches = sum([1 for i, x in enumerate(predicted_order) if x == ground_truth_order[i]])
    print(f"Elementi nella posizione assoluta corretta: {matches}/{len(clip_labels)}")
    
    if matches == len(clip_labels):
        print("\n🏆 SUCCESSO TOTALE! Sequenza perfettamente ricostruita.")
    else:
        print("\n🤔 Sequenza imperfetta, ma il modello ci ha provato.")

if __name__ == "__main__":
    main()