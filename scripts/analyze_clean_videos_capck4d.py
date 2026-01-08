import pandas as pd
import os
import ast

# --- CONFIGURAZIONE ---
ANNOTATIONS_DIR = "data/annotations_repo/annotation_csv"
ERROR_CSV = os.path.join(ANNOTATIONS_DIR, "error_annotations.csv")
REC_STEP_CSV = os.path.join(ANNOTATIONS_DIR, "recording_id_step_idx.csv")
ACTIVITY_MAP_CSV = os.path.join(ANNOTATIONS_DIR, "activity_idx_step_idx.csv")

def main():
    # Verifica esistenza file
    for p in [ERROR_CSV, REC_STEP_CSV, ACTIVITY_MAP_CSV]:
        if not os.path.exists(p):
            print(f"ERRORE: File non trovato {p}")
            return
    
    # Lettura CSV e eliminazione spazi vuoti
    df_errors = pd.read_csv(ERROR_CSV, sep=None, engine='python')
    df_rec_step = pd.read_csv(REC_STEP_CSV, sep=None, engine='python')
    df_activity = pd.read_csv(ACTIVITY_MAP_CSV, sep=None, engine='python')

    df_errors.columns = df_errors.columns.str.strip()
    df_rec_step.columns = df_rec_step.columns.str.strip()
    df_activity.columns = df_activity.columns.str.strip()

    # Ci sono vari controlli perché in giro non sempre le colonne hanno nomi coerenti
    # qualche volta activity_idx, qualche volta activity_id, ecc.

    rec_activity_col = 'activity_id'
    if rec_activity_col is None:
        print("ERRORE: Non trovo la colonna activity_id in recording_id_step_idx.csv")
        return

    # Creazione mappa Activity ID -> Activity Name
    map_activity_col = 'activity_idx' if 'activity_idx' in df_activity.columns else 'activity_id'
    activity_map = df_activity.groupby(map_activity_col)['activity_name'].first().to_dict()
    
    # Lista con tutte le activity
    all_known_activities = set(activity_map.values())

    # STEP 1: Trovare video SENZA errori
    video_error_sums = df_errors.groupby('recording_id')['has_errors'].sum()
    clean_video_ids = video_error_sums[video_error_sums == 0].index.tolist()
    clean_video_ids = [str(x) for x in clean_video_ids]

    print(f"\n--- STATISTICHE GENERALI ---")
    print(f"Totale video nel dataset: {len(video_error_sums)}")
    print(f"Video senza errori (Clean): {len(clean_video_ids)}")
    print(f"Totale Activity (Ricette) nel dataset: {len(all_known_activities)}")


    # STEP 2: Filtrare video SENZA step ripetuti
    clean_unique_videos = []
    clean_unique_details = []

    for _, row in df_rec_step.iterrows():
        rec_id = str(row['recording_id'])
        
        if rec_id not in clean_video_ids:
            continue
            
        try:
            steps = ast.literal_eval(row['step_indices'])
        except:
            continue
            
        # Verifica duplicati
        if len(steps) == len(set(steps)):
            clean_unique_videos.append(rec_id)
            
            act_id = row[rec_activity_col]
            act_name = activity_map.get(act_id, "Unknown")
            
            clean_unique_details.append({
                "id": rec_id,
                "activity": act_name,
                "num_steps": len(steps)
            })

    # REPORT FINALE
    print(f"\n--- RISULTATO FILTRAGGIO ---")
    print(f"Numero Video Perfetti (No Errori + No Ripetizioni): {len(clean_unique_videos)}")
    
    df_perfetti = pd.DataFrame(clean_unique_details)
    
    # Insieme delle activity che hanno almeno un video adatto
    if not df_perfetti.empty:
        activities_with_videos = set(df_perfetti['activity'].unique())
        activity_counts = df_perfetti['activity'].value_counts()
    else:
        activities_with_videos = set()
        activity_counts = pd.Series()

    # Insieme delle activity che NON hanno video
    activities_zero = all_known_activities - activities_with_videos

    # --- STAMPA 1: ACTIVITY CON VIDEO ---
    if not df_perfetti.empty:
        print("\n" + "="*60)
        print("✅ ACTIVITY CON VIDEO ADATTI (Ordinate per frequenza)")
        print("="*60)
        
        for activity, count in activity_counts.items():
            # Filtra video di questa activity
            videos = df_perfetti[df_perfetti['activity'] == activity]
            # Ordina per numero di step (più corto = più facile)
            videos = videos.sort_values('num_steps')
            
            ids_list = videos['id'].tolist()
            steps_list = videos['num_steps'].tolist()
            
            print(f"\nActivity: {activity} ({count} video)")
            # Stampa primi 5 ID
            for vid, nstep in zip(ids_list[:5], steps_list[:5]):
                print(f"  - ID: {vid:<8} (Steps: {nstep})")
    else:
        print("\n❌ NESSUN VIDEO ADATTO TROVATO.")

    # --- STAMPA 2: ACTIVITY CON 0 VIDEO ---
    if activities_zero:
        print("\n" + "="*60)
        print(f"⛔ ACTIVITY CON 0 VIDEO ADATTI ({len(activities_zero)})")
        print("="*60)
        # Ordina alfabeticamente per leggibilità
        for act in sorted(list(activities_zero)):
            print(f"  - {act}")
    else:
        print("\nTutte le activity hanno almeno un video adatto!")

    # Salvataggio su file
    os.makedirs("output", exist_ok=True)
    with open("output/clean_videos_list.txt", "w") as f:
        f.write("--- LISTA VIDEO PERFETTI ---\n")
        for v in clean_unique_videos:
            f.write(f"{v}\n")
    print("\nLista ID video salvata in output/clean_videos_list.txt")

if __name__ == "__main__":
    main()