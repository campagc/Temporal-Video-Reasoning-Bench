import pandas as pd
import os

csv_path = "data/annotations_repo/annotation_csv/error_annotations.csv"

if not os.path.exists(csv_path):
    print(f"ERRORE: Non trovo il file {csv_path}")
    exit()

# Leggi il CSV
df = pd.read_csv(csv_path, sep=None, engine='python')
df.columns = df.columns.str.strip()

# LOGICA: 
# 1. Raggruppiamo per 'recording_id'
# 2. Per ogni video, sommiamo i valori di 'has_errors'
# 3. Se la somma è 0, significa che TUTTI gli step sono 0 (nessun errore)
video_errors = df.groupby('recording_id')['has_errors'].sum()

# Filtriamo gli ID che hanno somma errori pari a 0
clean_video_ids = video_errors[video_errors == 0].index.tolist()

print(f"Totale video analizzati: {df['recording_id'].nunique()}")
print(f"Video totalmente senza errori: {len(clean_video_ids)}")

if len(clean_video_ids) > 0:
    # Prendiamo l'ID così com'è (senza int())
    selected_id = clean_video_ids[0]
    
    # Filtriamo il dataframe originale per quell'ID
    # Usiamo lo stesso tipo di dato trovato da pandas
    video_rows = df[df['recording_id'] == selected_id]
    
    if not video_rows.empty:
        info_video = video_rows.iloc[0]
        
        print(f"\n--- VIDEO COMPLETAMENTE PULITO SELEZIONATO ---")
        print(f"ID (recording_id): {selected_id}")
        print(f"Descrizione primo step: {info_video['description']}")
    else:
        print(f"Errore strano: ID {selected_id} trovato ma non recuperabile nel dataframe.")
else:
    print("Nessun video totalmente pulito trovato.")