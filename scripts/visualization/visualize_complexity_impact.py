import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# 🛠️ CONFIGURAZIONE
# ==============================================================================
# Ensure that questo path corrisponda alla tua cartella output reale
BASE_INPUT_DIR = "output_grid_internvl_captaincook"
OUTPUT_ANALYSIS_DIR = os.path.join(BASE_INPUT_DIR, "deep_insights")
os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)

# ==============================================================================
# CARICAMENTO DATI
# ==============================================================================
def extract_params_from_path(filepath):
    """Estrae frame e clip dal percorso del file in modo robusto"""
    parts = filepath.split(os.sep)
    n_frames = None
    n_clips = None
    
    for part in parts:
        if part.startswith("frames_") and part[7:].isdigit():
            n_frames = int(part.split("_")[1])
        elif part.startswith("clips_") and part[6:].isdigit():
            n_clips = int(part.split("_")[1])
            
    return n_frames, n_clips

def load_all_data():
    print(f"Scanning folder: {os.path.abspath(BASE_INPUT_DIR)}")
    search_pattern = os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv")
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        print(f"Nessun file CSV trovato!")
        return pd.DataFrame()

    df_list = []
    for f in all_files:
        try:
            n_frames, n_clips = extract_params_from_path(f)
            
            # Se il path non contiene le info, prova a cercarle nel nome file o salta
            if n_frames is None or n_clips is None:
                continue

            df = pd.read_csv(f)
            df['Frames'] = n_frames
            df['Clips'] = n_clips
            df_list.append(df)
        except Exception as e:
            print(f"Skip {f}: {e}")
            continue
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

# ==============================================================================
#  GENERAZIONE GRAFICI
# ==============================================================================
def plot_model_vs_random(df, frames, clips):
    """Genera lo scatter plot per una specifica configurazione"""
    
    # Filtra i dati
    subset = df[(df['Clips'] == clips) & (df['Frames'] == frames)].copy()
    
    if subset.empty:
        return

    # Raggruppa per activities (Media tra le run se ce ne sono più di una)
    activity_stats = subset.groupby('activity').agg({
        'LLM_Closure_Acc': 'mean',
        'RND_Closure_Acc': 'mean'
    }).reset_index()

    plt.figure(figsize=(10, 10))
    
    # Scatter Plot
    # Usiamo 'hue' solo se vuoi colorare per activities, altrimenti un colore unico è più pulito se sono tante
    sns.scatterplot(
        data=activity_stats,
        x='RND_Closure_Acc',
        y='LLM_Closure_Acc',
        s=150,
        alpha=0.7,
        edgecolor='black'
    )
    
    # Diagonale (Random Baseline)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label="Random Guessing Line")
    
    # Aree colorate per interpretazione immediata
    plt.fill_between([0, 1], [0, 1], 1, color='green', alpha=0.05, label="Skill Area (Better than Random)")
    plt.fill_between([0, 1], 0, [0, 1], color='red', alpha=0.05, label="Failure Area (Worse than Random)")

    # Etichette intelligenti: Mostra il nome solo per gli outlier significativi
    texts = []
    for i in range(len(activity_stats)):
        row = activity_stats.iloc[i]
        diff = row['LLM_Closure_Acc'] - row['RND_Closure_Acc']
        
        # Etichetta se il modello è molto più bravo del caso (>15%) o molto peggio (<-10%)
        if diff > 0.15 or diff < -0.10:
            plt.text(
                row['RND_Closure_Acc']+0.01, 
                row['LLM_Closure_Acc'], 
                row['activity'], 
                fontsize=9, 
                alpha=0.8,
                color='darkblue' if diff > 0 else 'darkred'
            )

    plt.title(f"Model vs Random Difficulty (Frames={frames}, Clips={clips})", fontsize=16)
    plt.xlabel("Task Ease (Random Baseline Accuracy)", fontsize=12)
    plt.ylabel("Model Accuracy (LLM)", fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal', adjustable='box')

    # Salvataggio
    filename = f"scatter_complexity_F{frames}_C{clips}.png"
    out_path = os.path.join(OUTPUT_ANALYSIS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    print(f"Grafico generato: {filename}")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    df = load_all_data()
    
    if not df.empty:
        # Trova tutte le combinazioni uniche (es. 6-5, 8-5, 8-8...)
        configs = df[['Frames', 'Clips']].drop_duplicates().values
        print(f"\n🎯 Found {len(configs)} configurazioni. Inizio generazione grafici...\n")
        
        for frames, clips in configs:
            plot_model_vs_random(df, int(frames), int(clips))
            
        print(f"\n🎉 Finito! Controlla la cartella: {OUTPUT_ANALYSIS_DIR}")
    else:
        print(f"Nessun dato caricato.")