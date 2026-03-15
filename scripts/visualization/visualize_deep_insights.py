import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# 🛠️ CONFIGURAZIONE
# ==============================================================================
# IMPORTANTE: Ensure that questo path corrisponda alla tua cartella di output reale
# Se usi source 2, potrebbe essere "output_grid_experiment" o "output_grid_experiment_captaincook"
BASE_INPUT_DIR = "output_grid_internvl_captaincook"
OUTPUT_ANALYSIS_DIR = os.path.join(BASE_INPUT_DIR, "deep_insights")
os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)

# ==============================================================================
# CARICAMENTO DATI ROBUSTO
# ==============================================================================
def extract_params_from_path(filepath):
    """Tenta di estrarre frames e clips dal percorso file in modo sicuro."""
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
    print(f"Cerco file CSV in: {os.path.abspath(BASE_INPUT_DIR)}")
    search_pattern = os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv")
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        print(f"NESSUN file *_metrics.csv trovato! Controlla BASE_INPUT_DIR.")
        return pd.DataFrame()

    print(f"Trovati {len(all_files)} file CSV.")
    df_list = []
    
    for f in all_files:
        try:
            # Estrazione parametri dal path
            n_frames, n_clips = extract_params_from_path(f)
            
            # Se il path non è standard, prova a leggere se c'è scritto nel CSV (opzionale) o salta
            if n_frames is None or n_clips is None:
                # Fallback: prova a vedere se sono nel nome file o salta
                continue

            df = pd.read_csv(f)
            df['Frames'] = n_frames
            df['Clips'] = n_clips
            
            # Estrai Run ID
            filename = os.path.basename(f)
            if "run_" in filename:
                try:
                    run_id = int(filename.split('_')[1])
                    df['Run_ID'] = run_id
                except:
                    df['Run_ID'] = 0 # Default se fallisce
            
            df_list.append(df)
        except Exception as e:
            print(f"Errore lettura {f}: {e}")
            continue
            
    if not df_list: 
        return pd.DataFrame()
        
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Caricamento completato. Totale righe dati: {len(full_df)}")
    return full_df

# ==============================================================================
#  1. CLASSIFICA ATTIVITÀ (TOP & FLOP)
# ==============================================================================
def plot_activity_leaderboard(df, frames, clips):
    """Genera grafico per una specifica configurazione"""
    subset = df[(df['Clips'] == clips) & (df['Frames'] == frames)]
    
    if subset.empty: return

    # Raggruppa e ordina
    leaderboard = subset.groupby('activity')['LLM_Closure_Acc'].mean().reset_index()
    leaderboard = leaderboard.sort_values(by='LLM_Closure_Acc', ascending=False)

    plt.figure(figsize=(10, 14)) # Più alto per far stare tutte le label
    
    sns.barplot(
        data=leaderboard,
        y='activity',
        x='LLM_Closure_Acc',
        palette='viridis',
        hue='activity',
        legend=False
    )
    
    plt.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')
    plt.title(f"Activity Accuracy (Frames={frames}, Clips={clips})", fontsize=15)
    plt.xlabel("Mean Closure Accuracy", fontsize=12)
    plt.ylabel("")
    plt.xlim(0, 1.05)
    plt.grid(axis='x', alpha=0.3)
    
    # Salva
    filename = f"leaderboard_F{frames}_C{clips}.png"
    out_path = os.path.join(OUTPUT_ANALYSIS_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Generato: {filename}")
    plt.close()

# ==============================================================================
#  2. CONSISTENCY PLOT
# ==============================================================================
def plot_consistency(df, frames, clips):
    subset = df[(df['Clips'] == clips) & (df['Frames'] == frames)]
    if subset.empty: return
    
    # Calcola Statistiche per video attraverso le run
    video_stats = subset.groupby(['video_id', 'activity'])['LLM_Closure_Acc'].agg(['mean', 'std']).reset_index()
    video_stats = video_stats.dropna() # Rimuove chi ha una sola run (std è NaN)

    if video_stats.empty:
        print(f"Impossibile generare Consistency Plot per F{frames}C{clips}: serve più di 1 run.")
        return

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=video_stats,
        x='mean',
        y='std',
        alpha=0.6,
        s=80,
        color="#3498db"
    )

    plt.title(f"Stability Analysis (Frames={frames}, Clips={clips})", fontsize=15)
    plt.xlabel("Average Accuracy", fontsize=12)
    plt.ylabel("Standard Deviation (Instability)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    filename = f"consistency_F{frames}_C{clips}.png"
    out_path = os.path.join(OUTPUT_ANALYSIS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    print(f"Generato: {filename}")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    df = load_all_data()
    
    if not df.empty:
        # Trova tutte le combinazioni uniche presenti nei dati
        configs = df[['Frames', 'Clips']].drop_duplicates().values
        print(f"\n🔎 Found {len(configs)} configurazioni uniche. Generazione grafici...")
        
        for frames, clips in configs:
            print(f"\n--- Analisi Config: Frames {int(frames)} | Clips {int(clips)} ---")
            plot_activity_leaderboard(df, int(frames), int(clips))
            plot_consistency(df, int(frames), int(clips))
            
        print(f"\n🎉 Finito! Controlla: {OUTPUT_ANALYSIS_DIR}")
    else:
        print("\n ERRORE CRITICO: DataFrame vuoto. Verifica:")
        print(f"1. La cartella '{BASE_INPUT_DIR}' esiste?")
        print("2. I file csv sono dentro sottocartelle tipo 'frames_X/clips_Y'?")