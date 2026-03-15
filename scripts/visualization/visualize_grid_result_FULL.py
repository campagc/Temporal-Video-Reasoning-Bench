import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
BASE_INPUT_DIR = "output_grid_internvl_captaincook"
OUTPUT_ANALYSIS_DIR = os.path.join(BASE_INPUT_DIR, "analysis_full_metrics")
DESCRIPTION = "Captaincook4d - Qwen2.5-VL"  # Modifica questa descrizione per ogni dataset/model

# Lista delle metriche da analizzare (coppia: NomeColonnaCSV, EtichettaLeggibile)
METRICS_TO_ANALYZE = [
    ("LLM_Closure_Acc", "Pairwise Accuracy (Closure)", "RND_Closure_Acc"),
    ("LLM_Closure_Tau", "Kendall's Tau (Closure)", None), # Tau random è complesso, spesso non c'è baseline diretta nel csv
    ("LLM_Reduction_Acc", "Pairwise Accuracy (Reduction)", "RND_Reduction_Acc"),
    ("LLM_Reduction_Tau", "Kendall's Tau (Reduction)", None)
]

# ==============================================================================
#  LOGICA DI ANALISI
# ==============================================================================

def extract_params_from_path(filepath):
    parts = filepath.split(os.sep)
    n_frames = None
    n_clips = None
    run_id = None
    
    for part in parts:
        if part.startswith("frames_") and part[7:].isdigit():
            n_frames = int(part.split("_")[1])
        elif part.startswith("clips_") and part[6:].isdigit():
            n_clips = int(part.split("_")[1])
            
    filename = os.path.basename(filepath)
    if "run_" in filename and "_metrics.csv" in filename:
        try:
            run_id = int(filename.split("_")[1])
        except: run_id = 0
        
    return n_frames, n_clips, run_id

def main():
    print("==================================================")
    print(f"STARTING FULL METRIC ANALYSIS")
    print("==================================================")
    
    os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)
    
    # 1. Raccolta Dati
    all_runs_data = []
    search_pattern = os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"Nessun file CSV trovato in {BASE_INPUT_DIR}")
        return

    print(f"Trovati {len(csv_files)} file di risultati.")

    for filepath in csv_files:
        n_frames, n_clips, run_id = extract_params_from_path(filepath)
        if n_frames is None or n_clips is None: continue
            
        try:
            df = pd.read_csv(filepath)
            
            # Creiamo un dizionario base con i metadati
            run_summary = {
                "Frames": n_frames,
                "Clips": n_clips,
                "Run_ID": run_id,
                "Num_Videos": len(df)
            }
            
            # Aggiungiamo le medie per TUTTE le colonne numeriche di interesse
            # Cerca tutte le colonne che iniziano con LLM_ o RND_
            for col in df.columns:
                if col.startswith("LLM_") or col.startswith("RND_") or col.startswith("Delta_"):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        run_summary[col] = df[col].mean()

            all_runs_data.append(run_summary)
            
        except Exception as e:
            print(f"Errore lettura {filepath}: {e}")

    df_runs = pd.DataFrame(all_runs_data)
    
    if df_runs.empty:
        print(f"Nessun dato valido estratto.")
        return

    # Salvataggio CSV Riassuntivo Grezzo (Medie per Run)
    raw_summary_path = os.path.join(OUTPUT_ANALYSIS_DIR, "runs_summary_raw.csv")
    df_runs.to_csv(raw_summary_path, index=False)
    print(f"\n Tabella raw salvata: {raw_summary_path}")

    # ==============================================================================
    # 📈 GENERAZIONE LOOP GRAFICI
    # ==============================================================================
    sns.set_theme(style="whitegrid")
    
    for metric_col, label_text, random_col in METRICS_TO_ANALYZE:
        
        # Verifica se la colonna esiste nei dati (magari Reduction Tau non c'è in vecchi csv)
        if metric_col not in df_runs.columns:
            print(f"Skipping {metric_col}: colonna non trovata.")
            continue

        print(f"🎨 Generazione grafici per: {metric_col}...")
        
        # --- GRAFICO A: Performance Assoluta ---
        plt.figure(figsize=(10, 6))
        
        sns.lineplot(
            data=df_runs, 
            x="Clips", 
            y=metric_col, 
            hue="Frames", 
            style="Frames",
            palette="viridis", 
            markers=True, 
            linewidth=2.5,
            err_style="bars",
            errorbar=("sd", 1)
        )
        
        # Aggiungi Random Baseline se disponibile e presente
        if random_col and random_col in df_runs.columns:
            rnd_baseline = df_runs.groupby("Clips")[random_col].mean().reset_index()
            plt.plot(rnd_baseline["Clips"], rnd_baseline[random_col], '--', color='gray', label='Random Baseline', alpha=0.7)

        plt.title(f"{DESCRIPTION}", fontsize=15)
        plt.xlabel("Number of Clips", fontsize=12)
        plt.ylabel(label_text, fontsize=12)
        plt.legend(title="Frames")
        
        # Se è una Accuracy o Tau standardizzato, fissa i limiti
        if "Acc" in metric_col or "Tau" in metric_col:
            plt.ylim(0, 1.05) # Tau normalizzato 0-1

        filename_abs = f"plot_{metric_col}_absolute.png"
        plt.savefig(os.path.join(OUTPUT_ANALYSIS_DIR, filename_abs), dpi=300, bbox_inches='tight')
        plt.close()

        # --- GRAFICO B: Delta (Solo se esiste la colonna Delta corrispondente) ---
        # Cerchiamo se esiste una colonna Delta_... corrispondente
        # Es: LLM_Closure_Acc -> Delta_Closure_Acc
        # Assumiamo la naming convention: Delta_X = LLM_X - RND_X
        expected_delta_col = metric_col.replace("LLM_", "Delta_")
        
        if expected_delta_col in df_runs.columns:
            plt.figure(figsize=(10, 6))
            
            sns.lineplot(
                data=df_runs, 
                x="Clips", 
                y=expected_delta_col, 
                hue="Frames", 
                style="Frames",
                palette="magma", 
                markers=True, 
                linewidth=2.5,
                err_style="bars"
            )
            
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.title(f"Improvement over Random: {label_text}", fontsize=15)
            plt.xlabel("Number of Clips", fontsize=12)
            plt.ylabel(f"Delta {label_text}", fontsize=12)
            
            filename_delta = f"plot_{metric_col}_delta.png"
            plt.savefig(os.path.join(OUTPUT_ANALYSIS_DIR, filename_delta), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"\n🎉 ANALISI FULL METRICS COMPLETATA! Controlla: {OUTPUT_ANALYSIS_DIR}")

if __name__ == "__main__":
    main()