import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
#  CONFIGURAZIONE ESPERIMENTI BATCH
# ==============================================================================
# Lista delle cartelle da analizzare e le rispettive descrizioni
EXPERIMENTS = [
    ("output_grid_experiment_captaincook", "CaptainCook4D - Qwen2.5-VL"),
    ("output_grid_experiment_com", "COM Kitchens - Qwen2.5-VL"),
    ("output_grid_internvl_captaincook", "CaptainCook4D - InternVL3.5"),
    ("output_grid_internvl_com", "COM Kitchens - InternVL3.5")
]

# Lista delle metriche da analizzare (coppia: NomeColonnaCSV, EtichettaLeggibile, ColonnaBaseline)
METRICS_TO_ANALYZE = [
    ("LLM_Closure_Acc", "Pairwise Accuracy (Closure)", "RND_Closure_Acc"),
    ("LLM_Closure_Tau", "Kendall's Tau (Closure)", None),
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

def process_experiment(base_input_dir, description):
    print(f"\n{'='*60}")
    print(f"INIZIO ANALISI: {description}")
    print(f"Cartella base: {base_input_dir}")
    print(f"{'='*60}")
    
    output_analysis_dir = os.path.join(base_input_dir, "analysis_frames_evolution")
    os.makedirs(output_analysis_dir, exist_ok=True)
    
    # 1. Raccolta Dati
    all_runs_data = []
    search_pattern = os.path.join(base_input_dir, "**", "*_metrics.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"Nessun file CSV trovato in {base_input_dir}. Salto al prossimo...")
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

    # Salvataggio CSV Riassuntivo Grezzo
    raw_summary_path = os.path.join(output_analysis_dir, "runs_summary_raw.csv")
    df_runs.to_csv(raw_summary_path, index=False)
    print(f"Tabella raw salvata in: {raw_summary_path}")

    # ==============================================================================
    # 📈 GENERAZIONE LOOP GRAFICI (INVERTITI: X=Frames, Hue=Clips)
    # ==============================================================================
    sns.set_theme(style="whitegrid")
    
    for metric_col, label_text, random_col in METRICS_TO_ANALYZE:
        
        if metric_col not in df_runs.columns:
            print(f"Skipping {metric_col}: colonna non trovata.")
            continue

        print(f"🎨 Generazione grafici per: {metric_col}...")
        
        # --- GRAFICO A: Performance Assoluta ---
        plt.figure(figsize=(10, 6))
        
        # Invertiti x con hue/style e cambiata la palette in 'husl'
        sns.lineplot(
            data=df_runs, 
            x="Frames", 
            y=metric_col, 
            hue="Clips", 
            style="Clips",
            palette="husl", 
            markers=True, 
            linewidth=2.5,
            err_style="bars",
            errorbar=("sd", 1)
        )
        
        # Aggiungi Random Baseline raggruppata per Frames (invece di Clips)
        if random_col and random_col in df_runs.columns:
            rnd_baseline = df_runs.groupby("Frames")[random_col].mean().reset_index()
            plt.plot(rnd_baseline["Frames"], rnd_baseline[random_col], '--', color='gray', label='Random Baseline', alpha=0.7)

        plt.title(f"{description}", fontsize=15)
        plt.xlabel("Number of Frames", fontsize=12) # Etichetta aggiornata
        plt.ylabel(label_text, fontsize=12)
        plt.legend(title="Clips") # Legenda aggiornata
        
        if "Acc" in metric_col or "Tau" in metric_col:
            plt.ylim(0, 1.05) 

        filename_abs = f"plot_{metric_col}_absolute_by_frames.png"
        plt.savefig(os.path.join(output_analysis_dir, filename_abs), dpi=300, bbox_inches='tight')
        plt.close()

        # --- GRAFICO B: Delta ---
        expected_delta_col = metric_col.replace("LLM_", "Delta_")
        
        if expected_delta_col in df_runs.columns:
            plt.figure(figsize=(10, 6))
            
            # Invertiti x con hue/style e cambiata la palette in 'mako'
            sns.lineplot(
                data=df_runs, 
                x="Frames", 
                y=expected_delta_col, 
                hue="Clips", 
                style="Clips",
                palette="mako", 
                markers=True, 
                linewidth=2.5,
                err_style="bars"
            )
            
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.title(f"Improvement over Random: {label_text}\n({description})", fontsize=15)
            plt.xlabel("Number of Frames", fontsize=12) # Etichetta aggiornata
            plt.ylabel(f"Delta {label_text}", fontsize=12)
            plt.legend(title="Clips") # Legenda aggiornata
            
            filename_delta = f"plot_{metric_col}_delta_by_frames.png"
            plt.savefig(os.path.join(output_analysis_dir, filename_delta), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"🎉 Analisi per {description} COMPLETATA! Output in: {output_analysis_dir}")


def main():
    print("==================================================")
    print(f"STARTING BATCH METRIC ANALYSIS (BY FRAMES)")
    print("==================================================")
    
    # Esegue il ciclo per ciascun esperimento definito
    for base_dir, desc in EXPERIMENTS:
        if os.path.exists(base_dir):
            process_experiment(base_dir, desc)
        else:
            print(f"\n ATTENZIONE: La cartella '{base_dir}' non esiste o non si trova in questa directory. L'esperimento '{desc}' verrà saltato.")
            
    print("\n TUTTI GLI ESPERIMENTI DISPONIBILI SONO STATI PROCESSATI! ")


if __name__ == "__main__":
    main()