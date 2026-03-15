import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

"""
python scripts/visualization/visualize_grid_results.py
python scripts/visualization/visualize_accuracy_bands.py
python scripts/visualization/visualize_model_vs_random.py
python scripts/visualization/visualize_deep_insights.py
python scripts/visualization/visualize_complexity_impact.py
python scripts/visualization/visualize_accuracy_split.py
python scripts/visualization/visualize_grid_result_FULL.py
python scripts/visualization/analyze_global_leaderboard.py
python scripts/visualization/analyze_v2_categories.py
"""

BASE_INPUT_DIR = "output_grid_internvl_captaincook"
OUTPUT_ANALYSIS_DIR = os.path.join(BASE_INPUT_DIR, "analysis")

# Metrica principale da graficare
MAIN_METRIC = "LLM_Closure_Acc"
MAIN_METRIC_LABEL = "Pairwise Accuracy (Closure)"

# ==============================================================================
#  LOGICA DI ANALISI
# ==============================================================================

def extract_params_from_path(filepath):
    """
    Estrae n_frames e n_clips dal percorso del file.
    Es: output_grid_experiment/frames_8/clips_5/run_0_metrics.csv
    """
    parts = filepath.split(os.sep)
    
    n_frames = None
    n_clips = None
    run_id = None
    
    for part in parts:
        if part.startswith("frames_"):
            n_frames = int(part.split("_")[1])
        elif part.startswith("clips_"):
            n_clips = int(part.split("_")[1])
            
    filename = os.path.basename(filepath)
    if filename.startswith("run_") and "_metrics.csv" in filename:
        run_id = int(filename.split("_")[1])
        
    return n_frames, n_clips, run_id

def main():
    print("==================================================")
    print(f"STARTING ANALYSIS")
    print("==================================================")
    
    # 1. Raccolta Dati
    all_runs_data = []
    
    # Cerca tutti i file metrics.csv ricorsivamente
    search_pattern = os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"Nessun file CSV trovato in {BASE_INPUT_DIR}")
        return

    print(f"Trovati {len(csv_files)} file di risultati.")

    for filepath in csv_files:
        n_frames, n_clips, run_id = extract_params_from_path(filepath)
        
        if n_frames is None or n_clips is None:
            continue
            
        try:
            df = pd.read_csv(filepath)
            
            # Calcoliamo la media delle metriche per QUESTA specifica run
            # (Media su tutti i video di questa run)
            run_summary = {
                "Frames": n_frames,
                "Clips": n_clips,
                "Run_ID": run_id,
                # Metriche LLM
                "LLM_Acc": df["LLM_Closure_Acc"].mean(),
                "LLM_Tau": df["LLM_Closure_Tau"].mean(),
                # Metriche Random (Baseline)
                "RND_Acc": df["RND_Closure_Acc"].mean(),
                # Delta (Quanto il modello è meglio del caso)
                "Delta_Acc": df["Delta_Closure_Acc"].mean(),
                "Num_Videos": len(df)
            }
            all_runs_data.append(run_summary)
            
        except Exception as e:
            print(f"Errore lettura {filepath}: {e}")

    # Creazione DataFrame Master
    df_runs = pd.DataFrame(all_runs_data)
    
    if df_runs.empty:
        print(f"Nessun dato valido estratto.")
        return

    # 2. Aggregazione (Media tra le Run)
    # Raggruppiamo per Frames e Clips, calcolando media e dev. standard tra le diverse Run
    df_grouped = df_runs.groupby(["Frames", "Clips"]).agg(
        Mean_LLM_Acc=("LLM_Acc", "mean"),
        Std_LLM_Acc=("LLM_Acc", "std"),
        Mean_Delta_Acc=("Delta_Acc", "mean"),
        Mean_RND_Acc=("RND_Acc", "mean")
    ).reset_index()

    # Creazione cartella output
    os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)
    
    # Salvataggio CSV Riassuntivo
    summary_csv_path = os.path.join(OUTPUT_ANALYSIS_DIR, "global_summary_table.csv")
    df_grouped.to_csv(summary_csv_path, index=False)
    print(f"\n Tabella riassuntiva salvata: {summary_csv_path}")
    print(df_grouped.to_string())

    # ==============================================================================
    # 📈 GENERAZIONE GRAFICI
    # ==============================================================================
    sns.set_theme(style="whitegrid")
    
    # --- GRAFICO 1: Performance Assoluta (Clips vs Accuracy) ---
    plt.figure(figsize=(10, 6))
    
    # Usiamo df_runs (i dati grezzi delle run) così Seaborn calcola automaticamente 
    # le bande di confidenza (o barre di errore)
    sns.lineplot(
        data=df_runs, 
        x="Clips", 
        y="LLM_Acc", 
        hue="Frames", 
        style="Frames",
        palette="viridis", 
        markers=True, 
        dashes=False,
        linewidth=2.5,
        err_style="bars", # Barre di errore basate sulla varianza tra le run
        errorbar=("sd", 1) # Standard Deviation
    )
    
    # Aggiungi linea baseline random (media approssimativa)
    # Prendiamo la media della random baseline per ogni numero di clip
    rnd_baseline = df_runs.groupby("Clips")["RND_Acc"].mean().reset_index()
    plt.plot(rnd_baseline["Clips"], rnd_baseline["RND_Acc"], '--', color='gray', label='Random Baseline', alpha=0.7)

    plt.title("Model Performance: Sequence Ordering", fontsize=16)
    plt.xlabel("Number of Clips (Sequence Length)", fontsize=12)
    plt.ylabel("Pairwise Accuracy (Closure)", fontsize=12)
    plt.legend(title="Frames per Clip")
    plt.xticks(sorted(df_runs["Clips"].unique()))
    plt.ylim(0, 1.0)
    
    plot_path_1 = os.path.join(OUTPUT_ANALYSIS_DIR, "plot_performance_absolute.png")
    plt.savefig(plot_path_1, dpi=300, bbox_inches='tight')
    print(f"Grafico 1 salvato: {plot_path_1}")

    # --- GRAFICO 2: Delta Performance (Miglioramento rispetto al Random) ---
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df_runs, 
        x="Clips", 
        y="Delta_Acc", 
        hue="Frames", 
        style="Frames",
        palette="magma", 
        markers=True, 
        linewidth=2.5,
        err_style="bars"
    )
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1) # Linea dello zero (uguale al random)
    
    plt.title("Improvement over Random Guessing", fontsize=16)
    plt.xlabel("Number of Clips (Sequence Length)", fontsize=12)
    plt.ylabel("Delta Accuracy (Model - Random)", fontsize=12)
    plt.legend(title="Frames per Clip")
    plt.xticks(sorted(df_runs["Clips"].unique()))
    
    plot_path_2 = os.path.join(OUTPUT_ANALYSIS_DIR, "plot_performance_delta.png")
    plt.savefig(plot_path_2, dpi=300, bbox_inches='tight')
    print(f"Grafico 2 salvato: {plot_path_2}")

    print("\n🎉 ANALISI COMPLETATA!")

if __name__ == "__main__":
    main()