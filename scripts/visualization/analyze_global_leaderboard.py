import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
#  CONFIGURAZIONE
# ==============================================================================
BASE_INPUT_DIR = "output_grid_experiment_captaincook" 
OUTPUT_DIR = os.path.join(BASE_INPUT_DIR, "global_leaderboard")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- TITOLO GLOBALE DEL GRAFICO ---
DESCRIPTION = "Qwen2.5-VL - Captaincook4d"

# ==============================================================================
# 1. CARICAMENTO DATI
# ==============================================================================
def load_all_data():
    print(f"Scanning directory: {BASE_INPUT_DIR}")
    all_files = glob.glob(os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv"), recursive=True)
    
    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except Exception as e:
            pass

    if not df_list:
        print(f"No files found.")
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(full_df)} total results.")
    return full_df

# ==============================================================================
# 2. CALCOLO CLASSIFICHE
# ==============================================================================
def calculate_leaderboard(df):
    leaderboard = df.groupby('activity').agg(
        Global_Acc=('LLM_Closure_Acc', 'mean'),
        Mean_Delta=('Delta_Closure_Acc', 'mean'),
        Win_Rate_Pct=('Delta_Closure_Acc', lambda x: (x > 0).mean() * 100),
        Total_Runs=('video_id', 'count')
    ).reset_index()

    # Calcolo del Random reale medio
    leaderboard['Actual_Random'] = leaderboard['Global_Acc'] - leaderboard['Mean_Delta']

    # Ordiniamo per Delta
    leaderboard = leaderboard.sort_values(by='Mean_Delta', ascending=False)
    
    return leaderboard

# ==============================================================================
# 3. VISUALIZZAZIONE
# ==============================================================================
def plot_comprehensive_leaderboard(leaderboard):
    # Impostiamo un layout con 3 grafici affiancati
    fig, axes = plt.subplots(1, 3, figsize=(24, 12), sharey=True)
    sns.set_theme(style="whitegrid")

    # --- AGGIUNTA TITOLO GLOBALE ---
    fig.suptitle(DESCRIPTION, fontsize=22, fontweight='bold', y=0.98)

    # --- PLOT 1: Global Accuracy vs ACTUAL Random ---
    sns.barplot(
        data=leaderboard,
        y='activity',
        x='Global_Acc',
        ax=axes[0],
        palette="viridis",
        hue='activity', legend=False
    )
    
    # Aggiungiamo i marcatori del random specifico per ogni activities (Rombi Rossi)
    y_coords = range(len(leaderboard))
    axes[0].scatter(
        x=leaderboard['Actual_Random'], 
        y=y_coords, 
        color='red', 
        marker='D',   
        s=70,         
        zorder=5,     
        label='Actual Random Baseline'
    )

    axes[0].set_title("1. Accuracy vs Activity-Specific Random", fontsize=15, fontweight='bold')
    axes[0].set_xlabel("Accuracy (0-1)", fontsize=12)
    axes[0].set_xlim(0, 1.0)
    axes[0].legend(loc='lower right', frameon=True)

    # --- PLOT 2: Mean Delta ---
    colors_delta = ['#2ecc71' if x > 0 else '#e74c3c' for x in leaderboard['Mean_Delta']]
    sns.barplot(
        data=leaderboard,
        y='activity',
        x='Mean_Delta',
        ax=axes[1],
        palette=colors_delta,
        hue='activity', legend=False
    )
    axes[1].set_title("2. Skill vs Random (Mean Delta)", fontsize=15, fontweight='bold')
    axes[1].set_xlabel("Delta (Model - Random)", fontsize=12)
    axes[1].axvline(0, color='black', linewidth=1.5)
    axes[1].set_ylabel("")

    # --- PLOT 3: Win Rate ---
    sns.barplot(
        data=leaderboard,
        y='activity',
        x='Win_Rate_Pct',
        ax=axes[2],
        palette="magma",
        hue='activity', legend=False
    )
    axes[2].set_title("3. Consistency (Win Rate % vs Random)", fontsize=15, fontweight='bold')
    axes[2].set_xlabel("% of Runs where Model > Random", fontsize=12)
    axes[2].set_xlim(0, 100)
    axes[2].axvline(50, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel("")

    for i in axes[2].containers:
        axes[2].bar_label(i, fmt='%.0f%%', padding=3)

    # Aggiusta il layout per fare spazio al titolo globale in alto
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "comprehensive_leaderboard_v2.png")
    plt.savefig(out_path, dpi=300)
    print(f"Grafico salvato: {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    df = load_all_data()

    if not df.empty:
        lb = calculate_leaderboard(df)
        
        csv_path = os.path.join(OUTPUT_DIR, "activity_ranking_summary.csv")
        lb.round(4).to_csv(csv_path, index=False)

        print(f"\n--- {DESCRIPTION} ---")
        print("\n TOP 5 Activities (Best Delta):")
        print(lb[['activity', 'Global_Acc', 'Actual_Random', 'Mean_Delta']].head(5).to_string(index=False))

        plot_comprehensive_leaderboard(lb)
        print(f"Analisi completata. File in: {OUTPUT_DIR}")
    else:
        print(f"Dati non trovati.")