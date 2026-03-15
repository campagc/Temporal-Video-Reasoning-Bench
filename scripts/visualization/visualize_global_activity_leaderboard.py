import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
#  CONFIGURAZIONE
# ==============================================================================
# Sostituisci con "output_grid_experiment_com" se hai usato Qwen invece di InternVL
BASE_INPUT_DIR = "output_grid_internvl_com" 
OUTPUT_DIR = os.path.join(BASE_INPUT_DIR, "global_leaderboard_com")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. CARICAMENTO DATI
# ==============================================================================
def load_all_data():
    print(f"Scanning directory: {BASE_INPUT_DIR}")
    # Cerca ricorsivamente tutti i file metrics.csv
    all_files = glob.glob(os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv"), recursive=True)
    
    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading file {f}: {e}")
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

    # Ordiniamo per Delta 
    leaderboard = leaderboard.sort_values(by='Mean_Delta', ascending=False)
    return leaderboard

# ==============================================================================
# 3. VISUALIZZAZIONE DINAMICA PER COMKITCHENS
# ==============================================================================
def plot_comprehensive_leaderboard(leaderboard):
    num_activities = len(leaderboard)
    print(f"Generating graph for {num_activities} activities...")

    # ALTEZZA DINAMICA: Calcoliamo l'altezza in base al numero di recipes
    # Garantiamo almeno 0.3 pollici per ricetta, con un minimo di 12 pollici
    fig_height = max(12, num_activities * 0.35)
    
    # Se ci sono tante recipes, riduciamo il font per non accavallare i testi
    tick_fontsize = 8 if num_activities > 50 else 10
    label_fontsize = 6 if num_activities > 50 else 9

    fig, axes = plt.subplots(1, 3, figsize=(24, fig_height), sharey=True)
    sns.set_theme(style="whitegrid")
    
    # --- PLOT 1: Global Accuracy (Assoluta) ---
    sns.barplot(
        data=leaderboard, y='activity', x='Global_Acc', ax=axes[0],
        palette="viridis", hue='activity', legend=False
    )
    axes[0].set_title("1. Global Accuracy", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Accuracy (0-1)", fontsize=14)
    axes[0].set_xlim(0, 1.0)
    axes[0].axvline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='y', labelsize=tick_fontsize)

    # --- PLOT 2: Mean Delta (Intelligenza vs Random) ---
    colors_delta = ['#2ecc71' if x > 0 else '#e74c3c' for x in leaderboard['Mean_Delta']]
    sns.barplot(
        data=leaderboard, y='activity', x='Mean_Delta', ax=axes[1],
        palette=colors_delta, hue='activity', legend=False
    )
    axes[1].set_title("2. Skill vs Random (Mean Delta)", fontsize=16, fontweight='bold')
    axes[1].set_xlabel("Delta (Model - Random)", fontsize=14)
    axes[1].axvline(0, color='black', linewidth=1) 
    axes[1].set_ylabel("") 

    # --- PLOT 3: Win Rate (Consistenza) ---
    sns.barplot(
        data=leaderboard, y='activity', x='Win_Rate_Pct', ax=axes[2],
        palette="magma", hue='activity', legend=False
    )
    axes[2].set_title("3. Consistency (Win Rate %)", fontsize=16, fontweight='bold')
    axes[2].set_xlabel("% of Runs Model > Random", fontsize=14)
    axes[2].set_xlim(0, 100)
    axes[2].axvline(50, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel("")

    # Etichette percentuali sul grafico 3 (con dimensione dinamica)
    for i in axes[2].containers:
        axes[2].bar_label(i, fmt='%.0f%%', padding=3, fontsize=label_fontsize)

    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "comprehensive_leaderboard_ALL.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive graph saved to: {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    df = load_all_data()
    
    if not df.empty:
        lb = calculate_leaderboard(df)
        
        # Saving CSV to insert tables in LaTeX
        csv_path = os.path.join(OUTPUT_DIR, "activity_ranking_summary.csv")
        lb.round(4).to_csv(csv_path, index=False)
        
        print("\n TOP 10 Activities (Best Delta):")
        print(lb[['activity', 'Mean_Delta', 'Win_Rate_Pct']].head(10).to_string(index=False))
        
        print("\n BOTTOM 10 Activities (Worst Delta):")
        print(lb[['activity', 'Mean_Delta', 'Win_Rate_Pct']].tail(10).to_string(index=False))
        
        plot_comprehensive_leaderboard(lb)
        print(f"\n Analysis completed! Files exported to: {OUTPUT_DIR}")
    else:
        print(f"Data not found. Check BASE_INPUT_DIR.")