import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
#  CONFIGURAZIONE
# ==============================================================================
BASE_INPUT_DIR = "output_grid_internvl_com" # Assicurati sia la cartella giusta
OUTPUT_DIR = os.path.join(BASE_INPUT_DIR, "global_leaderboard_recipes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. CARICAMENTO DATI E PULIZIA NOMI
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
    
    # --- IL TRUCCO È QUI ---
    # In COMKitchens l'activity è salvata come "RecipeID_KitchenID" (es. "omelet_101").
    # Usiamo rsplit per staccare l'ultimo pezzo (KitchenID) e tenere solo la ricetta.
    # Poi sostituiamo eventuali altri underscore con spazi e mettiamo l'iniziale maiuscola.
    full_df['recipe_name'] = full_df['activity'].apply(
        lambda x: str(x).rsplit('_', 1)[0].replace('_', ' ').title() if '_' in str(x) else str(x).title()
    )
    
    print(f"Loaded {len(full_df)} total results.")
    print(f"Found {full_df['recipe_name'].nunique()} unique recipes starting from {full_df['activity'].nunique()} videos.")
    
    return full_df

# ==============================================================================
# 2. CALCOLO CLASSIFICHE AGGREGATE PER RICETTA
# ==============================================================================
def calculate_leaderboard(df):
    # Raggruppiamo per il NUOVO campo 'recipe_name'
    leaderboard = df.groupby('recipe_name').agg(
        Global_Acc=('LLM_Closure_Acc', 'mean'),
        Mean_Delta=('Delta_Closure_Acc', 'mean'),
        Win_Rate_Pct=('Delta_Closure_Acc', lambda x: (x > 0).mean() * 100),
        Total_Runs=('video_id', 'count')
    ).reset_index()

    # Ordiniamo per Delta 
    leaderboard = leaderboard.sort_values(by='Mean_Delta', ascending=False)
    return leaderboard

# ==============================================================================
# 3. VISUALIZZAZIONE
# ==============================================================================
def plot_comprehensive_leaderboard(leaderboard):
    num_activities = len(leaderboard)
    print(f"Generating graph for {num_activities} grouped recipes...")

    # Altezza dinamica ma più contenuta ora che abbiamo meno righe
    fig_height = max(10, num_activities * 0.35)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, fig_height), sharey=True)
    sns.set_theme(style="whitegrid")
    
    # --- PLOT 1: Global Accuracy ---
    sns.barplot(
        data=leaderboard, y='recipe_name', x='Global_Acc', ax=axes[0],
        palette="viridis", hue='recipe_name', legend=False
    )
    axes[0].set_title("1. Global Accuracy", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Accuracy (0-1)", fontsize=14)
    axes[0].set_xlim(0, 1.0)
    axes[0].axvline(0.5, color='red', linestyle='--', alpha=0.5)

    # --- PLOT 2: Mean Delta ---
    colors_delta = ['#2ecc71' if x > 0 else '#e74c3c' for x in leaderboard['Mean_Delta']]
    sns.barplot(
        data=leaderboard, y='recipe_name', x='Mean_Delta', ax=axes[1],
        palette=colors_delta, hue='recipe_name', legend=False
    )
    axes[1].set_title("2. Skill vs Random (Mean Delta)", fontsize=16, fontweight='bold')
    axes[1].set_xlabel("Delta (Model - Random)", fontsize=14)
    axes[1].axvline(0, color='black', linewidth=1) 
    axes[1].set_ylabel("") 

    # --- PLOT 3: Win Rate ---
    sns.barplot(
        data=leaderboard, y='recipe_name', x='Win_Rate_Pct', ax=axes[2],
        palette="magma", hue='recipe_name', legend=False
    )
    axes[2].set_title("3. Consistency (Win Rate %)", fontsize=16, fontweight='bold')
    axes[2].set_xlabel("% of Runs Model > Random", fontsize=14)
    axes[2].set_xlim(0, 100)
    axes[2].axvline(50, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel("")

    for i in axes[2].containers:
        axes[2].bar_label(i, fmt='%.0f%%', padding=3, fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "recipe_leaderboard_COM.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    df = load_all_data()
    
    if not df.empty:
        lb = calculate_leaderboard(df)
        
        csv_path = os.path.join(OUTPUT_DIR, "recipe_ranking_summary.csv")
        lb.round(4).to_csv(csv_path, index=False)
        
        print("\n TOP 5 Recipes (Best Delta):")
        print(lb[['recipe_name', 'Mean_Delta', 'Win_Rate_Pct', 'Total_Runs']].head(5).to_string(index=False))
        
        print("\n BOTTOM 5 Recipes (Worst Delta):")
        print(lb[['recipe_name', 'Mean_Delta', 'Win_Rate_Pct', 'Total_Runs']].tail(5).to_string(index=False))
        
        plot_comprehensive_leaderboard(lb)
        print(f"\n All completed! Files in: {OUTPUT_DIR}")