import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
#  CONFIGURAZIONE GLOBALE
# ==============================================================================
BASE_INPUT_DIR = "output_grid_experiment_captaincook"
OUTPUT_PLOT_DIR = os.path.join(BASE_INPUT_DIR, "analysis_categories_3")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# 📝 VARIABILE PER IL TITOLO DEL GRAFICO
CHART_TITLE = "Trend Delta Accuracy (6 Frames) - Qwen2.5-VL"

# ==============================================================================
# 🗺️ CATEGORY MAP
# ==============================================================================
CATEGORY_MAP = {
    "panfriedtofu": "Low Visual Change",
    "spicedhotchocolate": "Low Visual Change",
    "spicytunaavocadowraps": "High Visual Change",
    "breakfastburritos": "Low Visual Change",
    "tomatochutney": "Low Visual Change",
    "microwaveeggsandwich": "Low Visual Change",
    "cucumberraita": "Low Visual Change",
    "broccolistirfry": "Low Visual Change",
    "ramen": "Low Visual Change",
    "microwavefrenchtoast": "Low Visual Change",
    "coffee": "Low Visual Change",
    "mugcake": "Low Visual Change",
    "tomatomozzarellasalad": "High Visual Change",
    "blenderbananapancakes": "Low Visual Change",
    "herbomeletwithfriedtomatoes": "Low Visual Change",
    "scrambledeggs": "Low Visual Change",
    "cheesepimiento": "Low Visual Change",
    "capresebruschetta": "High Visual Change",
    "microwavemugpizza": "Low Visual Change",
    "buttercorncup": "Low Visual Change",
    "zoodles": "High Visual Change"
}

# ==============================================================================
# 1. CARICAMENTO E FILTRAGGIO DATI
# ==============================================================================
def extract_params(filepath):
    parts = filepath.split(os.sep)
    # Cerchiamo le cartelle frames_X e clips_X
    n_frames = None
    n_clips = None
    for p in parts:
        if 'frames_' in p:
            n_frames = int(p.split('_')[1])
        if 'clips_' in p:
            n_clips = int(p.split('_')[1])
    return n_frames, n_clips

def load_data_6_frames_only():
    all_files = glob.glob(os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv"), recursive=True)
    df_list = []

    for f in all_files:
        frames, clips = extract_params(f)
        
        # 🛑 FILTRO: Solo configurazioni con 6 frames
        if frames != 6 or clips is None:
            continue

        try:
            temp_df = pd.read_csv(f)
            temp_df['Frames'] = frames
            temp_df['Clips'] = clips
            
            # Normalizzazione nomi per mapping
            temp_df['activity_norm'] = (
                temp_df['activity']
                .astype(str)
                .str.lower()
                .str.replace(" ", "", regex=False)
                .str.replace("_", "", regex=False)
            )

            temp_df['Category'] = temp_df['activity_norm'].map(CATEGORY_MAP)
            temp_df['Category'] = temp_df['Category'].fillna('Uncategorized')

            df_list.append(temp_df)
        except Exception as e:
            print(f"Errore lettura {f}: {e}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)

# ==============================================================================
# 2. GENERAZIONE GRAFICO
# ==============================================================================
def plot_delta_comparison(df):
    # --- FIX SINTASSI: Filtro corretto ---
    df_clean = df[df['Category'] != 'Uncategorized'].copy()

    if df_clean.empty:
        print(f"Nessun dato categorizzato disponibile per il grafico.")
        return

    # Ordiniamo per Clips (asse X progressivo)
    df_clean = df_clean.sort_values(by='Clips')

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    custom_palette = {
        'High Visual Change': '#2ecc71',
        'Low Visual Change': '#e74c3c'
    }

    # Plot del trend
    sns.lineplot(
        data=df_clean,
        x="Clips",
        y="Delta_Closure_Acc",
        hue="Category",
        marker="o",
        markersize=10,
        linewidth=3,
        palette=custom_palette
    )

    # Titolo e Label
    plt.title(CHART_TITLE, fontsize=18, fontweight='bold', pad=20)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    
    plt.xlabel("Number of Clips", fontsize=14, fontweight='bold')
    plt.ylabel("Delta Accuracy (Model - Random)", fontsize=14, fontweight='bold')
    
    # Tick asse X precisi
    unique_clips = sorted(df_clean['Clips'].unique())
    plt.xticks(unique_clips)

    plt.legend(title="Complexity Category", fontsize=12, title_fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_PLOT_DIR, "delta_trend_comparison_6F.png")
    plt.savefig(out_path, dpi=300)
    print(f"Graph saved to: {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"🔎 Analisi in corso su: {BASE_INPUT_DIR}")
    
    df_final = load_data_6_frames_only()

    if not df_final.empty:
        print(f"Loaded {len(df_final)} record.")
        plot_delta_comparison(df_final)
    else:
        print(f"Nessun dato trovato per 6 frames.")