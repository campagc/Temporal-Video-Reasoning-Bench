import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
#  CONFIGURAZIONE
# ==============================================================================
BASE_INPUT_DIR = "output_grid_internvl_captaincook" # Ensure that sia la cartella giusta
OUTPUT_PLOT_DIR = os.path.join(BASE_INPUT_DIR, "analysis_categories_2")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# Definiamo le fasce di accuratezza
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
LABELS = ['0-20% (Pessimo)', '20-40% (Basso)', '40-60% (Random/Medio)', '60-80% (Buono)', '80-100% (Eccellente)']
COLORS = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"] 

# --- 🗺️ MAPPING CATEGORIE (Normalizzato: lowercase, no spaces) ---
# Ho convertito la tua lista nel formato che probabilmente è nel CSV

'''
# HIGHT VISUAL CHANGE
CATEGORY_MAP = {
    # HIGH CHANGE
    "tomatomozzarellasalad": "High Visual Change",
    "panfriedtofu": "High Visual Change",
    "spicytunaavocadowraps": "High Visual Change",
    "breakfastburritos": "High Visual Change",
    "capresebruschetta": "High Visual Change",
    "microwaveeggsandwich": "High Visual Change",

    # LOW CHANGE
    "cucumberraita": "Low Visual Change",
    "broccolistirfry": "Low Visual Change",
    "ramen": "Low Visual Change",
    "microwavefrenchtoast": "Low Visual Change",
    "coffee": "Low Visual Change",
    "mugcake": "Low Visual Change",
    "blenderbananapancakes": "Low Visual Change",
    "spicedhotchocolate": "Low Visual Change",
    "herbomeletwithfriedtomatoes": "Low Visual Change",
    "scrambledeggs": "Low Visual Change",
    "cheesepimiento": "Low Visual Change",
    "microwavemugpizza": "Low Visual Change",
    "buttercorncup": "Low Visual Change",
    "tomatochutney": "Low Visual Change",
    "zoodles": "Low Visual Change"
}

'''

'''
# MORE TO COLOR
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
'''


# ==============================================================================
# 1. CARICAMENTO E PREPARAZIONE DATI
# ==============================================================================
def extract_params(filepath):
    parts = filepath.split(os.sep)
    n_frames = next((int(p.split('_')[1]) for p in parts if 'frames_' in p), None)
    n_clips = next((int(p.split('_')[1]) for p in parts if 'clips_' in p), None)
    return n_frames, n_clips

def load_data():
    all_files = glob.glob(os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv"), recursive=True)
    df_list = []
    
    print(f"Trovati {len(all_files)} file CSV.")
    
    for f in all_files:
        frames, clips = extract_params(f)
        if frames is None or clips is None: continue
        
        try:
            temp_df = pd.read_csv(f)
            temp_df['Frames'] = frames
            temp_df['Clips'] = clips
            temp_df['Config_Label'] = f"{frames}F / {clips}C"
            
            # --- APPLICAZIONE CATEGORIA ---
            # Normalizziamo la colonna activity per il match (tutto minuscolo, niente spazi)
            temp_df['activity_norm'] = temp_df['activity'].astype(str).str.lower().str.replace(" ", "").str.replace("_", "")
            temp_df['Category'] = temp_df['activity_norm'].map(CATEGORY_MAP)
            
            # Fallback per activities non trovate
            temp_df['Category'] = temp_df['Category'].fillna('Uncategorized')
            
            df_list.append(temp_df)
        except Exception as e:
            print(f"Errore lettura {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

# ==============================================================================
# 2. VISUALIZZAZIONE COMPARATIVA
# ==============================================================================

def plot_side_by_side_distribution(df):
    """Genera due grafici affiancati: High vs Low Change"""
    
    # Filtriamo solo le categorie note
    df_clean = df[df['Category'] != 'Uncategorized'].copy()
    
    if df_clean.empty:
        print(f"Nessun dato categorizzato trovato. Controlla i nomi delle activities nel CSV.")
        return

    # Binning
    df_clean['Accuracy_Band'] = pd.cut(
        df_clean['LLM_Closure_Acc'], 
        bins=BINS, 
        labels=LABELS, 
        right=False, 
        include_lowest=True
    )
    # Fix per valore 1.0 esatto
    df_clean.loc[df_clean['LLM_Closure_Acc'] >= 0.8, 'Accuracy_Band'] = LABELS[-1]

    categories = ["High Visual Change", "Low Visual Change"]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    
    for i, cat in enumerate(categories):
        ax = axes[i]
        subset = df_clean[df_clean['Category'] == cat]
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        # Aggregazione
        grouped = subset.groupby(['Config_Label', 'Accuracy_Band'], observed=False).size().reset_index(name='Counts')
        pivot_df = grouped.pivot(index='Config_Label', columns='Accuracy_Band', values='Counts').fillna(0)
        
        # Percentuali
        pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        
        # Plot
        pivot_pct.plot(
            kind='bar', 
            stacked=True, 
            color=COLORS, 
            edgecolor='black', 
            width=0.8,
            ax=ax,
            legend=False
        )
        
        ax.set_title(f"{cat}\n(N={len(subset)})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Configurazione", fontsize=12)
        if i == 0:
            ax.set_ylabel("Percentuale Video (%)", fontsize=12)
        
        # Labels sulle barre
        for c in ax.containers:
            ax.bar_label(c, fmt='%.0f%%', label_type='center', color='black', fontsize=8, padding=2)

    # Legenda unica
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Accuratezza", loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # Spazio per la legenda
    
    out_path = os.path.join(OUTPUT_PLOT_DIR, "compare_high_vs_low.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Grafico comparativo salvato: {out_path}")

def plot_delta_comparison(df):
    """Grafico a linee del Delta Accuracy (Modello - Random) per categoria"""
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df,
        x="Config_Label",
        y="Delta_Closure_Acc",
        hue="Category",
        style="Category",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette=["#2ecc71", "#e74c3c"] # Verde per High, Rosso per Low
    )
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1, label="Random Guess Level")
    plt.title("Guadagno rispetto al caso (Delta Accuracy)", fontsize=14)
    plt.ylabel("Delta Accuracy (Positive is Good)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(OUTPUT_PLOT_DIR, "delta_trend_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Grafico trend salvato: {out_path}")

# ==============================================================================
# 3. REPORT TESTUALE
# ==============================================================================
def print_category_report(df):
    print("\n" + "="*60)
    print("📋 REPORT PER CATEGORIA VISIVA")
    print("="*60)
    
    cats = ["High Visual Change", "Low Visual Change"]
    
    for cat in cats:
        subset = df[df['Category'] == cat]
        if subset.empty: continue
        
        print(f"\n--- {cat.upper()} ---")
        
        # Raggruppa per configurazione
        configs = sorted(subset['Config_Label'].unique())
        
        for conf in configs:
            sub_conf = subset[subset['Config_Label'] == conf]
            mean_acc = sub_conf['LLM_Closure_Acc'].mean()
            mean_delta = sub_conf['Delta_Closure_Acc'].mean()
            
            # Quanti > 60% (Buoni)
            good_count = len(sub_conf[sub_conf['LLM_Closure_Acc'] >= 0.6])
            total = len(sub_conf)
            
            print(f"🔹 {conf}: Avg Acc: {mean_acc:.2f} | Delta: {mean_delta:+.2f} | Buoni: {good_count}/{total} ({good_count/total:.0%})")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"🔎 Analisi Categorie su directory: {BASE_INPUT_DIR}")
    df = load_data()
    
    if not df.empty:
        # Controllo mapping
        uncategorized = df[df['Category'] == 'Uncategorized']['activity'].unique()
        if len(uncategorized) > 0:
            print(f"Attenzione: Alcune activities non sono state mappate: {uncategorized}")
            print(f"-> Aggiungile al CATEGORY_MAP nello script.")
        
        plot_side_by_side_distribution(df)
        plot_delta_comparison(df)
        print_category_report(df)
        
        print(f"\n Analisi completata. Risultati in: {OUTPUT_PLOT_DIR}")
    else:
        print(f"Nessun dato trovato.")