import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
#  CONFIGURAZIONE
# ==============================================================================
BASE_INPUT_DIR = "output_grid_internvl_captaincook"
OUTPUT_PLOT_DIR = os.path.join(BASE_INPUT_DIR, "plots_distribution_split")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# Fasce di accuratezza (Le stesse di prima)
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
LABELS = ['0-20% (Pessimo)', '20-40% (Basso)', '40-60% (Medio)', '60-80% (Buono)', '80-100% (Eccellente)']
# Colori: Rosso -> Giallo -> Verde
COLORS = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"] 

# ==============================================================================
# CARICAMENTO DATI (Robustezza V2)
# ==============================================================================
def extract_params(filepath):
    parts = filepath.split(os.sep)
    n_frames = None
    n_clips = None
    for part in parts:
        if part.startswith("frames_") and part[7:].isdigit():
            n_frames = int(part.split("_")[1])
        elif part.startswith("clips_") and part[6:].isdigit():
            n_clips = int(part.split("_")[1])
    return n_frames, n_clips

def load_data():
    print(f"Scansione: {os.path.abspath(BASE_INPUT_DIR)}")
    all_files = glob.glob(os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv"), recursive=True)
    df_list = []
    
    for f in all_files:
        frames, clips = extract_params(f)
        if frames is None or clips is None: continue
        try:
            temp_df = pd.read_csv(f)
            temp_df['Frames'] = frames
            temp_df['Clips'] = clips
            df_list.append(temp_df)
        except: pass
            
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# ==============================================================================
#  GENERAZIONE GRAFICI SPLITTATI
# ==============================================================================
def plot_split_by_clips(df):
    # Prepara le fasce per tutto il dataset
    df['Accuracy_Band'] = pd.cut(
        df['LLM_Closure_Acc'], 
        bins=BINS, 
        labels=LABELS, 
        right=False, 
        include_lowest=True
    )
    # Fix per il valore 1.0 esatto
    df.loc[df['LLM_Closure_Acc'] >= 0.8, 'Accuracy_Band'] = LABELS[-1]

    # Trova i valori unici di Clips (es. 5, 8)
    unique_clips = sorted(df['Clips'].unique())
    
    print(f"🎯 Found configurazioni Clip: {unique_clips}")

    for current_clip in unique_clips:
        print(f"Generating graph for {current_clip} Clips...")
        
        # 1. Filtra solo le righe con QUESTO numero di clip
        subset = df[df['Clips'] == current_clip].copy()
        
        # 2. Crea etichetta asse X basata SOLO sui Frames
        subset['X_Label'] = subset['Frames'].astype(str) + " Frames"
        
        # 3. Aggregazione: Conta quanti video per ogni Frame + Fascia
        grouped = subset.groupby(['X_Label', 'Accuracy_Band'], observed=False).size().reset_index(name='Counts')
        
        # 4. Pivot per il grafico stacked
        pivot_df = grouped.pivot(index='X_Label', columns='Accuracy_Band', values='Counts').fillna(0)
        
        # 5. Normalizza a percentuali (Somma riga = 100%)
        pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        
        # Se vuoto, salta
        if pivot_pct.empty: continue

        # 6. Plotting
        plt.figure(figsize=(10, 7))
        ax = pivot_pct.plot(
            kind='bar', 
            stacked=True, 
            color=COLORS, 
            edgecolor='black', 
            width=0.6,
            figsize=(9, 7),
            rot=0 # Etichette orizzontali
        )

        plt.title(f"Impatto dei Frame a Complessità Fissa ({current_clip} Clips)", fontsize=15, pad=15)
        plt.ylabel("Percentuale Video (%)", fontsize=12)
        plt.xlabel("Granularità Visiva", fontsize=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Accuratezza")
        plt.ylim(0, 100)
        
        # Aggiungi etichette %
        for c in ax.containers:
            labels = [f'{v.get_height():.1f}%' if v.get_height() > 3 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='black', padding=3)

        plt.tight_layout()
        
        # Salvataggio
        filename = f"impact_frames_fixed_C{current_clip}.png"
        out_path = os.path.join(OUTPUT_PLOT_DIR, filename)
        plt.savefig(out_path, dpi=300)
        print(f"Salvato: {out_path}")
        plt.close()

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        plot_split_by_clips(df)
        print("\n🎉 Finito! Controlla la cartella 'plots_distribution_split'")
    else:
        print(f"Nessun dato trovato.")