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
OUTPUT_PLOT_DIR = os.path.join(BASE_INPUT_DIR, "plots_distribution")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# Definiamo le fasce di accuratezza
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
LABELS = ['0-20% (Pessimo)', '20-40% (Basso)', '40-60% (Random/Medio)', '60-80% (Buono)', '80-100% (Eccellente)']
# Colori semaforici: Rosso -> Giallo -> Verde
COLORS = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"] 

# ==============================================================================
# 1. CARICAMENTO E PREPARAZIONE DATI
# ==============================================================================
def extract_params(filepath):
    """Estrae configurazione dal percorso file"""
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
            temp_df['Config_Label'] = f"{frames} Frame / {clips} Clips"
            df_list.append(temp_df)
        except Exception as e:
            print(f"Errore lettura {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

# ==============================================================================
# 2. GENERAZIONE GRAFICI
# ==============================================================================

def plot_stacked_distribution(df):
    """
    Crea un grafico a barre impilate normalizzato al 100%.
    Mostra la percentuale di video in ogni fascia di accuratezza per ogni configurazione.
    """
    # Creiamo la colonna delle fasce (Binning)
    df['Accuracy_Band'] = pd.cut(
        df['LLM_Closure_Acc'], 
        bins=BINS, 
        labels=LABELS, 
        right=False, 
        include_lowest=True
    )
    
    # Gestione del caso 1.0 che potrebbe essere escluso se right=False
    # Forziamo i valori > 1.0 (impossibili, ma per float precision) o 1.0 esatti nell'ultimo bin
    df.loc[df['LLM_Closure_Acc'] >= 0.8, 'Accuracy_Band'] = LABELS[-1]

    # Aggregazione: Conta quanti video per Configurazione + Fascia
    grouped = df.groupby(['Config_Label', 'Accuracy_Band'], observed=False).size().reset_index(name='Counts')
    
    # Pivot per avere le fasce come colonne
    pivot_df = grouped.pivot(index='Config_Label', columns='Accuracy_Band', values='Counts').fillna(0)
    
    # Converti in percentuali (Normalizzazione per riga)
    pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    # PLOT
    plt.figure(figsize=(12, 7))
    ax = pivot_pct.plot(
        kind='bar', 
        stacked=True, 
        color=COLORS, 
        edgecolor='black', 
        width=0.7,
        figsize=(12, 7)
    )

    plt.title("Distribuzione dell'Accuratezza del Modello per Configurazione", fontsize=16, pad=20)
    plt.ylabel("Percentuale dei Video Totali (%)", fontsize=12)
    plt.xlabel("Configurazione (Frame estratti / Clip da ordinare)", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Fascia di Accuratezza", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Aggiungi etichette di testo sulle barre
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='black', fontsize=9, padding=3)

    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_PLOT_DIR, "accuracy_distribution_stacked.png")
    plt.savefig(out_path, dpi=300)
    print(f"Grafico salvato: {out_path}")

def plot_absolute_counts(df):
    """
    Grafico a barre raggruppate per vedere i numeri assoluti (non percentuali).
    Utile per capire se abbiamo perso dati o se i campioni sono bilanciati.
    """
    df['Accuracy_Band'] = pd.cut(df['LLM_Closure_Acc'], bins=BINS, labels=LABELS, include_lowest=True)
    
    plt.figure(figsize=(14, 8))
    sns.countplot(
        data=df, 
        x='Accuracy_Band', 
        hue='Config_Label', 
        palette="viridis",
        edgecolor="black"
    )
    
    plt.title("Numero Assoluto di Video per Fascia di Accuratezza", fontsize=16)
    plt.xlabel("Fascia di Accuratezza", fontsize=12)
    plt.ylabel("Numero di Video (Somma di tutte le Run)", fontsize=12)
    plt.legend(title="Configurazione")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path = os.path.join(OUTPUT_PLOT_DIR, "accuracy_counts_grouped.png")
    plt.savefig(out_path, dpi=300)
    print(f"Grafico conteggi salvato: {out_path}")

def print_text_report(df):
    """Stampa un report testuale nel terminale"""
    print("\n" + "="*60)
    print("📋 REPORT DISTRIBUZIONE ACCURATEZZA")
    print("="*60)
    
    # Calcoliamo la banda
    df['Accuracy_Band'] = pd.cut(df['LLM_Closure_Acc'], bins=BINS, labels=LABELS, include_lowest=True)
    
    configs = df['Config_Label'].unique()
    for conf in sorted(configs):
        subset = df[df['Config_Label'] == conf]
        total = len(subset)
        
        # Conta quanti sono "Eccellenti" (>0.8)
        excellent = len(subset[subset['LLM_Closure_Acc'] >= 0.8])
        # Conta quanti sono "Buoni o meglio" (>0.6)
        good_plus = len(subset[subset['LLM_Closure_Acc'] >= 0.6])
        # Conta quanti sono "Pessimi" (<0.2)
        bad = len(subset[subset['LLM_Closure_Acc'] < 0.2])
        
        print(f"\n🔹 {conf} (Totale Video analizzati nelle 3 run: {total})")
        print(f"Eccellenti (80-100%): {excellent} ({excellent/total:.1%})")
        print(f"Buoni+    (>60%):    {good_plus} ({good_plus/total:.1%})")
        print(f"Pessimi   (<20%):    {bad} ({bad/total:.1%})")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    df = load_data()
    
    if not df.empty:
        # 1. Grafico a barre impilate (Percentuali) - IL PIÙ UTILE
        plot_stacked_distribution(df)
        
        # 2. Grafico conteggi assoluti
        plot_absolute_counts(df)
        
        # 3. Report testuale
        print_text_report(df)
        
        print("\n🎉 Finito! Controlla la cartella 'output_grid_experiment/plots_distribution'.")
    else:
        print(f"Nessun dato trovato.")