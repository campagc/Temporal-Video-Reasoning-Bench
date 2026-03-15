import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import itertools

# ==============================================================================
#  CONFIGURAZIONE
# ==============================================================================
BASE_INPUT_DIR =  "output_grid_internvl_captaincook"
OUTPUT_PLOT_DIR = os.path.join(BASE_INPUT_DIR, "plots_verification")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# Quante simulazioni random fare per costruire la "curva della fortuna"
NUM_SIMULATIONS = 5000 

# ==============================================================================
# 1. GENERATORE DI FORTUNA (Monte Carlo Simulation)
# ==============================================================================
def calculate_kendall_tau_simple(seq):
    """Calcola Tau per una lista sequenziale semplice [0, 1, 2...]"""
    n = len(seq)
    if n <= 1: return 1.0
    
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Nel ground truth, i < j sempre (lista ordinata)
            # Nella predizione seq, vediamo se seq[i] < seq[j]
            if seq.index(i) < seq.index(j):
                concordant += 1
            else:
                discordant += 1
    
    total_pairs = concordant + discordant
    if total_pairs == 0: return 1.0
    tau = (concordant - discordant) / total_pairs
    return (tau + 1) / 2 # Normalizza 0-1 (equivalente all'accuracy pairwise su grafo lineare)

def simulate_random_distribution(num_clips, num_sims):
    """Genera N punteggi casuali per vedere quanto spesso si è fortunati"""
    scores = []
    ground_truth = list(range(num_clips))
    
    for _ in range(num_sims):
        # Mischia a caso
        prediction = ground_truth[:]
        random.shuffle(prediction)
        
        # Calcola score (assumiamo grafo lineare per semplicità nella simulazione teorica)
        # Nota: La pairwise accuracy su grafo lineare è matematicamente correlata a questo
        score = calculate_kendall_tau_simple(prediction)
        scores.append(score)
    return scores

# ==============================================================================
# 2. CARICAMENTO DATI REALI
# ==============================================================================
def load_real_data():
    all_files = glob.glob(os.path.join(BASE_INPUT_DIR, "**", "*_metrics.csv"), recursive=True)
    df_list = []
    
    for f in all_files:
        parts = f.split(os.sep)
        n_frames = next((int(p.split('_')[1]) for p in parts if 'frames_' in p), None)
        n_clips = next((int(p.split('_')[1]) for p in parts if 'clips_' in p), None)
        
        if n_frames is None or n_clips is None: continue
        
        try:
            temp_df = pd.read_csv(f)
            temp_df['Frames'] = n_frames
            temp_df['Clips'] = n_clips
            temp_df['Source'] = 'AI Model (Reale)'
            df_list.append(temp_df[['Clips', 'Frames', 'LLM_Closure_Acc', 'Source']])
        except: pass
            
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# ==============================================================================
# 3. PLOTTING
# ==============================================================================
def plot_comparison(real_df, clips_val, frames_val):
    """Confronta la distribuzione del modello con quella del caso puro"""
    
    # 1. Filtra i dati del modello per questa configurazione
    model_scores = real_df[
        (real_df['Clips'] == clips_val) & 
        (real_df['Frames'] == frames_val)
    ]['LLM_Closure_Acc'].tolist()
    
    if not model_scores: return

    # 2. Genera dati random (La "Curva della Fortuna")
    random_scores = simulate_random_distribution(clips_val, NUM_SIMULATIONS)
    
    # Crea DataFrame combinato per Seaborn
    plot_data = pd.DataFrame({
        'Accuracy': model_scores + random_scores,
        'Type': ['Modello AI'] * len(model_scores) + ['Caso Puro (Simulato)'] * len(random_scores)
    })

    # 3. Calcolo Statistiche
    avg_model = np.mean(model_scores)
    avg_random = np.mean(random_scores)
    
    # Calcola quanti del modello battono il 95° percentile del random (Significatività)
    threshold_95 = np.percentile(random_scores, 95)
    model_above_chance = sum(s > threshold_95 for s in model_scores)
    pct_above_chance = (model_above_chance / len(model_scores)) * 100

    # 4. Disegno
    plt.figure(figsize=(10, 6))
    
    # Istogramma sovrapposto (KDE + Hist)
    sns.histplot(
        data=plot_data, 
        x="Accuracy", 
        hue="Type", 
        kde=True, 
        stat="density", 
        common_norm=False,
        bins=15,
        palette={'Modello AI': '#2ca02c', 'Caso Puro (Simulato)': '#7f7f7f'},
        alpha=0.4
    )
    
    # Linee delle medie
    plt.axvline(avg_model, color='green', linestyle='--', linewidth=2, label=f'Media AI: {avg_model:.2f}')
    plt.axvline(avg_random, color='gray', linestyle='--', linewidth=2, label=f'Media Random: {avg_random:.2f}')
    plt.axvline(threshold_95, color='red', linestyle=':', linewidth=1.5, label='Soglia Fortuna (95%)')

    plt.title(f"Verifica: Bravura vs Fortuna ({frames_val} Frames, {clips_val} Clips)", fontsize=14)
    plt.xlabel("Closure Accuracy", fontsize=12)
    plt.ylabel("Densità", fontsize=12)
    plt.xlim(0, 1.05)
    plt.legend()
    
    # Annotazione
    info_text = (
        f"Video analizzati: {len(model_scores)}\n"
        f"Il {pct_above_chance:.1f}% dei video\n"
        f"supera la soglia di fortuna del 95%."
    )
    plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes, 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    filename = f"verification_F{frames_val}_C{clips_val}.png"
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, filename), dpi=300)
    print(f"Generato: {filename}")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"Generazione Grafici di Verifica (Model vs Pure Luck)...")
    
    df = load_real_data()
    
    if not df.empty:
        # Per ogni combinazione disponibile
        configs = df[['Frames', 'Clips']].drop_duplicates().values
        for frames, clips in configs:
            plot_comparison(df, int(clips), int(frames))
    else:
        print(f"Nessun dato trovato.")