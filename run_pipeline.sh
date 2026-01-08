#!/bin/bash

# ------------------------------------------------------------
# ||| ATTENZIONE: Eseguire SOLO su edu01 perché usa la GPU |||
# ------------------------------------------------------------

# STEP 0: Download Video
# STEP 1: Preprocess Video
# STEP 2: Esecuzione Modello (GPU)
# STEP 3: Calcolo Metriche

# UTILIZZO:
# e.g ./run_pipeline.sh 3
# e.g. NUM_FRAMES=16 NUM_CLIPS=10 NUM_RUNS=5 ./run_pipeline.sh 0 "10_1"

# ==============================================================================
# 🛠️ CONFIGURAZIONE GLOBALE UTENTE (Modifica qui i valori di default)
# ==============================================================================

# ID del video di default (usato se non passato come 2° argomento)
DEFAULT_VIDEO_ID="12_10"

# Parametri di Preprocessing
DEFAULT_NUM_FRAMES=8    # Quanti frame estrarre per clip

# Parametri di Inferenza (Modello)
DEFAULT_NUM_CLIPS=5     # Quante clip includere nel test
DEFAULT_NUM_RUNS=3      # Quante volte ripetere il test

# ==============================================================================

# --- CONFIGURAZIONE ERRORI ---
set -e # Ferma tutto se c'è un errore

# --- ATTIVAZIONE AMBIENTE ---
source ~/.bashrc
# source /opt/anaconda/etc/profile.d/conda.sh # Decommenta se necessario
conda activate video_env

# --- CARTELLA DI LAVORO ---
PROJECT_DIR="/home/giuliano.campagnolo/Progetto_Video"
cd "$PROJECT_DIR"

# ==============================================================================
# 1. GESTIONE ARGOMENTI E VARIABILI
# ==============================================================================

# Argomento 1: Step di partenza (Default: 0)
START_STEP=${1:-0}

# Argomento 2: Video ID
# Priorità: 1. Argomento script ($2) -> 2. Variabile Globale file
VIDEO_ID=${2:-$DEFAULT_VIDEO_ID}

# --- PARAMETRI OPZIONALI ---
# Priorità: 1. Variabile d'ambiente -> 2. Variabile Globale file

NUM_FRAMES=${NUM_FRAMES:-$DEFAULT_NUM_FRAMES}
NUM_CLIPS=${NUM_CLIPS:-$DEFAULT_NUM_CLIPS}
NUM_RUNS=${NUM_RUNS:-$DEFAULT_NUM_RUNS}

echo "=================================================="
echo "🚀 AVVIO PIPELINE VIDEO ANALYSIS"
echo "=================================================="
echo "🔹 Start Step:   $START_STEP"
echo "🔹 Video ID:     $VIDEO_ID"
echo "🔹 Configurazione:"
echo "   - Frames:     $NUM_FRAMES"
echo "   - Test Clips: $NUM_CLIPS"
echo "   - Runs:       $NUM_RUNS"
echo "=================================================="

# ==============================================================================
# 2. ESECUZIONE STEP
# ==============================================================================

# --- STEP 0: DOWNLOAD ---
if [ "$START_STEP" -le 0 ]; then
    echo -e "\n[0/3] ⬇️  Download Video ($VIDEO_ID)..."
    python scripts/download_single_video.py "$VIDEO_ID"
else
    echo -e "\n[0/3] Download Video... (SALTATO)"
fi

# --- STEP 1: PREPROCESS ---
if [ "$START_STEP" -le 1 ]; then
    echo -e "\n[1/3] 🎞️  Preprocessing ($VIDEO_ID | Frames: $NUM_FRAMES)..."
    python scripts/preprocess.py "$VIDEO_ID" --frames "$NUM_FRAMES"
else
    echo -e "\n[1/3] Preprocessing... (SALTATO)"
fi

# --- STEP 2: INFERENCE ---
if [ "$START_STEP" -le 2 ]; then
    echo -e "\n[2/3] 🧠 Esecuzione Modello (Clips: $NUM_CLIPS | Runs: $NUM_RUNS)..."
    python scripts/run_native_repeated.py --num_clips "$NUM_CLIPS" --num_runs "$NUM_RUNS"
else
    echo -e "\n[2/3] Esecuzione Modello... (SALTATO)"
fi

# --- STEP 3: EVALUATION ---
if [ "$START_STEP" -le 3 ]; then
    echo -e "\n[3/3] 📊 Calcolo Metriche..."
    python scripts/evaluate_metrics_dual.py
else
    echo -e "\n[3/3] Calcolo Metriche... (SALTATO)"
fi

echo -e "\n✅ PIPELINE COMPLETATA CON SUCCESSO!"
echo "=================================================="