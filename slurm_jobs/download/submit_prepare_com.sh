#!/bin/bash
#SBATCH -J com_prep
#SBATCH -o logs/extractions_com/prep_com_%j.out
#SBATCH -e logs/extractions_com/prep_com_%j.err
#SBATCH -p edu-medium
#SBATCH --gres=gpu:0     
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ESEMPIO DI SOTTOMISSIONE:
# sbatch --export=NUM_FRAMES=6 scripts/download/submit_prepare_com.sh
# sbatch --export=NUM_FRAMES=4 scripts/download/submit_prepare_com.sh

# --- GESTIONE ERRORI ---
set -e

# --- SETUP AMBIENTE ---
echo "Job iniziato on node $(hostname)"
# Load conda in a robust way
if ! command -v conda &> /dev/null; then
    echo "Error: conda could not be found. Please ensure Conda is installed and in your PATH."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh" || true

# Ensure the user creates the required conda environment before running
if ! conda activate video_env; then
    echo "Error: conda environment 'video_env' not found. Please create it first."
    exit 1
fi

# Calculate project root relative to this script
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs/extractions_com

# --- CONFIGURAZIONE VARIABILI ---
# Se non passo la variabile, usa 8 come default
TARGET_FRAMES=${NUM_FRAMES:-8}

echo "--------------------------------------------------"
echo " Avvio Preparazione COM Kitchens (Frames + Grafi)"
echo "🔹 Target Frames: $TARGET_FRAMES"
echo "--------------------------------------------------"

# Verifica esistenza dati grezzi
DATA_ROOT="data/comkitchens"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo " Errore: Cartella $DATA_ROOT non trovata. Controlla di aver estratto lo zip." >&2
  exit 2
fi

# --- ESECUZIONE SCRIPT ---
# Lancia lo script python che abbiamo creato prima
python -u scripts/download/prepare_comkitchens.py --frames "$TARGET_FRAMES"

echo " Preparazione completata per $TARGET_FRAMES frames."