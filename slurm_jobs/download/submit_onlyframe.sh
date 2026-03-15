#!/bin/bash
#SBATCH -J frame_extract
#SBATCH -o logs/extractions/extract_%j.out
#SBATCH -e logs/extractions/extract_%j.err
#SBATCH -p edu-medium
#SBATCH --gres=gpu:0     
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# esempio di sottomissione:
# sbatch --export=NUM_FRAMES=12 scripts/download/submit_onlyframe.sh

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
mkdir -p logs/extractions

# --- CONFIGURAZIONE VARIABILI ---
# Se non passo la variabile, usa 8 come default
TARGET_FRAMES=${NUM_FRAMES:-8}

echo "--------------------------------------------------"
echo " Avvio Estrazione Frame"
echo "🔹 Target Frames: $TARGET_FRAMES"
echo "--------------------------------------------------"

# Verifica esistenza metadata (il download deve essere già fatto)
METADATA="data/clean_videos_metadata.json"
if [[ ! -f "$METADATA" ]]; then
  echo " Errore: $METADATA non trovato. Esegui prima il download." >&2
  exit 2
fi

# --- ESECUZIONE SCRIPT ---
# Nota: Ensure that il percorso dello script python sia corretto
python -u scripts/download/frame_extractor.py --frames "$TARGET_FRAMES"

echo " Estrazione completata per $TARGET_FRAMES frames."