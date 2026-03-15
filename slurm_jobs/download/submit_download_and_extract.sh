#!/bin/bash
#SBATCH -J vid_down_extract
#SBATCH -o logs/down_extract_%j.out
#SBATCH -e logs/down_extract_%j.err
#SBATCH -p edu-medium
#SBATCH --gres=gpu:0 
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# --- GESTIONE ERRORI ---
set -e

# --- SETUP AMBIENTE ---
echo "Job started on $(date) on node $(hostname)"
source ~/.bashrc || true
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
mkdir -p logs

# --- CONFIGURAZIONE VARIABILI ---
# Se le variabili d'ambiente non sono settate, usa dei default
# MAX_DOWNLOADS (default: 20)
# NUM_FRAMES (default: 8)
TARGET_DOWNLOADS=${MAX_DOWNLOADS:-148}
TARGET_FRAMES=${NUM_FRAMES:-7}

echo "--------------------------------------------------"
echo " Avvio Pipeline"
echo "🔹 Max Downloads: $TARGET_DOWNLOADS"
echo "🔹 Frames per Clip: $TARGET_FRAMES"
echo "--------------------------------------------------"

# --- STEP 1: DOWNLOAD ---
echo -e "\n--- [1/2] RUNNING DOWNLOADER ---"
python -u scripts/download/download_manager.py --max_videos "$TARGET_DOWNLOADS"

# Verifica esistenza output
METADATA="data/clean_videos_metadata.json"
if [[ ! -f "$METADATA" ]]; then
  echo " Errore: $METADATA non trovato. Il download è fallito." >&2
  exit 2
fi

# --- STEP 2: EXTRACTION ---
echo -e "\n--- [2/2] RUNNING FRAME EXTRACTOR ---"
python -u scripts/download/frame_extractor.py --frames "$TARGET_FRAMES"

echo "--------------------------------------------------"
echo " Job Completato con successo il $(date)"
echo "--------------------------------------------------"