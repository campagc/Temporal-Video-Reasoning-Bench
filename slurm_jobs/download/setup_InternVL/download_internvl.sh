#!/bin/bash
#SBATCH -J download_internvl_v4
#SBATCH -o logs/d_internvl_v4_%j.out
#SBATCH -e logs/d_internvl_v4_%j.err
#SBATCH -p edu-long            
#SBATCH --gres=gpu:0        
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G          
#SBATCH --time=10:00:00        

# --- 1. SETUP ---
echo "=================================================="
echo "📅 Data inizio: $(date)"
echo "🆔 Job ID:      $SLURM_JOB_ID"
echo "🖥️  Nodo:       $SLURMD_NODENAME"
echo " Directory:   $(pwd)"
echo "=================================================="

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

# --- 2. FIX AMBIENTE (Forziamo l'aggiornamento) ---
echo " Trying to force installation of a recent version..."
# Forziamo una reinstallazione ignorando i pacchetti installati per pulire eventuali conflitti
pip install --force-reinstall --no-deps "huggingface_hub>=0.23.0"

# --- 3. SCRIPT PYTHON INTEGRATO ---
# Scriviamo un piccolo file python temporaneo per gestire il download
cat <<EOF > download_script.py
import os
from huggingface_hub import snapshot_download, login

# Definisci cartelle
model_id = "OpenGVLab/InternVL3_5-8B"
local_dir = "data/models/InternVL3_5-8B"

print(f" "Inizio download di {model_id}...")
print(f" "Destinazione: {local_dir}")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Importante per avere i file reali
        resume_download=True,          # Riprende se si blocca
        max_workers=4                  # Usa più CPU per scaricare
    )
    print(" "Download completato con successo (Python API)!")
except Exception as e:
    print(f" "Errore Python: {e}")
    exit(1)
EOF

# --- 4. ESECUZIONE ---
echo -e "\n️  [ACTION] Eseguo lo script Python di download..."
python -u download_script.py
EXIT_CODE=$?

# Pulizia file temporaneo
rm download_script.py

# --- 5. CONTROLLO FINALE ---
echo -e "\n=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " JOB COMPLETED!"
else
    echo " JOB FALLITO (Exit code: $EXIT_CODE)"
fi

echo -e "\n [DEBUG] Dimensione finale:"
du -sh "data/models/InternVL3_5-8B"
echo "📅 Data fine:   $(date)"
echo "=================================================="