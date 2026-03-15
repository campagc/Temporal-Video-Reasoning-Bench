#!/bin/bash
#SBATCH -J seq_ivl_debug
#SBATCH -o logs/seq_ivl_%j.out
#SBATCH -e logs/seq_ivl_%j.err
#SBATCH -p edu-thesis       
#SBATCH --gres=gpu:1    
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00   

# --- CONFIGURAZIONE ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_FILE="config_experiments_internvl.txt" 

# --- A. SETUP AMBIENTE ---
echo "=================================================="
echo "JOB START TIME: $(date)"
echo "JOB ID: $SLURM_JOB_ID"
echo "HOSTNAME: $(hostname)"
echo "=================================================="

cd "$PROJECT_DIR"
mkdir -p logs

# Load modules and conda
module load CUDA/12.3.0 || echo "  Warning: CUDA module not loaded"
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

# --- B. ADVANCED DIAGNOSTICS ---
echo -e "\n --- SYSTEM DIAGNOSTICS ---"
echo "Python Path: $(which python)"
echo "Python Version: $(python --version)"

echo -e "\n--- LIBRERIE CRITICHE ---"
pip list | grep -E "torch|transformers|accelerate|flash-attn|deepspeed|bitsandbytes"

echo -e "\n--- STATO GPU INIZIALE ---"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo -e "\n--- VERIFICA PYTORCH ---"
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'Cuda Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Current Device: {torch.cuda.current_device()}'); print(f'Device Name: {torch.cuda.get_device_name(0)}')"

echo -e "\n--- VERIFICA PATH MODELLO ---"
MODEL_PATH="data/models/InternVL3_5-8B"
if [ -d "$MODEL_PATH" ]; then
    echo "Cartella modello trovata: $MODEL_PATH"
    ls -lh "$MODEL_PATH" | head -n 5
else
    echo "ATTENZIONE: Cartella modello NON trovata in $MODEL_PATH"
fi

# export CUDA_LAUNCH_BLOCKING=1 #per debug ma rallenta
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

echo "=================================================="
echo " AVVIO LOOP ESPERIMENTI"
echo "=================================================="

# --- C. LOOP DI ESECUZIONE ---
count=0
while read -r FRAMES CLIPS DATASET; do
    # Salta righe vuote o commenti
    if [[ "$FRAMES" =~ ^#.* ]] || [[ -z "$FRAMES" ]]; then continue; fi
    
    ((count++))
    echo -e "\n🔸 [Task #$count] AVVIO: F=$FRAMES C=$CLIPS D=$DATASET"
    echo "   ⏰ Timestamp: $(date)"

    # Esegue Python
    python -u scripts/evaluation/internVL/run_grid_experiment_internvl.py \
        --frames "$FRAMES" \
        --clips "$CLIPS" \
        --dataset "$DATASET" \
        --repeats 3
    
    RET_CODE=$?
    
    if [ $RET_CODE -eq 0 ]; then
        echo "Task #$count Completato."
    else
        echo "Task #$count FALLITO (Exit Code: $RET_CODE)."
        echo "Controlla il file .err per lo stack trace completo."
    fi
    
    # Pulizia forzata buffer
    echo "Pulizia post-run..."
    sleep 5
    
done < <(grep -v '^#' "$CONFIG_FILE" | grep -v '^$')

echo -e "\nJOB TERMINATO: $(date)"