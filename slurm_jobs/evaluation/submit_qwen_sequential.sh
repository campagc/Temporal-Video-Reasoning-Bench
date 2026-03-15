#!/bin/bash
#SBATCH -J grid_seq_all
#SBATCH -o logs/seq_grid_%j.out
#SBATCH -e logs/seq_grid_%j.err
#SBATCH -p edu-thesis       
#SBATCH --gres=gpu:1        
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=42G
#SBATCH --time=23:59:59    

# --- CONFIGURAZIONE ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_FILE="config_experiments_qwen.txt"

# --- 1. SETUP INIZIALE ---
echo "--------------------------------------------------" >&2
echo "DEBUG: JOB START TIME: $(date)" >&2
echo "DEBUG: JOB_ID: $SLURM_JOB_ID" >&2
echo "--------------------------------------------------" >&2

# Stop solo se ci sono errori gravi di sistema, non se fallisce un singolo run python
set +e 

cd "$PROJECT_DIR"
mkdir -p logs

# Check file config
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERRORE: File $CONFIG_FILE non trovato in $(pwd)" >&2
    exit 1
fi

source ~/.bashrc
conda activate video_env

# --- 2. DEBUG INFO ---
echo "=================================================="
echo "📊 INFO SISTEMA"
echo "=================================================="
echo "🖥️  Nodo: $SLURMD_NODENAME"
echo "📂 Dir:  $(pwd)"
echo "📄 Config: $CONFIG_FILE"
echo "=================================================="

# --- 3. CICLO DI ESECUZIONE SEQUENZIALE ---
# Legge il file riga per riga. 
# 'grep -v' serve a ignorare le righe che iniziano con # (commenti) o sono vuote

count=0
while read -r FRAMES CLIPS DATASET; do
    
    # Incrementa contatore
    ((count++))
    
    echo -e "\n🔸 [Job Interno #$count] Inizio configurazione: F=$FRAMES C=$CLIPS D=$DATASET"
    echo "   ⏰ Ora: $(date)"

    # Esegue lo script Python
    # Il python userà la stessa GPU per tutti, uno alla volta
    python -u scripts/evaluation/qwen/run_grid_experiment.py \
        --frames "$FRAMES" \
        --clips "$CLIPS" \
        --dataset "$DATASET" \
        --repeats 3
    
    # Controllo esito (Exit code del python)
    if [ $? -eq 0 ]; then
        echo "   ✅ Completato con successo."
    else
        echo "   ❌ Errore durante l'esecuzione di questa configurazione."
        # Non facciamo exit, così prova a fare la prossima riga
    fi

    # Piccola pausa per far "respirare" la GPU e permettere il flush dei log
    sleep 2

done < <(grep -v '^#' "$CONFIG_FILE" | grep -v '^$')

# --- 4. CONCLUSIONE ---
echo -e "\n=================================================="
echo "🏁 TUTTI I TASK NEL FILE DI CONFIGURAZIONE SONO TERMINATI"
echo "📅 Data Fine: $(date)"
echo "=================================================="

echo "DEBUG: JOB ENDED AT: $(date)" >&2