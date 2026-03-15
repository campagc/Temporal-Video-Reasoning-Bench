#!/bin/bash
#SBATCH -J finish_setup
#SBATCH -o logs/finish_setup.out
#SBATCH -e logs/finish_setup.err
#SBATCH -p edu-short
#SBATCH --gres=gpu:1  
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00

# Carica il CUDA più recente disponibile
module load CUDA/12.3.0 

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

# 1. Aggiorna pip
pip install --upgrade pip

# 2. Installa/Aggiorna Transformers all'ultima versione (FONDAMENTALE)
pip install -U "transformers>=4.45"

# 3. Installa Flash Attention (richiede che il module CUDA sia caricato)
pip install flash-attn --no-build-isolation

# 4. Altre dipendenze richieste da InternVL
pip install timm einops decord opencv-python-headless termcolor