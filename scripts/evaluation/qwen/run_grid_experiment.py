import os
import sys
import torch
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from run_entire_dataset import run_inference_with_loaded_model
    from evaluate_entire_dataset import evaluate_file
except ImportError:
    print(f"Error importing local modules.")
    sys.exit(1)

# Default paths
BASE_OUTPUT_DIR = "output_grid_experiment"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

def main():
    # 1. Parsing Argomenti (per SLURM Array)
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, required=True, help="Number of frames")
    parser.add_argument("--clips", type=int, required=True, help="Number of clips")
    parser.add_argument("--dataset", type=str, default="com", choices=["com", "captaincook"])
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated runs")
    args = parser.parse_args()

    dataset_name = args.dataset
    n_frames = args.frames
    n_clips = args.clips
    num_repeats = args.repeats

    print("==================================================")
    print(f"OPTIMIZED GRID WORKER ({dataset_name.upper()})")
    print(f"Frames: {n_frames} | Clips: {n_clips}")
    print(f"Repeats: {num_repeats}")
    print("==================================================\n")

    # 2. Loading Model con Flash Attention 2
    print(f"Loading Model {MODEL_PATH}...")
    
    # Rileva automaticamente se Flash Attention è supportato (Ampere/Lovelace/Hopper)
    # Le GPU A30 e L40S lo supportano!
    if torch.cuda.is_available():
        try:
            attn_implementation = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
        except Exception:
            attn_implementation = "sdpa"
    else:
        attn_implementation = "sdpa"
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print(f"Model loaded with: {attn_implementation}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Configurazione Percorsi
    output_subdir = f"{BASE_OUTPUT_DIR}_{dataset_name}/frames_{n_frames}/clips_{n_clips}"
    os.makedirs(output_subdir, exist_ok=True)
    
    if dataset_name == "com":
        manifest_path = f"data/processed_frames_com_{n_frames}/dataset_manifest.json"
        graph_dir = "data/annotations_repo/task_graphs_com"
    else:
        manifest_path = f"data/processed_frames_{n_frames}/dataset_manifest.json"
        graph_dir = "data/annotations_repo/task_graphs"

    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return

    # 4. Loop Ripetizioni
    for run_idx in range(num_repeats):
        json_out = os.path.join(output_subdir, f"run_{run_idx}_results.json")
        csv_out = os.path.join(output_subdir, f"run_{run_idx}_metrics.csv")
        
        print(f"Run {run_idx+1}/{num_repeats} ...")
        
        if os.path.exists(csv_out):
            print(f"Already completed.")
            continue
            
        # A. Inferenza
        run_inference_with_loaded_model(
            model=model,
            processor=processor,
            manifest_path=manifest_path,
            output_path=json_out,
            num_clips=n_clips,
            seed_offset=run_idx,
            temperature=0
        )
        
        # B. Valutazione
        if os.path.exists(json_out):
            metrics = evaluate_file(json_out, csv_out, task_graphs_path=graph_dir)
            if metrics:
                print(f"Acc: {metrics.get('LLM_Closure_Acc', 0.0):.4f}")
        
    print(f"\n WORKER COMPLETED.")

if __name__ == "__main__":
    main()