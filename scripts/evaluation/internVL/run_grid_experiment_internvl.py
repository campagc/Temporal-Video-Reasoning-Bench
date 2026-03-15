import argparse
import os
import sys

# Aggiungi la cartella corrente al path per importare i moduli locali
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from run_internvl_dataset import run_inference_internvl
    from evaluate_entire_dataset import evaluate_file
except ImportError as e:
    print("X Error importing modules.")
    print(f"Ensure that 'run_internvl_dataset.py' e 'evaluate_entire_dataset.py' are in the folder: {os.getcwd()}")
    print(f"Error detail: {e}")
    sys.exit(1)

# --- CONFIGURAZIONE ---
MODEL_PATH = "data/models/InternVL3_5-8B" 

def main():
    # 1. Parsing Argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, required=True, help="Number of frames per clip")
    parser.add_argument("--clips", type=int, required=True, help="Number of clips to order")
    parser.add_argument("--dataset", type=str, default="com", choices=["com", "captaincook"])
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions with different seeds")
    args = parser.parse_args()

    n_frames = args.frames
    n_clips = args.clips
    dataset_name = args.dataset
    num_repeats = args.repeats

    print("==================================================")
    print(f"INTERNVL 3.5 GRID WORKER ({dataset_name.upper()})")
    print(f"Frames: {n_frames} | Clips: {n_clips}")
    print(f"Repeats: {num_repeats}")
    print(f"Model Path: {MODEL_PATH}")
    print("==================================================\n")

    # 2. Setup Percorsi Input/Output
    base_output_dir = f"output_grid_internvl_{dataset_name}"
    output_subdir = os.path.join(base_output_dir, f"frames_{n_frames}", f"clips_{n_clips}")
    os.makedirs(output_subdir, exist_ok=True)

    # Percorsi Dati
    if dataset_name == "com":
        manifest_path = f"data/processed_frames_com_{n_frames}/dataset_manifest.json"
        graph_dir = "data/annotations_repo/task_graphs_com"
    else:
        manifest_path = f"data/processed_frames_{n_frames}/dataset_manifest.json"
        graph_dir = "data/annotations_repo/task_graphs"

    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return

    # 3. Loop Ripetizioni
    for run_idx in range(num_repeats):
        json_out = os.path.join(output_subdir, f"run_{run_idx}_results.json")
        csv_out = os.path.join(output_subdir, f"run_{run_idx}_metrics.csv")
        
        print(f"Run {run_idx+1}/{num_repeats} ...")
        
        # A. INFERENZA
        if not os.path.exists(json_out):
            run_inference_internvl(
                model_path=MODEL_PATH,
                manifest_path=manifest_path,
                output_path=json_out,
                num_clips=n_clips,
                seed_offset=run_idx
            )
        else:
            print(f"Inference already completed (existing json file).")
            
        # B. VALUTAZIONE
        if os.path.exists(json_out):
            if not os.path.exists(csv_out):
                print(f"Calculating metrics...")
                metrics = evaluate_file(json_out, csv_out, task_graphs_path=graph_dir)
                if metrics:
                    acc = metrics.get('LLM_Closure_Acc', 0.0)
                    print(f"Closure Acc: {acc:.4f}")
            else:
                print(f"Metrics already calculated.")
        
    print(f"\nWORKER COMPLETED.")

if __name__ == "__main__":
    main()