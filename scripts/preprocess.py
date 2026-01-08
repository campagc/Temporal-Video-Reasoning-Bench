import pandas as pd
import os
import numpy as np
import json
import shutil
import sys
import argparse
from PIL import Image
from decord import VideoReader, cpu

# --- CONFIGURAZIONE ---
DEFAULT_VIDEO_ID = "9_12" # Precisabile su Terminale
RAW_VIDEOS_DIR = "data/raw_videos"
CSV_PATH = "data/annotations_repo/annotation_csv/step_annotations.csv" 
OUTPUT_DIR = "data/processed_frames"
TASK_GRAPHS_DIR = "data/annotations_repo/task_graphs"
DEFAULT_NUM_FRAMES = 8   # Frame per clip (Default) | modificabile con --frames
TRIM_MARGIN = 0.10   # 10% taglio testa/coda
MAX_SIDE = 1024

# Purtroppo il task graph non ha un buon sistema di ID collegati ai video, 
# perciò bisogna collegarli con la descrizione
def find_taskgraph_id(label, task_graph):
    if 'steps' not in task_graph:
        return None
    
    label_lower = label.lower()
    
    # Prima prova: match esatto (case-insensitive)
    for step_id, description in task_graph['steps'].items():
        if description.lower() == label_lower:
            return int(step_id)
    
    # Seconda prova: match parziale (per sicurezza)
    for step_id, description in task_graph['steps'].items():
        if description.lower() in label_lower:
            return int(step_id)
    
    # Terza prova: match parziale (per sicurezza)
    for step_id, description in task_graph['steps'].items():
        if label_lower in description.lower():
            return int(step_id)
    
    return None 

def main():
    # --- GESTIONE ARGOMENTI ---
    parser = argparse.ArgumentParser(description="Script di preprocessing video per estrazione frame.")
    
    parser.add_argument(
        "video_input", 
        nargs="?", 
        default=None, 
        help="ID del video (es. '9_12') oppure percorso completo al file video."
    )
    
    parser.add_argument(
        "--frames", 
        type=int, 
        default=DEFAULT_NUM_FRAMES, 
        help=f"Numero di frame da estrarre per ogni clip (Default: {DEFAULT_NUM_FRAMES})"
    )

    args = parser.parse_args()
    current_num_frames = args.frames

    # Gestione Video Input
    # 1. Se non do argomenti -> uso DEFAULT_VIDEO_ID
    # 2. Se do un percorso -> file video nel percorso
    # 3. Se non è un percorso -> lo interpreto come ID video da cercare in RAW_VIDEOS_DIR

    target_video_id = DEFAULT_VIDEO_ID
    target_video_path = os.path.join(RAW_VIDEOS_DIR, f"{DEFAULT_VIDEO_ID}.mp4")

    if args.video_input:
        user_input = args.video_input
        if os.path.exists(user_input) and os.path.isfile(user_input):
            print(f"Input rilevato come percorso file: {user_input}")
            target_video_path = user_input
            filename = os.path.basename(user_input)
            target_video_id = os.path.splitext(filename)[0]
        else:
            print(f"Input rilevato come ID video: {user_input}")
            target_video_id = user_input
            target_video_path = os.path.join(RAW_VIDEOS_DIR, f"{user_input}.mp4")
    else:
        print(f"Nessun video specificato. Uso default ID: {target_video_id}")

    # Controllo esistenza file video
    if not os.path.exists(target_video_path):
        print(f"ERRORE: Video non trovato al percorso: {target_video_path}")
        return

    print(f"-> Processing Video ID: {target_video_id}")
    print(f"-> Path: {target_video_path}")
    print(f"-> Frames per clip: {current_num_frames}")

    # -------------------------------

    # --- PULIZIA CARTELLA OUTPUT ---
    if os.path.exists(OUTPUT_DIR):
        print(f"Pulizia cartella output esistente: {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # -------------------------------

    # Load activity mapping CSVs
    print("Lettura CSV mapping attività...")
    recording_activity_df = pd.read_csv("data/annotations_repo/annotation_csv/recording_id_step_idx.csv")
    recording_activity_df.columns = recording_activity_df.columns.str.strip()
    
    activity_name_df = pd.read_csv("data/annotations_repo/annotation_csv/activity_idx_step_idx.csv")
    activity_name_df.columns = activity_name_df.columns.str.strip()
    
    # Create mapping: activity_id -> activity_name (lowercase, no spaces)
    activity_name_map = {}
    for _, row in activity_name_df.iterrows():
        activity_id = row['activity_idx']
        activity_name = row['activity_name'].lower().replace(" ", "")
        activity_name_map[activity_id] = activity_name
    
    # Get activity_id for current VIDEO_ID
    # Usa target_video_id invece della globale
    video_activity = recording_activity_df[recording_activity_df['recording_id'] == target_video_id]
    if video_activity.empty:
        print(f"ERRORE: Nessuna attività trovata nel CSV per VIDEO_ID {target_video_id}")
        return
    activity_id = video_activity.iloc[0]['activity_id']
    activity_name = activity_name_map.get(activity_id, "unknown")

    print("Lettura CSV annotazioni...")
    df = pd.read_csv(CSV_PATH, sep=None, engine='python')
    df.columns = df.columns.str.strip() 
    
    video_steps = df[df['recording_id'] == target_video_id].copy()
    
    if video_steps.empty:
        print(f"ERRORE: Nessuno step trovato nel CSV annotazioni per ID {target_video_id}")
        return

    if 'step_idx' in video_steps.columns:
        video_steps = video_steps.sort_values('step_idx')
    
    print(f"Trovati {len(video_steps)} step per il video {target_video_id}")
    print(f"Attività: {activity_name}")
    
    # Carica il task graph per mappare gli step ID
    task_graph_path = os.path.join(TASK_GRAPHS_DIR, f"{activity_name}.json")
    task_graph = None
    if os.path.exists(task_graph_path):
        with open(task_graph_path, "r") as f:
            task_graph = json.load(f)
        print(f"Task graph caricato da: {task_graph_path}")
    else:
        print(f"Task graph non trovato in {task_graph_path}")

    # Usa target_video_path
    vr = VideoReader(target_video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    ground_truth_list = []

    for index, row in video_steps.iterrows():
        label = row.get('description', row.get('step_description', f"step_{index}"))
        step_id = row.get('step_idx', index)
        
        try:
            start_sec = float(row['start_time'])
            end_sec = float(row['end_time'])
        except:
            continue

        seg_duration = end_sec - start_sec
        margin = seg_duration * TRIM_MARGIN
        trim_start = start_sec + margin
        trim_end = end_sec - margin
        
        f_start = int(trim_start * fps)
        f_end = int(trim_end * fps)

        if f_end <= f_start: continue
        f_end = min(f_end, total_frames - 1)
        f_start = max(0, f_start)

        # USA current_num_frames invece della costante globale
        frame_indices = np.linspace(f_start, f_end-1, current_num_frames, dtype=int)
        step_folder = os.path.join(OUTPUT_DIR, f"step_{step_id}")
        os.makedirs(step_folder, exist_ok=True)

        try:
            video_batch = vr.get_batch(frame_indices).asnumpy()
        except Exception as e:
            print(f"Errore step {step_id}: {e}")
            continue

        saved_frames = []
        
        for i, img_array in enumerate(video_batch):
            im = Image.fromarray(img_array)
            
            # Resize se necessario
            width, height = im.size
            if max(width, height) > MAX_SIDE:
                ratio = MAX_SIDE / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                im = im.resize(new_size, Image.Resampling.LANCZOS)
            
            fname = f"frame_{i}.jpg"
            save_path = os.path.join(step_folder, fname)
            im.save(save_path, quality=95)
            saved_frames.append(save_path)

        print(f"Step {step_id}: Salvati {len(saved_frames)} frame HD -> {label}")

        # Cerca l'ID nel task graph
        taskgraph_id = find_taskgraph_id(label, task_graph) if task_graph else None
        
        ground_truth_list.append({
            "step_id": int(step_id),
            "label": str(label),
            "activity_name": activity_name,
            "taskgraph_id": taskgraph_id,
            "frames": saved_frames
        })

    output_json = os.path.join(OUTPUT_DIR, "dataset_manifest.json")
    with open(output_json, "w") as f:
        json.dump(ground_truth_list, f, indent=4)

    print(f"\n✅ FASE 1 COMPLETATA (Alta Risoluzione).")

if __name__ == "__main__":
    main()