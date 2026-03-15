import pandas as pd
import os
import json
import shutil
import numpy as np
import argparse
from PIL import Image
from decord import VideoReader, cpu

import yaml
import logging

try:
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
except FileNotFoundError:
    config = {"project": {}}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEFAULT_NUM_FRAMES = config.get("project", {}).get("default_num_frames", 8)  # can be changed with --frames e.g. python frame_extractor.py --frames 10
TRIM_MARGIN = 0.10
MAX_SIDE = config.get("project", {}).get("max_side", 512)

# Paths
PROJECT_DIR = config.get("project", {}).get("dataset_root", "data")
ANNOTATIONS_DIR = os.path.join(PROJECT_DIR, "annotations_repo/annotation_csv")
METADATA_FILE = os.path.join(PROJECT_DIR, "clean_videos_metadata.json")
# OUTPUT_DIR and OUTPUT_MANIFEST are computed at runtime based on the number of frames
# Example: os.path.join(PROJECT_DIR, f"processed_frames_{num_frames}")

# ==============================================================================

def find_taskgraph_id(label, task_graph):
    if 'steps' not in task_graph: return None
    label_lower = label.lower()
    for sid, desc in task_graph['steps'].items():
        if desc.lower() == label_lower: return int(sid)
    for sid, desc in task_graph['steps'].items():
        if desc.lower() in label_lower: return int(sid)
    for sid, desc in task_graph['steps'].items():
        if label_lower in desc.lower(): return int(sid)
    return None

def preprocess_video(video_path, video_id, activity_name, task_graph_path, num_frames, output_dir):

    csv_steps = os.path.join(ANNOTATIONS_DIR, "step_annotations.csv")
    df = pd.read_csv(csv_steps, sep=None, engine='python')
    df.columns = df.columns.str.strip()
 
    if 'recording_id' in df.columns:
        df['recording_id'] = df['recording_id'].astype(str).str.strip()

    video_steps = df[df['recording_id'] == str(video_id)].copy()
    if video_steps.empty:
        return []
    if 'step_idx' in video_steps.columns:
        video_steps = video_steps.sort_values('step_idx')

    with open(task_graph_path, "r") as f:
        task_graph = json.load(f)

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except:
        logging.error(f"Error opening video {video_path}")
        return []

    total_frames = len(vr)
    fps = vr.get_avg_fps()
    processed_entries = []

    logging.info(f"Processing {video_id} ({len(video_steps)} steps)...")

    for index, row in video_steps.iterrows():
        step_id = row.get('step_idx', index)
        label = row.get('description', row.get('step_description', f"step_{index}"))

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

        if f_end <= f_start:
            continue
        f_end = min(f_end, total_frames - 1)
        f_start = max(0, f_start)

        # USE THE VALUE PASSED AS ARGUMENT
        frame_indices = np.linspace(f_start, f_end-1, num_frames, dtype=int)

        clip_folder_name = f"{video_id}_step_{step_id}"
        step_folder = os.path.join(output_dir, clip_folder_name)
        os.makedirs(step_folder, exist_ok=True)

        video_batch = vr.get_batch(frame_indices).asnumpy()
        saved_frames = []

        for i, img_array in enumerate(video_batch):
            im = Image.fromarray(img_array)
            width, height = im.size
            if max(width, height) > MAX_SIDE:
                ratio = MAX_SIDE / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                im = im.resize(new_size, Image.Resampling.LANCZOS)

            fname = f"frame_{i}.jpg"
            save_path = os.path.join(step_folder, fname)
            im.save(save_path, quality=85)
            saved_frames.append(save_path)

        tg_id = find_taskgraph_id(label, task_graph)

        processed_entries.append({
            "video_id": str(video_id),
            "step_id": int(step_id),
            "label": str(label),
            "activity_name": activity_name,
            "taskgraph_id": tg_id,
            "frames": saved_frames
        })

    return processed_entries

def main():
    # --- ARGUMENT MANAGEMENT ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=DEFAULT_NUM_FRAMES, 
                        help="Number of frames to extract per clip")
    args = parser.parse_args()
    
    current_frames = args.frames
    # --------------------------

    logging.info("==================================================")
    logging.info(f"FRAME EXTRACTOR (Frames per clip: {current_frames})")
    logging.info("==================================================")

    output_dir = os.path.join(PROJECT_DIR, f"processed_frames_{current_frames}")
    if os.path.exists(output_dir):
        logging.info(f"Cleaning previous output folder: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    output_manifest = os.path.join(output_dir, "dataset_manifest.json")

    if not os.path.exists(METADATA_FILE):
        logging.error("Error: Please run 01_download_manager.py first!")
        return

    with open(METADATA_FILE, "r") as f:
        video_list = json.load(f)
    
    logging.info(f"Found {len(video_list)} videos to process.")

    full_manifest = []
    
    for vid_meta in video_list:
        entries = preprocess_video(
            vid_meta['local_video_path'],
            vid_meta['video_id'],
            vid_meta['activity_name'],
            vid_meta['task_graph_path'],
            current_frames, # Pass the value
            output_dir
        )
        if entries:
            full_manifest.extend(entries)

    with open(output_manifest, "w") as f:
        json.dump(full_manifest, f, indent=4)

    logging.info("COMPLETED")
    logging.info(f"Manifest created: {output_manifest}")
    logging.info(f"Frames per clip used: {current_frames}")

if __name__ == "__main__":
    main()