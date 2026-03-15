import pandas as pd
import os
import ast
import json
import subprocess
import argparse

import yaml
import logging

# Load configuration
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
DEFAULT_MAX_VIDEOS = 148

# Paths
PROJECT_DIR = config.get("project", {}).get("dataset_root", "data")
ANNOTATIONS_DIR = os.path.join(PROJECT_DIR, "annotations_repo/annotation_csv")
TASK_GRAPHS_DIR = os.path.join(PROJECT_DIR, "annotations_repo/task_graphs")
LINKS_JSON_PATH = "scripts/download/metadata/download_links.json"
RAW_VIDEOS_DIR = os.path.join(PROJECT_DIR, "raw_videos")
METADATA_FILE = os.path.join(PROJECT_DIR, "clean_videos_metadata.json")

# ==============================================================================

def load_annotations():
    logging.info("--- 1. Loading Annotations ---")
    csv_errors = os.path.join(ANNOTATIONS_DIR, "error_annotations.csv")
    csv_rec_step = os.path.join(ANNOTATIONS_DIR, "recording_id_step_idx.csv")
    csv_activity = os.path.join(ANNOTATIONS_DIR, "activity_idx_step_idx.csv")
    
    if not all(os.path.exists(p) for p in [csv_errors, csv_rec_step, csv_activity]):
        raise FileNotFoundError("Some annotation CSV files are missing.")

    df_errors = pd.read_csv(csv_errors, sep=None, engine='python')
    df_rec_step = pd.read_csv(csv_rec_step, sep=None, engine='python')
    df_activity = pd.read_csv(csv_activity, sep=None, engine='python')

    for df in [df_errors, df_rec_step, df_activity]:
        df.columns = df.columns.str.strip()
    return df_errors, df_rec_step, df_activity

def get_clean_video_list(df_errors, df_rec_step, df_activity):
    logging.info("--- 2. Filtering Videos (No Errors, No Repetitions, TaskGraph Present) ---")
    map_col = 'activity_idx' if 'activity_idx' in df_activity.columns else 'activity_id'
    activity_map = df_activity.groupby(map_col)['activity_name'].first().to_dict()

    video_error_sums = df_errors.groupby('recording_id')['has_errors'].sum()
    clean_ids = set(str(x) for x in video_error_sums[video_error_sums == 0].index.tolist())

    valid_targets = []
    rec_act_col = 'activity_id'
    
    for _, row in df_rec_step.iterrows():
        rec_id = str(row['recording_id'])
        if rec_id not in clean_ids: continue
        try:
            steps = ast.literal_eval(row['step_indices'])
        except: continue
            
        if len(steps) == len(set(steps)):
            act_id = row[rec_act_col]
            raw_act_name = activity_map.get(act_id, "Unknown")
            normalized_name = raw_act_name.lower().replace(" ", "")
            task_graph_path = os.path.join(TASK_GRAPHS_DIR, f"{normalized_name}.json")
            
            if os.path.exists(task_graph_path):
                valid_targets.append({
                    "video_id": rec_id,
                    "activity_name": normalized_name,
                    "task_graph_path": task_graph_path,
                    "raw_activity_name": raw_act_name
                })
    
    logging.info(f"Found {len(valid_targets)} total valid videos in the dataset.")
    return valid_targets

def download_video(video_id):
    # (Identical code as before...)
    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    output_path = os.path.join(RAW_VIDEOS_DIR, f"{video_id}.mp4")
    
    if os.path.exists(output_path):
        return output_path
    
    if not os.path.exists(LINKS_JSON_PATH):
        logging.error(f"Link file not found: {LINKS_JSON_PATH}")
        return None
        
    with open(LINKS_JSON_PATH, "r") as f:
        links_db = json.load(f)
        
    if video_id in links_db:
        url = links_db[video_id].get("gopro_360p")
        if url:
            logging.info(f"Downloading {video_id}...")
            try:
                subprocess.run(["wget", "-q", "-O", output_path, url], check=True)
                return output_path
            except:
                return None
    return None

def main():
    # --- ARGUMENT MANAGEMENT ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_videos", type=int, default=DEFAULT_MAX_VIDEOS, 
                        help="Maximum number of videos to download")
    args = parser.parse_args()
    
    limit = args.max_videos
    # --------------------------

    logging.info(f"START DOWNLOADER (Limit: {limit})")
    
    df_err, df_rec, df_act = load_annotations()
    candidates = get_clean_video_list(df_err, df_rec, df_act)
    
    downloaded_metadata = []
    count = 0
    
    for cand in candidates:
        if limit is not None and count >= limit:
            break
            
        path = download_video(cand['video_id'])
        if path:
            cand['local_video_path'] = path
            downloaded_metadata.append(cand)
            count += 1
            logging.info(f"   --> OK: {cand['video_id']}")
    
    with open(METADATA_FILE, "w") as f:
        json.dump(downloaded_metadata, f, indent=4)
        
    logging.info(f"Saved list of {len(downloaded_metadata)} downloaded videos in: {METADATA_FILE}")

if __name__ == "__main__":
    main()