import os
import json
import argparse
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
import networkx as nx
from collections import defaultdict

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
DEFAULT_FRAMES_PER_CLIP = config.get("project", {}).get("default_num_frames", 6)
MAX_SIDE = config.get("project", {}).get("max_side", 512)
DATA_ROOT = "data/comkitchens"
TASK_GRAPHS_DIR = "data/annotations_repo/task_graphs_com"
OUTPUT_FRAMES_ROOT_TEMPLATE = "data/processed_frames_com_{}"

# ==============================================================================
# PARSING & GRAFI
# ==============================================================================

def load_and_patch_json(json_path, trans_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if os.path.exists(trans_path):
        try:
            with open(trans_path, 'r', encoding='utf-8') as f:
                trans = json.load(f)
            if "ingredients" in trans:
                data["ingredients"] = trans["ingredients"]
            if "steps" in trans:
                for i, step in enumerate(data["steps"]):
                    if i < len(trans["steps"]):
                        if "memo" in trans["steps"][i]:
                            step["memo"] = trans["steps"][i]["memo"]
        except Exception as e:
            logging.warning(f"Translation warning: {e}")
    return data

def get_frame_number(meta_part):
    if not meta_part:
        return None
    try:
        if isinstance(meta_part, list):
            if len(meta_part) > 0 and 'frame' in meta_part[0]:
                return int(meta_part[0]['frame'])
        elif isinstance(meta_part, dict):
            if 'frame' in meta_part:
                return int(meta_part['frame'])
    except:
        return None
    return None

def build_graph_and_labels(annotation):
    """
    Costruisce il grafo usando ID univoci per evitare collisioni pid-aid.
    Ritorna: G, node_descriptions, action_uid_map
    """
    G = nx.DiGraph()
    node_descriptions = {}
    
    # 1. MAPPING ID UNIVOCI
    # Action ID in COM non è univoco (si ripete per ogni persona).
    # Creiamo un mapping "pid-aid" -> ID progressivo (1, 2, 3...)
    action_uid_map = {}
    uid_counter = 1
    
    actions = annotation.get("actions_by_person", {})
    
    # Assegnazione ID
    for pid, acts in actions.items():
        for aid in acts.keys():
            action_uid_map[f"{pid}-{aid}"] = uid_counter
            uid_counter += 1

    # 2. COSTRUZIONE GRAFO
    for pid, acts in actions.items():
        for aid, details in acts.items():
            # Recupera ID univoco
            current_key = f"{pid}-{aid}"
            node_id = action_uid_map[current_key]
            
            # Descrizione
            step_idx = details.get("step_index")
            desc = ""
            if step_idx is not None and step_idx < len(annotation["steps"]):
                desc = annotation["steps"][step_idx].get("memo", "")
            
            desc = desc.strip()
            if desc.endswith("."): desc = desc[:-1]
            
            node_descriptions[node_id] = desc
            G.add_node(node_id)
            
            # Archi
            next_actions = details.get("is_input_of", [])
            if next_actions:
                for target_str in next_actions:
                    target_str = str(target_str)
                    if target_str in action_uid_map:
                        target_id = action_uid_map[target_str]
                        G.add_edge(node_id, target_id)
                        
    return G, node_descriptions, action_uid_map

# ==============================================================================
#  FILTRI AVANZATI
# ==============================================================================
"""
    Analizza se il grafo è lineare e che tipo di duplicati ha.
    Ritorna: is_linear, has_disjoint_duplicates
    """

def analyze_topology_and_duplicates(G, labels_dict):
    # Check Lineartà
    is_linear = False
    if len(G.nodes) >= 2:
        max_out = max([d for n, d in G.out_degree()], default=0)
        max_in = max([d for n, d in G.in_degree()], default=0)
        if max_out <= 1 and max_in <= 1:
            is_linear = True
    else:
        is_linear = True

    # Check Duplicati
    labels_list = [t.lower().strip() for t in labels_dict.values() if t]
    unique_labels = set(labels_list)
    
    total_duplicates = len(labels_list) - len(unique_labels)
    
    # Sequential
    sequential_duplicates = 0
    for u, v in G.edges():
        l_u = labels_dict.get(u, "").lower().strip()
        l_v = labels_dict.get(v, "").lower().strip()
        if l_u and l_v and l_u == l_v:
            sequential_duplicates += 1
            
    # Disjoint
    disjoint_duplicates = max(0, total_duplicates - sequential_duplicates)
    
    has_disjoint = disjoint_duplicates > 0
    
    return is_linear, has_disjoint

# ==============================================================================
# 🎬 PROCESSING
# ==============================================================================

def process_video_folder(recipe_path, output_frames_dir, num_frames, args):
    json_file = os.path.join(recipe_path, "gold_recipe.json")
    trans_file = os.path.join(recipe_path, "gold_recipe_translation_en.json")
    video_file = os.path.join(recipe_path, "front_compressed.mp4")
    
    if not os.path.exists(json_file) or not os.path.exists(video_file):
        return [], "missing_files"

    data = load_and_patch_json(json_file, trans_file)
    
    recipe_id = data.get("recipe_id", "unknown")
    kitchen_id = str(data.get("kitchen_id", "unknown"))
    video_unique_id = f"{recipe_id}_{kitchen_id}"
    
    # --- 1. Costruzione Grafo Corretto ---
    G, node_descs, uid_map = build_graph_and_labels(data)
    
    # Check base nodi
    if len(G.nodes) < 3:
        return [], "rejected_too_short"

    # --- 2. Filtri Avanzati ---
    is_linear, has_disjoint = analyze_topology_and_duplicates(G, node_descs)
    
    if args.filter_linear and is_linear:
        return [], "rejected_linear"
    
    if args.filter_disjoint and has_disjoint:
        return [], "rejected_disjoint_dups" 
        # Nota: I duplicati SEQUENZIALI passano qui!

    # --- 3. Check Clip Valide e Mapping ID ---
    actions = data.get("actions_by_person", {})
    valid_actions = []
    
    for pid, acts in actions.items():
        for aid, details in acts.items():
            meta = details.get("meta_info") or details.get("metainfo")
            start = get_frame_number(meta.get("before") if meta else None)
            end = get_frame_number(meta.get("after") if meta else None)
            
            if start is not None and end is not None and end > start:
                # Recuperiamo l'ID univoco calcolato in build_graph
                key = f"{pid}-{aid}"
                if key in uid_map:
                    real_id = uid_map[key]
                    valid_actions.append((real_id, start, end))
    
    if len(valid_actions) < 3:
        return [], "rejected_no_valid_clips"

    # --- 4. Salvataggio Task Graph ---
    os.makedirs(TASK_GRAPHS_DIR, exist_ok=True)
    graph_path = os.path.join(TASK_GRAPHS_DIR, f"{video_unique_id}.json")
    
    # Salviamo il grafo usando gli ID univoci (1..N)
    graph_data = {
        "activity": video_unique_id,
        "edges": list(G.edges()),
        "steps": {str(k): v for k, v in node_descs.items()}
    }
    with open(graph_path, 'w') as f:
        json.dump(graph_data, f, indent=4)

    # --- 5. Estrazione Frame ---
    try:
        vr = VideoReader(video_file, ctx=cpu(0))
    except:
        return [], "video_error"
        
    entries = []
    # Ordiniamo per ID per pulizia (non strettamente necessario ma utile per debug)
    valid_actions.sort(key=lambda x: x[0])
    
    for (real_id, start, end) in valid_actions:
        # Usiamo real_id per il nome folder (es. ..._act_12)
        clip_name = f"{video_unique_id}_act_{real_id}"
        step_folder = os.path.join(output_frames_dir, clip_name)
        
        # Skip se già esiste (opzionale, utile per resume)
        if not os.path.exists(step_folder):
            os.makedirs(step_folder, exist_ok=True)
            
            frame_indices = np.linspace(start, end-1, num_frames, dtype=int)
            frame_indices = np.clip(frame_indices, 0, len(vr)-1)
            
            try:
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
            except Exception as e:
                logging.error(f"Frame error for {clip_name}: {e}")
                continue
        else:
            # Se esiste già, ricostruiamo solo i path per il manifest
            saved_frames = [os.path.join(step_folder, f"frame_{i}.jpg") for i in range(num_frames)]

        entries.append({
            "video_id": video_unique_id,
            "step_id": real_id,         # ID 1..N coerente col grafo
            "taskgraph_id": real_id,    # ID 1..N coerente col grafo
            "label": node_descs.get(real_id, "unknown"),
            "activity_name": video_unique_id,
            "frames": saved_frames
        })
            
    return entries, "ok"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES_PER_CLIP)
    # Di default filtriamo i lineari e i duplicati DISGIUNTI (i sequenziali vanno bene)
    parser.add_argument("--filter_linear", action="store_true", default=True)
    parser.add_argument("--filter_disjoint", action="store_true", default=True)
    args = parser.parse_args()
    
    output_root = OUTPUT_FRAMES_ROOT_TEMPLATE.format(args.frames)
    
    logging.info(f"Processing COM Kitchens -> {output_root}")
    logging.info(f"Filter Linear: {args.filter_linear}")
    logging.info(f"Filter Disjoint Dups: {args.filter_disjoint} (Sequential Dups ACCEPTED)")
    
    manifest = []
    stats = defaultdict(int)
    
    for root, dirs, files in os.walk(DATA_ROOT):
        if "gold_recipe.json" in files:
            vid_entries, status = process_video_folder(root, output_root, args.frames, args)
            stats[status] += 1
            if status == "ok":
                manifest.extend(vid_entries)
                logging.info(f"OK: {os.path.basename(root)} ({len(vid_entries)} clips)")
            else:
                # print(f"Skip: {os.path.basename(root)} -> {status}")
                pass
            
    man_path = os.path.join(output_root, "dataset_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    logging.info("================ REPORT ================")
    logging.info(f"Total Scanned:      {sum(stats.values())}")
    logging.info(f"Processed valid:    {stats['ok']}")
    logging.info(f"Discarded (Linear): {stats['rejected_linear']}")
    logging.info(f"Discarded (Disjoint): {stats['rejected_disjoint_dups']}")
    logging.info(f"Various problems:   {stats['rejected_too_short'] + stats['rejected_no_valid_clips'] + stats['video_error']}")
    logging.info(f"Manifest saved at:  {man_path}")

if __name__ == "__main__":
    main()