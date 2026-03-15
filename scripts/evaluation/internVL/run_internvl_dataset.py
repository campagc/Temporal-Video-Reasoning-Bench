import json
import os
import random
import torch
import pandas as pd
import sys
import gc
import shutil
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# --- UTILS DIAGNOSTICA ---
def print_memory_status(step_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[MEM {step_name}] Alloc: {allocated:.2f}GB | Res: {reserved:.2f}GB")

# --- FUNZIONI DI RICERCA FILE ---
def find_remote_file(model_path, filename):
    local_path = os.path.join(model_path, filename)
    if os.path.exists(local_path): return local_path
    
    try:
        from transformers.dynamic_module_utils import resolve_trust_remote_code
        return resolve_trust_remote_code(model_path, filename, force_download=False)
    except: pass

    try:
        home = os.path.expanduser("~")
        search_pattern = os.path.join(home, ".cache/huggingface/modules/**/", filename)
        candidates = glob.glob(search_pattern, recursive=True)
        if candidates: return max(candidates, key=os.path.getmtime)
    except: pass
    return None

# --- PATCHER 1: Meta Tensor Fix ---
def apply_patch_meta_tensor(model_path):
    target_file = find_remote_file(model_path, "modeling_intern_vit.py")
    if not target_file: return
    
    print(f"📍 [Patch MetaTensor] File: {target_file}")
    with open(target_file, "r") as f: content = f.read()

    bad_line = "torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]"
    good_line = "torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device='cpu')]"

    if bad_line in content:
        print(f"🔧 Applicazione fix Meta Tensor...")
        with open(target_file, "w") as f: f.write(content.replace(bad_line, good_line))
        print(f"Patch applicata.")
    elif good_line in content:
        print(f"File già corretto.")

# --- PATCHER 2: Attribute Error Fix (CORRETTO E MIGLIORATO) ---
def apply_patch_attribute_error(model_path):
    target_file = find_remote_file(model_path, "modeling_internvl_chat.py")
    if not target_file: return
    
    print(f"📍 [Patch Attribute] File: {target_file}")
    with open(target_file, "r") as f: content = f.read()

    # Definiamo le stringhe
    search_init = "super().__init__(config)"
    
    # LA PATCH VECCHIA (CHE HA CAUSATO L'ERRORE DI OGGI)
    bad_patch = "self.all_tied_weights_keys = self._tied_weights_keys"
    
    # LA PATCH NUOVA (ROBUSTA: usa un dict vuoto se è None)
    good_patch = "self.all_tied_weights_keys = getattr(self, '_tied_weights_keys', None) or {}"
    
    # La stringa di sostituzione completa con indentazione corretta (8 spazi)
    replacement = f"super().__init__(config)\n        {good_patch}"

    # 1. Controlla se c'è la patch vecchia e sostituiscila
    if bad_patch in content and good_patch not in content:
        print(f"🔧 UPGRADE: Sostituzione patch difettosa con quella robusta...")
        new_content = content.replace(bad_patch, good_patch)
        with open(target_file, "w") as f: f.write(new_content)
        print(f"Patch aggiornata.")
        return

    # 2. Controlla se è già a posto
    if good_patch in content:
        print(f"File già corretto (Patch Robusta presente).")
        return

    # 3. Applica patch da zero
    if search_init in content:
        print(f"🔧 Applicazione patch Attributo (Fresh)...")
        # Sostituisce solo la prima occorrenza (quella nell'__init__)
        new_content = content.replace(search_init, replacement, 1)
        with open(target_file, "w") as f: f.write(new_content)
        print(f"Patch applicata.")

# --- PREPROCESSING UFFICIALE INTERNVL ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=1):
    try:
        image = Image.open(image_file).convert('RGB')
    except Exception as e:
        print(f"Errore apertura immagine {image_file}: {e}")
        return torch.zeros((1, 3, input_size, input_size))

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# --- LOGICA INFERENZA ---
def run_inference_internvl(model_path, manifest_path, output_path, num_clips, seed_offset=0):
    print(f"\n  INIT INTERNVL INFERENCE")
    print_memory_status("Start")
    
    # 0. APPLICA LE PATCH
    print("🚑 Avvio Auto-Patcher...")
    apply_patch_meta_tensor(model_path)
    apply_patch_attribute_error(model_path)

    # 1. TOKENIZER
    print("⏳ Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, fix_mistral_regex=True)
    except:
        print(f"Fallback Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # 2. MODEL LOADING
    print("⏳ Loading Model...")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # FORCE CPU LOAD (Fix Meta Tensor)
        model = AutoModel.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=False, 
            trust_remote_code=True,
            device_map=None
        )
        print(f"...Moving to GPU (Bfloat16)...")
        model = model.eval().to(torch.bfloat16).cuda()
        print_memory_status("Ready")

    except Exception as e:
        print(f"\n FATAL ERROR LOADING MODEL: {e}")
        raise e

    # 3. MANIFEST
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    df = pd.DataFrame(manifest)
    grouped = list(df.groupby('video_id'))
    results_data = []
    
    generation_config = dict(max_new_tokens=512, do_sample=False)

    print(f"Processing {len(grouped)} videos...")
    pbar = tqdm(grouped)
    
    for video_id, group in pbar:
        try:
            clips_data = []
            activity_name = group.iloc[0]['activity_name']
            clean_activity = activity_name.replace("_", " ")

            for _, row in group.iterrows():
                clips_data.append({
                    "original_step_id": row['step_id'],
                    "taskgraph_id": row['taskgraph_id'],
                    "frames": row['frames'],
                    "label": row['label']
                })

            rng = random.Random(f"{video_id}_{seed_offset}")
            rng.shuffle(clips_data)
            
            if len(clips_data) < num_clips: continue
            selected_clips = clips_data[:num_clips]
            random_ids = rng.sample(range(100, 1000), len(selected_clips))

            pixel_values_list = []
            num_patches_list = []
            input_map = []
            
            ids_list_str = ", ".join([str(sid) for sid in random_ids])
            prompt_text = (
                f"You are given {len(selected_clips)} shuffled segments of a video demonstrating the activity: '{clean_activity}'.\n"
                f"Each segment has a unique ID ({ids_list_str}).\n"
            )
            
            for idx, clip in enumerate(selected_clips):
                sid = str(random_ids[idx])
                prompt_text += f"Segment ID {sid}:\n"
                for frame_path in clip['frames']:
                    prompt_text += "<image>" 
                    pv = load_image(frame_path, max_num=1).to(torch.bfloat16).cuda()
                    pixel_values_list.append(pv)
                    num_patches_list.append(pv.size(0))
                prompt_text += "\n"
                input_map.append({"shuffled_id": sid, "original_step_id": clip["original_step_id"], "taskgraph_id": clip["taskgraph_id"]})

            prompt_text += (
                f"\nYour Goal: Reorder these segments into the correct chronological sequence for '{clean_activity}'.\n\n"
                f"Instructions:\n"
                f"1. ANALYZE: Briefly describe what is happening in each Segment ID.\n"
                f"2. REASON: Explain the logical order based on cooking procedure.\n"
                f"3. SOLVE: Output the final sequence.\n\n"
                "STRICT OUTPUT FORMAT:\n"
                "Analysis: [Brief descriptions]\n"
                "Reasoning: [Step-by-step logic]\n"
                f"Final Sequence: {random_ids[-1]}->{random_ids[0]}->... (Use actual IDs)\n"
            )

            if pixel_values_list:
                pixel_values = torch.cat(pixel_values_list, dim=0)
            else:
                pixel_values = None

            response = model.chat(
                tokenizer, 
                pixel_values, 
                prompt_text, 
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )
            
            results_data.append({
                "video_id": video_id,
                "activity_name": activity_name,
                "model_raw_output": response,
                "input_map": input_map
            })
            
            if len(results_data) % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            tqdm.write(f" Error video {video_id}: {e}")
            torch.cuda.empty_cache()
            continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=4)