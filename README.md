# Video Reasoning Research Project

This repository contains the code, evaluation scripts, and visualization tools for a video reasoning research project evaluating Large Vision-Language Models (such as InternVL and Qwen) on complex, multi-step video tasks. The pipeline includes downloading videos, extracting frames, running model inferences in a grid-search manner, and generating detailed analytical visualizations.

---

## 1. Environment & Dependencies

The project relies on a Conda environment (here named `video_env`). All required packages for handling video processing, LVLMs, and hardware optimizations are pre-configured in the provided `.yml` file.

### Setup Instructions
You simply need to create and activate the Conda environment using the provided file:
```bash
conda env create -f environment.yml
conda activate video_env

```

---

## 2. Data Setup & Manual Prerequisites

Before running the automated pipelines, you must manually download the core datasets (CaptainCook4D and ComKitchens) and place them in the correct directories.

### Expected Manual Downloads

1. **CaptainCook4D Annotations:** Download the official annotations repository and place it inside `data/annotations_repo/`. This folder must contain the `annotation_csv/`, `annotation_json/`, and the `task_graphs/` directories.
2. **Video Download Links:** The automated downloader requires a JSON mapping file containing the actual video URLs for CaptainCook4D. Place this file at:
`scripts/download/metadata/download_links.json`
3. **ComKitchens Dataset:** Download the raw ComKitchens dataset and place it entirely inside `data/comkitchens/`. Ensure its respective task graphs are placed in `data/annotations_repo/task_graphs_com/`.

### Expected Directory Structure

Your project tree should look like this before you start running any scripts:

```text
.
├── config.yaml
├── config_experiments_internvl.txt
├── config_experiments_qwen.txt
├── scripts/
│   └── download/
│       └── metadata/
│           └── download_links.json         <-- (Manual) Place your URL links here
└── data/
    ├── annotations_repo/                   <-- (Manual) Downloaded CC4D annotations
    │   ├── annotation_csv/                 
    │   ├── annotation_json/                
    │   ├── task_graphs/                    
    │   └── task_graphs_com/                <-- (Manual) ComKitchens task graphs
    └── comkitchens/                        <-- (Manual) Raw ComKitchens dataset files

```

*(Note: Create or verify the `config.yaml` in the root directory to define your dataset root and visual parameters like `default_num_frames: 8` and `max_side: 512`)*.

---

## 3. Automated Data Preparation

Once the manual prerequisites are in place, the scripts in `scripts/download/` will handle filtering, downloading the raw `.mp4` files, and structuring the processed frames.

### Data Pipeline Steps

1. **`download_manager.py`**: Reads `download_links.json` and the CSV annotations to filter out videos with errors. It downloads valid CaptainCook4D `.mp4` files into a new `data/raw_videos/` directory.
2. **`frame_extractor.py`**: Reads the `.mp4` files and task graphs, extracting the specific frames needed per step into a dynamically generated directory like `data/processed_frames_8/`.
3. **`prepare_comkitchens.py`**: A dedicated script that structures and extracts the frames specifically for the ComKitchens dataset.

---

## 4. Running Experiments (Slurm)

Experiment execution is orchestrated via Slurm batch scripts located in the `slurm_jobs/` directory. These scripts iterate through your configuration files (`config_experiments_internvl.txt` and `config_experiments_qwen.txt`) to run sequential grid experiments.

### Data Preparation Jobs

To download datasets and extract frames on your cluster:

```bash
# Download CaptainCook4D videos and extract frames
sbatch slurm_jobs/download/submit_download_and_extract.sh

# Setup ComKitchens specifically
sbatch slurm_jobs/download/submit_prepare_com.sh

# If you only need to run frame extraction
sbatch slurm_jobs/download/submit_onlyframe.sh

```

### Evaluation Jobs

To launch the Large Vision-Language Model evaluations:

```bash
# Run InternVL experiments
sbatch slurm_jobs/evaluation/submit_internvl_sequential.sh

# Run Qwen experiments
sbatch slurm_jobs/evaluation/submit_qwen_sequential.sh

```

*Note: Ensure your Slurm scripts contain the correct `#SBATCH` directives for your cluster's GPU partitions (e.g., A30, L40S, or Hopper GPUs which support Flash Attention).*

Results (metrics in `.csv` and detailed results in `.json`) are automatically saved to output directories like `output_grid_internvl_captaincook/` and `output_grid_experiment_com/`.

---

## 5. Visualization & Results

Once experiments are complete, the `scripts/visualization/` directory contains tools to analyze the `.csv` metric outputs. These tools generate insights into model capabilities, skill areas vs. random baselines, and recipe-specific performance.

You can run these Python scripts directly in your environment:

* **Grid Search Optimization Analysis:**
```bash
python scripts/visualization/visualize_grid_results.py

```


* **Global Model Performance (Skill Area vs Random Baseline):**
```bash
python scripts/visualization/visualize_model_vs_random.py

```


* **Leaderboards & Complexity Categories:**
```bash
python scripts/visualization/analyze_global_leaderboard.py
python scripts/visualization/visualize_recipe_leaderboard.py
python scripts/visualization/analyze_categories.py

```



Outputs from these scripts will typically print aggregated statistics to the console and save/render `.png` or `.pdf` plots into directories like `output_grid_internvl_captaincook/deep_insights/` for direct inclusion in your research paper.
