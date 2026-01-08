# Video Procedural Reasoning Pipeline (Single Video Prototype)

**Last Updated:** January 8, 2026

## 📖 Project Overview
This project provides a pipeline for testing **Qwen2.5-VL** on it's ability to understand and reason about procedural activities (e.g., cooking recipes).

**Current Status:** The code is currently optimized to process **one video at a time** and specifically for **testing**

The workflow consists of:
1.  Downloading a specific video from the CaptainCook4D dataset.
2.  Preprocessing and extracting frames for specific steps (clips).
3.  Shuffling the clips and asking the VLM to reconstruct the chronological order.
4.  Evaluating the model's performance against a Ground Truth Task Graph and a Random Baseline.

---

## 🚀 How to Run

The main entry point is the shell script `run_pipeline.sh`. It manages the environment, configuration, and sequential execution of Python scripts.

Run the full pipeline on the default video ID:
```bash
./run_pipeline.sh
```
The script accepts two arguments: `START_STEP` and `VIDEO_ID`.

```bash
# Syntax: ./run_pipeline.sh [START_STEP] [VIDEO_ID]

# Example: Run from scratch (Step 0) for video ID "10_1"
./run_pipeline.sh 0 "10_1"

# Example: Skip download/preprocessing, start directly from Inference (Step 2)
./run_pipeline.sh 2
```
You can also override default parameters (Frames per clip, Number of clips to test, Number of repeated runs) directly from the command line:

```bash
# Extract 16 frames per clip, test with 10 clips, and average results over 5 runs
NUM_FRAMES=16 NUM_CLIPS=10 NUM_RUNS=5 ./run_pipeline.sh 0 "9_12"
```

### Pipeline Steps Definition
*   **Step 0:** Download Video (`download_single_video.py`)
*   **Step 1:** Preprocess/Frame Extraction (`preprocess.py`)
*   **Step 2:** Model Inference on GPU (`run_native_repeated.py`)
*   **Step 3:** Calculate Metrics (`evaluate_metrics_dual.py`)

### Configuration Parameters
*   **NUM_FRAMES** (STEP 1) The number of frames to extract per video clip
*   **NUM_CLIP** (STEP 2) The size of the subset to test. All clips may be too much for the model
*   **NUM_RUNS** How many times to repeat the inference on the specific subset

---

## 📂 Scripts Description

Here is a summary of the Python scripts located in the `scripts/` directory:

**`analyze_clean_videos_capck4d.py`**: Scans the annotation CSVs to find videos with zero recorded errors and no repeated steps and generates a `clean_videos_list.txt` containing reliable video IDs for testing.

**`download_single_video.py`**: Downloads the raw MP4 file for a specific Video ID using the metadata links JSON. It defaults to the 360p GoPro version.

**`preprocess.py`**: Reads the downloaded video and corresponding step annotations. It extracts high-quality frames for every step (controlled by `NUM_FRAMES`), resizes them, and maps the specific video steps to abstract Task Graph nodes via text description matching. It outputs the `dataset_manifest.json`.

**`run_native_repeated.py`**: The core inference engine using **Qwen2.5-VL**. It loads a subset of clips (controlled by `NUM_CLIPS`), shuffles them, and asks the model to reconstruct the chronological timeline. This process is repeated `NUM_RUNS` times.

**`evaluate_metrics_dual.py`**: Benchmarks the model's output by comparing it against the Ground Truth. It calculates DAG-based metrics (checking if logical dependencies from the Task Graph are respected) and Exact Match metrics. It also simulates 200 random runs to provide a baseline comparison.

---

## 🛠 Requirements
*   **Hardware:** NVIDIA GPU (required for Step 2).
*   **Data:** Requires the `CaptainCook4D` annotations in `data/annotations_repo/` and videos.
