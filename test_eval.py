import os
import time
import pandas as pd
import requests
from jiwer import wer

# Configuration pointing to your local gold micro-dataset
API_URL = "http://localhost:8000/transcribe"
METADATA_PATH = "gold_eval_dataset/metadata.csv"
AUDIO_DIR = "gold_eval_dataset/audio"

print("--- Starting Local Micro-Dataset Benchmark Pipeline ---")

# 1. Load your local micro-dataset metadata
if not os.path.exists(METADATA_PATH):
    print(f"Error: {METADATA_PATH} not found. Did you run the build script?")
    exit(1)

df = pd.read_csv(METADATA_PATH)
metrics = []

# 2. Iterate through your curated gold samples
for i, row in df.iterrows():
    file_name = row["file_name"]
    ground_truth = row["transcription"]
    audio_path = os.path.join(AUDIO_DIR, file_name)

    print(f"\n[Sample {i+1}] File: {file_name}")
    print(f"  -> Ground Truth: {ground_truth}")

    if not os.path.exists(audio_path):
        print(f"  -> Warning: Audio file {audio_path} missing locally.")
        continue

    # Read local audio bytes
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # 3. Send to your Docker container and measure latency
    files = {"file": (file_name, audio_bytes, "audio/mpeg")}
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, files=files, timeout=30)
        latency = time.time() - start_time

        if response.status_code == 200:
            prediction = response.json().get("transcription", "")
            print(f"  -> Prediction:   {prediction}")
            print(f"  -> Latency:      {latency:.3f} seconds")

            # 4. Compute Word Error Rate (WER)
            sample_wer = wer(ground_truth, prediction)
            print(f"  -> WER:          {sample_wer:.4f}")

            metrics.append({
                "file_name": file_name,
                "latency": latency,
                "wer": sample_wer
            })
        else:
            print(f"  -> Server Error {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"  -> Connection Error (Is your Docker container running on port 8000?): {e}")

# Summary Statistics
if metrics:
    avg_latency = sum(m["latency"] for m in metrics) / len(metrics)
    avg_wer = sum(m["wer"] for m in metrics) / len(metrics)
    print("\n==============================")
    print("      GOLD BENCHMARK SUMMARY  ")
    print("==============================")
    print(f"Total Samples Evaluated: {len(metrics)}")
    print(f"Average Latency:         {avg_latency:.3f}s")
    print(f"Average WER:             {avg_wer:.4f}")
    print("==============================")