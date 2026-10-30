import os
import tarfile
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfFileSystem

DATASET_ID = "united-we-care/United-Syn-Med"
OUTPUT_DIR = "gold_eval_dataset"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

os.makedirs(AUDIO_DIR, exist_ok=True)

print("--- Building Gold Standard Clinical Micro-Dataset ---")

# 1. Stream the CSV metadata
dataset = load_dataset(
    "csv",
    data_files=f"hf://datasets/{DATASET_ID}/data/validation.csv",
    streaming=True,
)

# Define keywords to hunt for high-value pharmacology/clinical terms
keywords = ["mg", "tablet", "capsule", "prescription", "dose", "syrup", "ml", "dr", "patient"]

selected_samples = []
target_files = set()

# Scan stream to find up to 15-20 rich clinical samples
for sample in dataset["train"]:
    text = sample.get("transcription", "").lower()
    file_name = sample.get("file_name", "")
    
    # Check if text contains clinical/pharmacology markers
    if any(kw in text for kw in keywords) and file_name:
        selected_samples.append(sample)
        target_files.add(file_name)
        
        if len(selected_samples) >= 15:  # Target micro-dataset size
            print(f"Targeted {len(selected_samples)} high-value clinical samples.")
            break

# 2. Use HfFileSystem to stream-extract ONLY the target audio files without full download
print("Streaming targeted audio files from remote repository...")
fs = HfFileSystem()
tar_path = f"datasets/{DATASET_ID}/data/audio/validation.tar.gz"

extracted_count = 0
with fs.open(tar_path, "rb") as f_remote:
    with tarfile.open(fileobj=f_remote, mode="r:gz") as tar:
        for member in tar.getmembers():
            base_name = os.path.basename(member.name)
            if base_name in target_files:
                # Extract file contents into our local micro-dataset folder
                extracted_file_obj = tar.extractfile(member)
                if extracted_file_obj:
                    local_audio_path = os.path.join(AUDIO_DIR, base_name)
                    with open(local_audio_path, "wb") as f_out:
                        f_out.write(extracted_file_obj.read())
                    extracted_count += 1
                    print(f"  -> Extracted: {base_name}")

# 3. Save the micro-dataset metadata CSV
metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
df_micro = pd.DataFrame(selected_samples)
df_micro.to_csv(metadata_path, index=False)

print(f"\nSuccessfully created Gold Micro-Dataset with {extracted_count} audio files at ./{OUTPUT_DIR}/!")