import csv
import os
import re
import io
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

# 1. Configuration
DATASET_ID = "AbstractTTS/IEMOCAP"
OUTPUT_DIR = "metadata_common_6"  # Changed directory name for clarity
AUDIO_SAVE_DIR = "iemocap_wavs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)

# --- THE FIX: NEW MAPPING LOGIC ---
# Goal: Align with EmoDB Common-6 Schema
# Target IDs:
# 0: Anger
# 1: Sadness
# 2: Happiness (Merge 'happy' and 'excited')
# 3: Neutral
# 4: Fear
# 5: Disgust
# Dropped: 'frustrated', 'surprise', 'other' (not in Common-6)

emotion_map_common6 = {
    "angry": 0,
    "sad": 1,
    "happy": 2,
    "excited": 2,   # Merging Excited into Happiness
    "neutral": 3,
    "fear": 4,
    "disgust": 5,
    # "frustrated": -1, # Will be ignored automatically
    # "surprise": -1    # Will be ignored automatically
}

def get_session_number(file_name):
    match = re.search(r'Ses0(\d)', file_name)
    return int(match.group(1)) if match else 0

def main():
    print(f"Loading {DATASET_ID}...")
    try:
        ds = load_dataset(DATASET_ID, split="train")
        # Disable internal decoding to bypass TorchCodec issues
        ds = ds.cast_column("audio", Audio(decode=False))
        print("Dataset loaded. Starting manual extraction...")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    rows = []
    skipped_counts = {"frustrated": 0, "surprise": 0, "other": 0}
    
    for i, sample in enumerate(ds):
        if i % 100 == 0:
            print(f" Progress: {i}/{len(ds)} samples processed")
            
        emo_str = sample["major_emotion"]
        
        # Check if emotion is in our target map
        if emo_str in emotion_map_common6:
            label = emotion_map_common6[emo_str]
            
            # --- AUDIO SAVING LOGIC ---
            filename = sample["file"] 
            if not filename.endswith(".wav"):
                filename += ".wav"
            
            abs_path = os.path.abspath(os.path.join(AUDIO_SAVE_DIR, filename))
            
            if not os.path.exists(abs_path):
                try:
                    audio_bytes = sample["audio"]["bytes"]
                    with io.BytesIO(audio_bytes) as f:
                        audio_array, sampling_rate = sf.read(f)
                    
                    if sampling_rate != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
                    
                    sf.write(abs_path, audio_array, 16000)
                except Exception as e:
                    print(f"  Warning: Could not save {filename}: {e}")
                    continue
            
            # Collect metadata
            text = sample["transcription"]
            session_num = get_session_number(sample["file"])
            rows.append([abs_path, text, label, session_num])
        else:
            # Track what we are skipping
            if emo_str in skipped_counts:
                skipped_counts[emo_str] += 1
            else:
                skipped_counts["other"] += 1

    print(f"\nProcessing complete. Skipped emotions: {skipped_counts}")
    print(f"Retained {len(rows)} samples compatible with Common-6 schema.")

    # Save CSV Files
    header = ["audio_file", "raw_text", "label", "session"]

    # Standard split logic (Session 1-3 Train, 4 Val, 5 Test)
    splits = {
        "train": [r for r in rows if r[3] in [1, 2, 3]],
        "val":   [r for r in rows if r[3] == 4],
        "test":  [r for r in rows if r[3] == 5]
    }

    for name, split_rows in splits.items():
        path = os.path.join(OUTPUT_DIR, f"IEMOCAP_Common6_{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(split_rows)
        print(f"Split '{name}' saved with {len(split_rows)} samples.")

if __name__ == "__main__":
    main()