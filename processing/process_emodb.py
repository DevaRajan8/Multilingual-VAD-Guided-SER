import csv
import os
import shutil
import soundfile as sf
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Configuration
# REPLACE THIS with the path to your downloaded folder containing the .wav files
SOURCE_AUDIO_DIR = "./wav" 

OUTPUT_DIR = "metadata_emodb"
PROCESSED_AUDIO_DIR = "emodb_wavs_16k"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

# 2. Mappings based on EmoDB Documentation
# Code -> Text
TRANSCRIPTIONS = {
    "a01": "Der Lappen liegt auf dem Eisschrank.",
    "a02": "Das will sie am Mittwoch abgeben.",
    "a04": "Heute Abend könnte ich es ihm sagen.",
    "a05": "Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.",
    "a07": "In sieben Stunden wird es soweit sein.",
    "b01": "Was sind denn das für Tüten, die da unter dem Tisch stehen?",
    "b02": "Sie haben es gerade hoch getragen und jetzt gehen sie wieder runter.",
    "b03": "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.",
    "b09": "Ich will das eben wegbringen und dann mit Karl was trinken gehen.",
    "b10": "Die wird auf dem Platz sein, wo wir sie immer hinlegen."
}

# Code -> Emotion Label (String)
EMOTION_CODE_MAP = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral"
}

# Emotion Label -> Integer ID (for training)
# You can adjust these integers if you want to match specific model heads
LABEL_MAP = {
    "anger": 0,
    "boredom": 1,
    "disgust": 2,
    "fear": 3,
    "happiness": 4,
    "sadness": 5,
    "neutral": 6
}

def process_audio(src_path, dest_path, target_sr=16000):
    """
    Reads audio, resamples to target_sr, and saves to dest_path.
    Returns True if successful, False otherwise.
    """
    try:
        # Load with librosa (automatically resamples if needed)
        y, sr = librosa.load(src_path, sr=target_sr)
        
        # Save using soundfile
        sf.write(dest_path, y, target_sr)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def main():
    if not os.path.exists(SOURCE_AUDIO_DIR):
        print(f"Error: Source directory '{SOURCE_AUDIO_DIR}' not found.")
        return

    print(f"Scanning {SOURCE_AUDIO_DIR}...")
    
    files = [f for f in os.listdir(SOURCE_AUDIO_DIR) if f.endswith(".wav")]
    print(f"Found {len(files)} audio files.")

    metadata_rows = []
    
    for idx, filename in enumerate(files):
        if idx % 50 == 0:
            print(f"Processing: {idx}/{len(files)}")

        # --- PARSING LOGIC based on EmoDB docs ---
        # Example: "03a01Nc.wav"
        # 0-1: Speaker (03)
        # 2-4: Sentence Code (a01)
        # 5:   Emotion Code (N)
        # 6:   Repetition (c)
        
        try:
            speaker_id = filename[0:2]
            sentence_code = filename[2:5]
            emotion_code = filename[5]
            
            # 1. Get Text
            text = TRANSCRIPTIONS.get(sentence_code)
            if not text:
                print(f"Warning: Unknown sentence code '{sentence_code}' in {filename}. Skipping.")
                continue

            # 2. Get Label
            emotion_str = EMOTION_CODE_MAP.get(emotion_code)
            if not emotion_str:
                print(f"Warning: Unknown emotion code '{emotion_code}' in {filename}. Skipping.")
                continue
            
            label_id = LABEL_MAP[emotion_str]

            # 3. Process Audio (Resample & Save)
            src_abs_path = os.path.join(SOURCE_AUDIO_DIR, filename)
            dest_abs_path = os.path.join(PROCESSED_AUDIO_DIR, filename)
            
            # Only process if file doesn't already exist (resume capability)
            if not os.path.exists(dest_abs_path):
                success = process_audio(src_abs_path, dest_abs_path)
                if not success:
                    continue
            
            # 4. Collect Metadata
            # Note: We use absolute path for the CSV to avoid path issues later
            abs_path_str = os.path.abspath(dest_abs_path)
            metadata_rows.append([abs_path_str, text, label_id, speaker_id])

        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            continue

    # --- CSV WRITING ---
    header = ["audio_file", "raw_text", "label", "speaker_id"]
    
    # 1. Save Full Metadata
    full_csv_path = os.path.join(OUTPUT_DIR, "EmoDB_full.csv")
    with open(full_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(metadata_rows)
    print(f"\nFull metadata saved to {full_csv_path}")

    # --- SPLITTING LOGIC (Speaker Independent) ---
    # EmoDB usually has 10 speakers. We should split by speaker to prevent leakage.
    unique_speakers = sorted(list(set(r[3] for r in metadata_rows)))
    print(f"Identified Speakers: {unique_speakers}")

    if len(unique_speakers) >= 3:
        # Reserve last 2 speakers for test/val, rest for train
        # (Standard EmoDB practice varies, but 80/10/10 or Leave-One-Speaker-Out is common)
        test_speakers = [unique_speakers[-1]]      # Last speaker
        val_speakers = [unique_speakers[-2]]       # Second to last
        train_speakers = unique_speakers[:-2]      # The rest
    else:
        # Fallback for tiny subsets (random split)
        print("Warning: Too few speakers for speaker-split. Doing random split.")
        train_speakers = unique_speakers
        val_speakers = []
        test_speakers = []

    splits = {
        "train": [r for r in metadata_rows if r[3] in train_speakers],
        "val":   [r for r in metadata_rows if r[3] in val_speakers],
        "test":  [r for r in metadata_rows if r[3] in test_speakers]
    }

    # If fallback random split is needed (only if unique speakers < 3)
    if not splits["val"] and not splits["test"]:
        train_data, test_data = train_test_split(metadata_rows, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        splits = {"train": train_data, "val": val_data, "test": test_data}

    for name, rows in splits.items():
        path = os.path.join(OUTPUT_DIR, f"EmoDB_{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Split '{name}' saved with {len(rows)} samples.")

if __name__ == "__main__":
    main()