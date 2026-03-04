import csv
import os
import re
import io
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

TARGET_SR = 16000
TARGET_RMS = 0.1   # Normalize all clips to this RMS loudness level

# 1. Configuration
DATASET_ID = "AbstractTTS/IEMOCAP"
OUTPUT_DIR = "/dist_home/suryansh/sharukesh/speech/metadata"
AUDIO_SAVE_DIR = "/dist_home/suryansh/sharukesh/speech/iemocap_wavs"
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

            # save_ok tracks whether we have a valid preprocessed file.
            # We only add to rows if preprocessing succeeded (or was already done).
            save_ok = os.path.exists(abs_path)

            if not save_ok:
                try:
                    audio_bytes = sample["audio"]["bytes"]

                    # --- Step 1: Decode bytes and resample to 16 kHz / mono ---
                    # We read raw bytes via BytesIO, then use librosa.resample if
                    # the source SR differs. mono=True averages any stereo channels.
                    with io.BytesIO(audio_bytes) as f:
                        audio_array, sampling_rate = sf.read(f)

                    # Convert stereo to mono by averaging channels
                    if audio_array.ndim > 1:
                        audio_array = np.mean(audio_array, axis=1)

                    # Resample if not already 16 kHz
                    if sampling_rate != TARGET_SR:
                        audio_array = librosa.resample(
                            audio_array, orig_sr=sampling_rate, target_sr=TARGET_SR
                        )

                    # --- Step 2: Trim leading/trailing silence ---
                    # top_db=20: frames >20 dB quieter than the peak are treated
                    # as silence and removed. IEMOCAP utterances are already fairly
                    # tight, but this standardizes edge behavior across all datasets.
                    audio_array, _ = librosa.effects.trim(audio_array, top_db=20)

                    # Skip clips that are too short after trimming (<0.3s)
                    if len(audio_array) < TARGET_SR * 0.3:
                        print(f"  Skipping {filename} — too short after trim ({len(audio_array)/TARGET_SR:.2f}s)")
                        continue

                    # --- Step 3: RMS Amplitude Normalization ---
                    # RMS = sqrt(mean(y^2)) measures average signal loudness.
                    # We scale the signal so its RMS equals TARGET_RMS (0.1).
                    # This compensates for loudness differences between IEMOCAP
                    # (naturalistic, variable) and EmoDB/EMOVO (studio-acted).
                    rms = np.sqrt(np.mean(audio_array ** 2))
                    if rms > 1e-6:  # Guard against silent/zero-level clips
                        audio_array = audio_array * (TARGET_RMS / rms)
                    audio_array = np.clip(audio_array, -1.0, 1.0)  # prevent overflow

                    # Save the preprocessed audio
                    sf.write(abs_path, audio_array, TARGET_SR)
                    save_ok = True
                except Exception as e:
                    print(f"  Warning: Could not save {filename}: {e}")

            # Only collect metadata if the preprocessed file is confirmed on disk
            if save_ok:
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