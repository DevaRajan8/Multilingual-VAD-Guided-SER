"""
Generate metadata CSVs for SUBESCO (Bangla) dataset — Common-6 emotions only.
Parses filenames like: F_01_OISHI_S_10_ANGRY_1.wav
Splits by speaker (no speaker leakage across train/val/test).

Output columns: audio_file, raw_text, label, speaker_id
Label schema (common_6): 0=Anger, 1=Sadness, 2=Happiness, 3=Neutral, 4=Fear, 5=Disgust
"""

import os
import glob
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ===================== CONFIGURATION =====================

# Path to folder containing all SUBESCO .wav files (flat directory)
SUBESCO_WAV_DIR = "/dist_home/suryansh/sharukesh/speech/bangla/SUBESCO"

# Output directory for the CSVs
OUTPUT_DIR = "/dist_home/suryansh/sharukesh/speech/metadata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output directory for preprocessed 16kHz audio (mirroring EMOVO/EmoDB/RAVDESS)
PROCESSED_AUDIO_DIR = "/dist_home/suryansh/sharukesh/speech/wavs_16k/subesco_wavs_16k"
os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

# ── PREPROCESSING CONSTANTS ────────────────────────────────────────────────────
TARGET_SR  = 16000
TARGET_RMS = 0.1   # Normalize all clips to this RMS loudness level

# ===================== BANGLA SENTENCES =====================
# From the SUBESCO paper (Table 2) — the 10 sentences used in the corpus.
# mBERT supports Bangla, so these will produce meaningful embeddings.

SENTENCE_MAP = {
    1:  "মৌমাছির চাক দেখে কুকুরটি ঘেউঘেউ করছে",
    2:  "তুমি সব উলটপালট করে দিয়েছ",
    3:  "সে কোনকিছুনা বলেই চলে গেছে",
    4:  "তোমাকে এখনই আমার সাথে যেতে হবে",
    5:  "তোমার কাজটা করা ঠিক হয়নি",
    6:  "ইঁদুর ছানাটা হারিয়ে গেল",
    7:  "এখন প্রশ্ন করা যাবে না",
    8:  "দরজার বাইরে কুকুরটি দাঁড়িয়ে আছে",
    9:  "একদিন পরেই তার বিয়ে",
    10: "ডাকাতেরা ঢাল তলোয়ার নিয়ে এল",
}

# ===================== EMOTION MAPPING =====================
# Map SUBESCO filename emotion tags → common_6 label IDs
# Surprise → -1 (skip, not in common_6)

EMOTION_TO_LABEL = {
    "ANGRY":     0,   # Anger
    "ANGER":     0,   # Anger (alternate)
    "SAD":       1,   # Sadness
    "SADNESS":   1,   # Sadness (alternate)
    "HAPPY":     2,   # Happiness
    "HAPPINESS": 2,   # Happiness (alternate)
    "NEUTRAL":   3,   # Neutral
    "FEAR":      4,   # Fear
    "DISGUST":   5,   # Disgust
    "SURPRISE":  -1,  # SKIP
}

# ── AUDIO PREPROCESSING ────────────────────────────────────────────────────────

def process_audio(src_path, dest_path):
    """
    Same 4-step preprocessing pipeline used across all datasets:

    Step 1 — Resample to 16 kHz:
        SUBESCO is recorded at 44.1 kHz. librosa.load(sr=16000) resamples
        it using a sinc-based algorithm to match emotion2vec's expected SR.

    Step 2 — Convert to Mono:
        mono=True averages channels. SUBESCO files are mono, but this
        guards against any edge-case stereo recordings.

    Step 3 — Trim Silence:
        librosa.effects.trim(top_db=20) removes frames at the start and
        end that are >20 dB quieter than the peak frame. Clips any silent
        padding common in lab-recorded Bangla speech datasets.

    Step 4 — RMS Amplitude Normalization:
        Scales the signal so RMS = TARGET_RMS (0.1). This compensates for
        loudness differences between Bangla (SUBESCO), English (IEMOCAP),
        German (EmoDB), Italian (EMOVO), and English-acted (RAVDESS).
    """
    try:
        # Step 1 + 2: Load at 16 kHz, force mono
        y, sr = librosa.load(src_path, sr=TARGET_SR, mono=True)

        # Step 3: Trim leading and trailing silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # Safety check: skip clips too short after trimming (<0.3s)
        if len(y) < TARGET_SR * 0.3:
            print(f"  Skipping {src_path} — too short after trimming ({len(y)/TARGET_SR:.2f}s)")
            return False

        # Step 4: RMS Normalization
        rms = np.sqrt(np.mean(y ** 2))
        if rms > 1e-6:
            y = y * (TARGET_RMS / rms)
        y = np.clip(y, -1.0, 1.0)

        sf.write(dest_path, y, TARGET_SR)
        return True
    except Exception as e:
        print(f"  Error processing {src_path}: {e}")
        return False

# ===================== PARSE FILENAMES =====================

def parse_filename(filepath):
    """
    Parse a SUBESCO filename like: F_01_OISHI_S_10_ANGRY_1.wav
    Returns dict with keys: audio_file, speaker_id, sentence_num, emotion, raw_text, label
    or None if the emotion should be skipped.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = basename.split("_")
    
    # Format: Gender_SpeakerNum_Name_S_SentenceNum_Emotion_RepNum
    # Example: F_01_OISHI_S_10_ANGRY_1
    # parts:   [F, 01, OISHI, S, 10, ANGRY, 1]
    
    gender = parts[0]           # F or M
    speaker_num = parts[1]      # 01-10
    speaker_name = parts[2]     # Name
    # parts[3] == "S"
    sentence_num = int(parts[4])  # 1-10
    emotion = parts[5]          # ANGRY, DISGUST, etc.
    # parts[6] = repetition number
    
    speaker_id = f"{gender}_{speaker_num}"

    label = EMOTION_TO_LABEL.get(emotion, None)
    if label is None or label == -1:
        return None  # Skip Surprise or unknown emotions
    
    raw_text = SENTENCE_MAP.get(sentence_num, "")
    if not raw_text:
        print(f"Warning: No text found for sentence {sentence_num} in {filepath}")
        return None

    return {
        "audio_file": filepath,   # Will be replaced with preprocessed path in main()
        "raw_text": raw_text,
        "label": label,
        "speaker_id": speaker_id,
    }


def main():
    # 1. Gather all wav files
    wav_files = sorted(glob.glob(os.path.join(SUBESCO_WAV_DIR, "*.wav")))
    print(f"Found {len(wav_files)} wav files in {SUBESCO_WAV_DIR}")
    
    # 2. Parse all filenames
    records = []
    skipped = 0
    for f in wav_files:
        parsed = parse_filename(f)
        if parsed is None:
            skipped += 1
            continue

        # ── PREPROCESS AUDIO ──────────────────────────────────────────────────
        # Build a unique destination filename (same convention as RAVDESS/EMOVO)
        dest_filename = "SUBESCO_" + os.path.basename(f)
        dest_path = os.path.join(PROCESSED_AUDIO_DIR, dest_filename)

        if not os.path.exists(dest_path):
            success = process_audio(f, dest_path)
            if not success:
                skipped += 1
                continue

        # Store the preprocessed path (not the original raw path)
        parsed["audio_file"] = os.path.abspath(dest_path)
        records.append(parsed)
    
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} samples (skipped {skipped} Surprise samples)")
    print(f"Emotion distribution:\n{df['label'].value_counts().sort_index()}")
    print(f"Speakers: {sorted(df['speaker_id'].unique())}")
    
    # 3. Speaker-disjoint split: 70% train, 15% val, 15% test
    #    Balanced by gender: 7M+7F train, 1-2M+1-2F val, 1-2M+1-2F test
    speakers = sorted(df['speaker_id'].unique())
    male_speakers = [s for s in speakers if s.startswith('M')]
    female_speakers = [s for s in speakers if s.startswith('F')]
    
    print(f"\nMale speakers ({len(male_speakers)}):   {male_speakers}")
    print(f"Female speakers ({len(female_speakers)}): {female_speakers}")
    
    # Split: 7 train, 1 val, 2 test per gender (or adjust as needed)
    # For 10 speakers per gender → 7/1/2 split
    m_train = male_speakers[:7]
    m_val   = male_speakers[7:8]
    m_test  = male_speakers[8:]
    
    f_train = female_speakers[:7]
    f_val   = female_speakers[7:8]
    f_test  = female_speakers[8:]
    
    train_speakers = set(m_train + f_train)
    val_speakers   = set(m_val + f_val)
    test_speakers  = set(m_test + f_test)
    
    print(f"\nTrain speakers ({len(train_speakers)}): {sorted(train_speakers)}")
    print(f"Val speakers   ({len(val_speakers)}):   {sorted(val_speakers)}")
    print(f"Test speakers  ({len(test_speakers)}):  {sorted(test_speakers)}")
    
    train_df = df[df['speaker_id'].isin(train_speakers)]
    val_df   = df[df['speaker_id'].isin(val_speakers)]
    test_df  = df[df['speaker_id'].isin(test_speakers)]
    
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # 4. Save CSVs
    train_df.to_csv(os.path.join(OUTPUT_DIR, "SUBESCO_Common6_train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "SUBESCO_Common6_val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "SUBESCO_Common6_test.csv"), index=False)
    
    print(f"\nCSVs saved to {OUTPUT_DIR}")
    print("  - SUBESCO_Common6_train.csv")
    print("  - SUBESCO_Common6_val.csv")
    print("  - SUBESCO_Common6_test.csv")


if __name__ == "__main__":
    main()
