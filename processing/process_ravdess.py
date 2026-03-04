"""
process_ravdess.py

Processes the RAVDESS Audio-Speech dataset into the Common-6 emotion schema
and generates train/val/test CSV files, exactly like process_emovo.py and
process_emodb.py.

HOW TO USE:
    1. Download the dataset manually from:
       https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
    2. Unzip it. You'll get a folder with subfolders: Actor_01/, Actor_02/, ... Actor_24/
       Each subfolder contains .wav files named like: 03-01-06-01-02-01-12.wav
    3. Set SOURCE_ROOT_DIR below to the path of that folder.
    4. Run: python process_ravdess.py

RAVDESS Filename Format (7 codes separated by '-'):
    Code 1 — Modality:         01=full-AV, 02=video-only, 03=audio-only
    Code 2 — Vocal channel:    01=speech,  02=song
    Code 3 — Emotion:          01=neutral, 02=calm, 03=happy, 04=sad,
                               05=angry,   06=fearful, 07=disgust, 08=surprised
    Code 4 — Intensity:        01=normal,  02=strong
    Code 5 — Statement:        01="Kids are talking by the door"
                               02="Dogs are sitting by the door"
    Code 6 — Repetition:       01=1st, 02=2nd
    Code 7 — Actor:            01–24 (odd=male, even=female)

Example: 03-01-06-01-02-01-12.wav
    → audio-only, speech, fearful, normal, statement-2, 1st rep, Actor12 (female)

Common-6 Emotion Schema (shared across all datasets):
    0=Anger, 1=Sadness, 2=Happiness, 3=Neutral, 4=Fear, 5=Disgust
    Calm (02) and Surprised (08) are dropped (not in Common-6).

Speaker-Independent Split (24 actors):
    Train : Actor_01 – Actor_20
    Val   : Actor_21 – Actor_22
    Test  : Actor_23 – Actor_24
"""

import csv
import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# Set this to the folder that contains Actor_01/, Actor_02/, ..., Actor_24/
SOURCE_ROOT_DIR = "/dist_home/suryansh/sharukesh/speech/RAVDESS"

OUTPUT_DIR      = "/dist_home/suryansh/sharukesh/speech/metadata"
PROCESSED_AUDIO_DIR = "/dist_home/suryansh/sharukesh/speech/ravdess_wavs_16k"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

# ── EMOTION MAPPING ────────────────────────────────────────────────────────────
# RAVDESS code → Common-6 ID
# Common-6: 0=Anger, 1=Sadness, 2=Happiness, 3=Neutral, 4=Fear, 5=Disgust
EMOTION_MAP = {
    1: 3,   # neutral   → Neutral
    2: -1,  # calm      → DROP (not in Common-6)
    3: 2,   # happy     → Happiness
    4: 1,   # sad       → Sadness
    5: 0,   # angry     → Anger
    6: 4,   # fearful   → Fear
    7: 5,   # disgust   → Disgust
    8: -1,  # surprised → DROP (not in Common-6)
}

# ── STATEMENT (Text) MAPPING ───────────────────────────────────────────────────
STATEMENT_MAP = {
    1: "Kids are talking by the door.",
    2: "Dogs are sitting by the door.",
}

# ── SPEAKER SPLIT ──────────────────────────────────────────────────────────────
# 24 actors; split by actor ID for speaker independence
TRAIN_ACTORS = set(range(1, 21))   # Actor_01 – Actor_20
VAL_ACTORS   = {21, 22}            # Actor_21 – Actor_22
TEST_ACTORS  = {23, 24}            # Actor_23 – Actor_24

# ── PREPROCESSING CONSTANTS ────────────────────────────────────────────────────
TARGET_SR  = 16000
TARGET_RMS = 0.1   # Normalize all clips to this RMS loudness level


def process_audio(src_path, dest_path):
    """
    Full audio preprocessing pipeline — same as EMOVO and EmoDB:

    Step 1 — Resample to 16 kHz:
        RAVDESS is originally recorded at 48 kHz (like EMOVO). librosa.load
        with sr=16000 performs high-quality sinc resampling down to 16 kHz.
        emotion2vec_plus_large is trained at 16 kHz; this ensures consistent
        acoustic feature extraction.

    Step 2 — Convert to Mono:
        mono=True averages any stereo channels. RAVDESS files are mono, but
        this guard prevents errors on any edge-case stereo files.

    Step 3 — Trim Silence:
        librosa.effects.trim(top_db=20) removes leading/trailing frames that
        are >20 dB quieter than the loudest frame. RAVDESS recordings can have
        short silent pads at edges from the recording session.

    Step 4 — RMS Amplitude Normalization:
        RMS = sqrt(mean(y^2)) measures average signal power. We scale the
        waveform so its RMS = TARGET_RMS (0.1), bringing RAVDESS (studio-acted)
        to the same loudness reference as IEMOCAP (naturalistic conversation).
        np.clip() prevents any clipping artifacts from peaky signals.
    """
    try:
        # Step 1 + 2: Load, resample to 16 kHz, force mono
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

        # Save processed 16 kHz WAV
        sf.write(dest_path, y, TARGET_SR)
        return True
    except Exception as e:
        print(f"  Error processing {src_path}: {e}")
        return False


def main():
    root = Path(SOURCE_ROOT_DIR)
    if not root.exists():
        print(f"Error: SOURCE_ROOT_DIR '{SOURCE_ROOT_DIR}' not found.")
        print("Please download and unzip:")
        print("  https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip")
        return

    metadata_rows = []

    # ── SCAN ACTOR FOLDERS ────────────────────────────────────────────────────
    actor_dirs = sorted([d for d in root.iterdir() if d.is_dir() and "Actor" in d.name])
    print(f"Found {len(actor_dirs)} actor folders under {SOURCE_ROOT_DIR}")

    for actor_dir in actor_dirs:
        # Actor number from folder name: "Actor_01" → 1
        try:
            actor_num = int(actor_dir.name.split("_")[1])
        except (IndexError, ValueError):
            print(f"  Warning: Skipping unexpected folder '{actor_dir.name}'")
            continue

        wav_files = sorted(actor_dir.glob("*.wav"))
        print(f"  {actor_dir.name}: {len(wav_files)} files")

        for wav_file in wav_files:
            try:
                # ── PARSE FILENAME ─────────────────────────────────────────
                # Format: 03-01-06-01-02-01-12.wav
                codes = wav_file.stem.split("-")
                if len(codes) != 7:
                    print(f"    Skipping malformed filename: {wav_file.name}")
                    continue

                modality_code  = int(codes[0])
                vocal_code     = int(codes[1])
                emotion_code   = int(codes[2])
                # intensity_code = int(codes[3])  # Not used for emotion mapping
                statement_code = int(codes[4])
                actor_code     = int(codes[6])

                # ── FILTER: Audio-only speech files only ───────────────────
                # Modality 03 = audio-only, Vocal channel 01 = speech
                # We skip video-only (02) and song (02) files
                if modality_code != 3 or vocal_code != 1:
                    continue

                # ── MAP EMOTION → Common-6 ─────────────────────────────────
                if emotion_code not in EMOTION_MAP:
                    continue
                label_id = EMOTION_MAP[emotion_code]
                if label_id == -1:
                    continue  # Drop 'calm' and 'surprised'

                # ── GET TEXT ───────────────────────────────────────────────
                raw_text = STATEMENT_MAP.get(statement_code)
                if not raw_text:
                    print(f"    Warning: Unknown statement code {statement_code} in {wav_file.name}")
                    continue

                # ── PROCESS AUDIO ──────────────────────────────────────────
                dest_filename = f"RAVDESS_Actor{actor_code:02d}_{wav_file.name}"
                dest_path = os.path.join(PROCESSED_AUDIO_DIR, dest_filename)

                if not os.path.exists(dest_path):
                    success = process_audio(str(wav_file), dest_path)
                    if not success:
                        continue

                # ── COLLECT METADATA ───────────────────────────────────────
                gender = "female" if actor_code % 2 == 0 else "male"
                metadata_rows.append([
                    os.path.abspath(dest_path),  # audio_file
                    raw_text,                    # raw_text
                    label_id,                    # label (0-5)
                    f"Actor_{actor_code:02d}",   # speaker_id
                    gender,                      # gender (extra info)
                ])

            except Exception as e:
                print(f"    Error parsing {wav_file.name}: {e}")
                continue

    print(f"\nTotal valid samples collected: {len(metadata_rows)}")

    # ── SPLIT BY ACTOR (SPEAKER-INDEPENDENT) ──────────────────────────────────
    # We extract the actor number from column[3] ("Actor_XX")
    splits = {
        "train": [r for r in metadata_rows if int(r[3].split("_")[1]) in TRAIN_ACTORS],
        "val":   [r for r in metadata_rows if int(r[3].split("_")[1]) in VAL_ACTORS],
        "test":  [r for r in metadata_rows if int(r[3].split("_")[1]) in TEST_ACTORS],
    }

    # ── WRITE CSVs ─────────────────────────────────────────────────────────────
    # Using 4-column format matching IEMOCAP/EmoDB/EMOVO CSVs:
    #   audio_file | raw_text | label | speaker_id
    # (gender is dropped from CSV to stay consistent with other datasets)
    header = ["audio_file", "raw_text", "label", "speaker_id"]

    for name, rows in splits.items():
        path = os.path.join(OUTPUT_DIR, f"RAVDESS_Common6_{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            # Write only the 4 columns (drop gender at index 4)
            writer.writerows([[r[0], r[1], r[2], r[3]] for r in rows])
        print(f"Split '{name}' saved → {path}  ({len(rows)} samples)")

    # ── SUMMARY ────────────────────────────────────────────────────────────────
    print("\n── Emotion distribution ──")
    label_names = {0: "Anger", 1: "Sadness", 2: "Happiness", 3: "Neutral", 4: "Fear", 5: "Disgust"}
    from collections import Counter
    counts = Counter(r[2] for r in metadata_rows)
    for lid, name in label_names.items():
        print(f"  {name:12s}: {counts.get(lid, 0)}")


if __name__ == "__main__":
    main()
