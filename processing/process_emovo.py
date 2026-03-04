import csv
import os
import numpy as np
import soundfile as sf
import librosa

# 1. Configuration
# REPLACE THIS with the path to your unzipped 'EMOVO' folder 
# (The one containing folders f1, f2, documents, etc.)
SOURCE_ROOT_DIR = "/dist_home/suryansh/sharukesh/speech/EMOVO" 

OUTPUT_DIR = "/dist_home/suryansh/sharukesh/speech/metadata"
PROCESSED_AUDIO_DIR = "/dist_home/suryansh/sharukesh/speech/emovo_wavs_16k"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

# 2. Mappings

# Emotion -> Common-6 ID
# Target: 0:Anger, 1:Sadness, 2:Happiness, 3:Neutral, 4:Fear, 5:Disgust
EMOTION_MAP = {
    "rab": 0,  # rabbia -> Anger
    "tri": 1,  # triste -> Sadness
    "gio": 2,  # gioia -> Happiness
    "neu": 3,  # neutro -> Neutral
    "pau": 4,  # paura -> Fear
    "dis": 5,  # disgusto -> Disgust
    "sor": -1  # sorpresa -> DROP (Not in Common-6)
}

# Sentence Code -> Italian Text
SENTENCE_MAP = {
    # Short (b)
    "b1": "Gli operai si alzano presto.",
    "b2": "I vigili sono muniti di pistola.",
    "b3": "La cascata fa molto rumore.",
    # Long (l)
    "l1": "L’autunno prossimo Tony partirà per la Spagna nella prima metà di ottobre.",
    "l2": "Ora prendo la felpa di là ed esco per fare una passeggiata.",
    "l3": "Un attimo dopo s’è incamminato ... ed è inciampato.",
    "l4": "Vorrei il numero telefonico del Signor Piatti.",
    # Nonsense (n)
    "n1": "La casa forte vuole col pane.",
    "n2": "La forza trova il passo e l’aglio rosso.",
    "n3": "Il gatto sta scorrendo nella pera.",
    "n4": "Insalata pastasciutta coscia d’agnello limoncello.",
    "n5": "Uno quarantatré dieci mille cinquantasette venti.",
    # Questions (d)
    "d1": "Sabato sera cosa farà?",
    "d2": "Porti con te quella cosa?"
}

TARGET_RMS = 0.1  # Normalize all clips to this RMS loudness level

def process_audio(src_path, dest_path, target_sr=16000):
    """
    Loads audio from src_path and applies the full preprocessing pipeline:

    Step 1 - Resample to 16 kHz:
        librosa.load(..., sr=16000) reads the audio at any original sample rate
        and resamples it down (or up) to exactly 16000 Hz using a high-quality
        sinc resampler. This ensures emotion2vec always receives audio at the
        sample rate it was trained on, regardless of whether the source was
        EMOVO (48 kHz), EmoDB (16 kHz), or any other rate.

    Step 2 - Convert to Mono:
        If the recording has 2 channels (stereo), librosa averages them into a
        single channel. emotion2vec expects mono audio; mixing to mono prevents
        feature extraction errors and removes any stereo-specific loudness
        differences between datasets.

    Step 3 - Trim Silence:
        librosa.effects.trim removes stretches of silence from the start and
        end of the clip. top_db=20 means any frame more than 20 dB quieter than
        the loudest frame is considered silence and gets clipped off.
        Why: EMOVO files often have noticeable leading/trailing silence from the
        recording session. Trimming ensures emotion2vec focuses attention on
        actual speech, not silence — consistent with how IEMOCAP clips are
        already tightly segmented.

    Step 4 - RMS Amplitude Normalization:
        RMS (Root Mean Square) is the average "loudness" of the signal.
        We compute it as: rms = sqrt(mean(y^2))
        Then scale the signal so its RMS equals TARGET_RMS (0.1, in [-1,1] range).
        Why: EmoDB is studio-recorded acted speech (loud and clean), IEMOCAP is
        naturalistic dyadic conversation (variable loudness), and EMOVO is Italian
        acted speech. Without normalization, the model could exploit loudness as a
        spurious cue rather than learning true emotion patterns. RMS normalization
        brings all three datasets to the same loudness ballpark.
    """
    try:
        # Step 1: Load + Resample to target_sr (handles any source SR)
        y, sr = librosa.load(src_path, sr=target_sr, mono=True)  # Step 2: mono=True here

        # Step 3: Trim leading and trailing silence
        # top_db=20: frames >20dB below peak are treated as silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # Safety check: skip if clip is too short after trimming (<0.3s)
        if len(y) < target_sr * 0.3:
            print(f"  Skipping {src_path} — too short after trimming ({len(y)/target_sr:.2f}s)")
            return False

        # Step 4: RMS Normalization
        rms = np.sqrt(np.mean(y ** 2))
        if rms > 1e-6:  # Avoid division by zero for silent clips
            y = y * (TARGET_RMS / rms)
        # Clip to [-1, 1] to prevent any float overflow artifacts
        y = np.clip(y, -1.0, 1.0)

        # Save the processed audio as a new 16kHz WAV
        sf.write(dest_path, y, target_sr)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def main():
    if not os.path.exists(SOURCE_ROOT_DIR):
        print(f"Error: Directory '{SOURCE_ROOT_DIR}' not found.")
        return

    # EMOVO actors based on your image
    actors = ["f1", "f2", "f3", "m1", "m2", "m3"]
    
    metadata_rows = []
    
    print(f"Scanning {SOURCE_ROOT_DIR}...")

    for actor in actors:
        actor_path = os.path.join(SOURCE_ROOT_DIR, actor)
        if not os.path.exists(actor_path):
            print(f"Warning: Actor folder {actor} not found in root.")
            continue
            
        files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]
        print(f"  Processing {actor}: found {len(files)} files")

        for filename in files:
            # Format usually: emo-actor-sentence.wav (e.g., rab-f1-b1.wav)
            # Some versions might use underscores or different ordering. 
            # We assume: code-actor-sentence based on standard EMOVO.
            
            try:
                name_parts = filename.replace(".wav", "").replace("_", "-").split("-")
                
                # Safety check for filename format
                if len(name_parts) != 3:
                    print(f"    Skipping malformed file: {filename}")
                    continue

                emo_code = name_parts[0]
                # actor_id = name_parts[1] # We already know actor from the folder loop
                sent_code = name_parts[2]

                # 1. Get Emotion Label
                if emo_code not in EMOTION_MAP:
                    continue # Skip unknown codes
                
                label_id = EMOTION_MAP[emo_code]
                if label_id == -1:
                    continue # Skip Surprise (not in Common-6)

                # 2. Get Raw Text
                raw_text = SENTENCE_MAP.get(sent_code, "")
                if not raw_text:
                    print(f"    Warning: No text map for code {sent_code} in {filename}")
                    continue

                # 3. Process Audio
                src_abs_path = os.path.join(actor_path, filename)
                dest_filename = f"EMOVO_{actor}_{filename}" # Unique name
                dest_abs_path = os.path.join(PROCESSED_AUDIO_DIR, dest_filename)
                
                if not os.path.exists(dest_abs_path):
                    success = process_audio(src_abs_path, dest_abs_path)
                    if not success: continue

                # 4. Add to rows
                # Using absolute path for CSV
                metadata_rows.append([os.path.abspath(dest_abs_path), raw_text, label_id, actor])

            except Exception as e:
                print(f"Error parsing {filename}: {e}")
                continue

    # --- CSV WRITING ---
    header = ["audio_file", "raw_text", "label", "speaker_id"]
    
    # Speaker Independent Split
    # Train: f1, f2, m1, m2
    # Val: f3
    # Test: m3
    
    splits = {
        "train": [r for r in metadata_rows if r[3] in ["f1", "f2", "m1", "m2"]],
        "val":   [r for r in metadata_rows if r[3] == "f3"],
        "test":  [r for r in metadata_rows if r[3] == "m3"]
    }

    for name, rows in splits.items():
        path = os.path.join(OUTPUT_DIR, f"EMOVO_Common6_{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Split '{name}' saved with {len(rows)} samples.")

if __name__ == "__main__":
    main()