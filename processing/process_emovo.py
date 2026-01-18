import csv
import os
import soundfile as sf
import librosa

# 1. Configuration
# REPLACE THIS with the path to your unzipped 'EMOVO' folder 
# (The one containing folders f1, f2, documents, etc.)
SOURCE_ROOT_DIR = "./EMOVO" 

OUTPUT_DIR = "metadata_common_6"
PROCESSED_AUDIO_DIR = "emovo_wavs_16k"
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

def process_audio(src_path, dest_path, target_sr=16000):
    try:
        # Load with librosa (handles resampling automatically)
        y, sr = librosa.load(src_path, sr=target_sr)
        # Save as 16kHz wav
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