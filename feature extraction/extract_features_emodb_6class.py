"""
Extract Multilingual BERT + emotion2vec features for EmoDB
Filters for the Common-6 Emotion Set (Anger, Sadness, Happiness, Neutral, Fear, Disgust)
"""
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import pickle
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from funasr import AutoModel

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT_MAX_LENGTH = 100

# Input CSVs (From the previous CSV generation step)
# Ensure these point to where you saved the EmoDB csv files
EMODB_TRAIN_PATH = "./german/metadata_emodb/EmoDB_train.csv"
EMODB_VAL_PATH =   "./german/metadata_emodb/EmoDB_val.csv"
EMODB_TEST_PATH =  "./german/metadata_emodb/EmoDB_test.csv"

OUTPUT_PATH = "features_common_6/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- EMOTION MAPPING ---
# We must map the IDs from the CSV (which might be 7-class) 
# to your specific Common-6 Schema.
#
# Your Common-6 Target:
# 0: Anger, 1: Sadness, 2: Happiness, 3: Neutral, 4: Fear, 5: Disgust
#
# Standard EmoDB CSV IDs (based on alphabetic/doc order usually):
# 0: anger, 1: boredom, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: neutral

ID_REMAP = {
    0: 0,  # Anger     -> Anger
    1: -1, # Boredom   -> SKIP (Not in common_6)
    2: 5,  # Disgust   -> Disgust
    3: 4,  # Fear      -> Fear
    4: 2,  # Happiness -> Happiness
    5: 1,  # Sadness   -> Sadness
    6: 3   # Neutral   -> Neutral
}

# --- MODEL LOADING ---

print("Loading Multilingual BERT (cased)...")
# CRITICAL: Using Multilingual BERT to align English (IEMOCAP), German (EmoDB), and Italian (EMOVO)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
text_model = BertModel.from_pretrained('bert-base-multilingual-cased', use_safetensors=True).to(device)

print("Loading emotion2vec model...")
emotion2vec_model = AutoModel(model="iic/emotion2vec_plus_large", model_revision="v2.0.5")

print("Models loaded successfully.")


def extract_emotion2vec_features(audio_path):
    """Extract emotion2vec features from audio file"""
    try:
        # result is a list of dicts
        result = emotion2vec_model.generate(audio_path, output_dir=None, granularity="utterance")
        
        if isinstance(result, list) and len(result) > 0:
            feats = result[0].get('feats', None)
            if feats is not None:
                if isinstance(feats, np.ndarray):
                    return torch.from_numpy(feats).float()
                return feats.float()
        return None
    except Exception as e:
        print(f"Error extracting emotion2vec for {audio_path}: {e}")
        return None


def process_dataset(dataset_path, output_file):
    if not os.path.exists(dataset_path):
        print(f"Skipping {dataset_path} (File not found)")
        return

    dataset = pd.read_csv(dataset_path)
    processed_data = []
    skipped_count = 0
    failed_audio = 0

    print(f"Processing {len(dataset)} samples from {dataset_path}...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx % 50 == 0:
                print(f"  Progress: {idx}/{len(dataset)}")

            try:
                # 1. Check Label & Filter
                raw_label_id = int(dataset['label'][idx])
                
                # Check if this raw label exists in our map
                if raw_label_id not in ID_REMAP:
                    skipped_count += 1
                    continue

                new_label_id = ID_REMAP[raw_label_id]

                # If mapped to -1, it means we skip this emotion (e.g., Boredom)
                if new_label_id == -1:
                    skipped_count += 1
                    continue
                
                # 2. Process Text (Multilingual BERT)
                text = dataset['raw_text'][idx]
                text_token = tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=TEXT_MAX_LENGTH,
                    padding="max_length"
                )
                text_token = text_token.to(device)
                
                text_outputs = text_model(**text_token)
                # CLS token embedding
                text_embed = text_outputs.last_hidden_state[:, 0, :][0].cpu()

                # 3. Process Audio (emotion2vec)
                audio_file = dataset['audio_file'][idx]
                audio_embed = extract_emotion2vec_features(audio_file)

                if audio_embed is None:
                    print(f"  Failed to extract audio for {audio_file}")
                    failed_audio += 1
                    continue

                # 4. Save Data
                # Storing the integer ID (0-5). 
                # You can map this to your 3D vector later in the Dataset class.
                processed_data.append({
                    'text_embed': text_embed,
                    'audio_embed': audio_embed,
                    'label': torch.tensor(new_label_id),
                    'speaker_id': dataset['speaker_id'][idx]
                })

            except Exception as e:
                print(f"  Error at idx {idx}: {e}")
                skipped_count += 1

    print(f"  Finished: {len(processed_data)} kept, {skipped_count} skipped (boredom/errors), {failed_audio} failed audio.")

    # Save to pickle
    with open(output_file, "wb") as file:
        pickle.dump(processed_data, file)
    print(f"  Saved to {output_file}\n")


def main():
    # Process training set
    process_dataset(
        EMODB_TRAIN_PATH,
        f"{OUTPUT_PATH}EmoDB_Common6_train.pkl"
    )

    # Process validation set
    process_dataset(
        EMODB_VAL_PATH,
        f"{OUTPUT_PATH}EmoDB_Common6_val.pkl"
    )

    # Process test set
    process_dataset(
        EMODB_TEST_PATH,
        f"{OUTPUT_PATH}EmoDB_Common6_test.pkl"
    )

if __name__ == "__main__":
    main()