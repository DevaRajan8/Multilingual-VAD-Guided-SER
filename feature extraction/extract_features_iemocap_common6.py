"""
Extract Multilingual BERT + emotion2vec features for IEMOCAP (Common-6)
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

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
TEXT_MAX_LENGTH = 100

# --- UPDATED PATHS (Using the new Common-6 CSVs) ---
IEMOCAP_TRAIN_PATH = "metadata_common_6/IEMOCAP_Common6_train.csv"
IEMOCAP_VAL_PATH =   "metadata_common_6/IEMOCAP_Common6_val.csv"
IEMOCAP_TEST_PATH =  "metadata_common_6/IEMOCAP_Common6_test.csv"

# Output folder for the final pickle files
OUTPUT_PATH = "features_common_6/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- 1. Load Multilingual BERT ---
print("Loading Multilingual BERT (cased)...")
# CRITICAL: Using Multilingual BERT so English (IEMOCAP) and German (EmoDB)
# share the same embedding space.
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
text_model = BertModel.from_pretrained('bert-base-multilingual-cased', use_safetensors=True).to(device)

# --- 2. Load emotion2vec ---
print("Loading emotion2vec model...")
emotion2vec_model = AutoModel(model="iic/emotion2vec_plus_large", model_revision="v2.0.5")

print("Models loaded successfully")


def extract_emotion2vec_features(audio_path):
    """Extract emotion2vec features from audio file"""
    try:
        # result is usually a list of dicts
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
    """Processes a dataset and saves the features as a pickle file."""
    if not os.path.exists(dataset_path):
        print(f"Skipping {dataset_path} (File not found)")
        return

    dataset = pd.read_csv(dataset_path)
    processed_data = []
    failed = 0

    print(f"Processing {len(dataset)} samples from {dataset_path}...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(dataset)}")

            try:
                # --- Text Processing (Multilingual BERT) ---
                text = dataset['raw_text'][idx]
                # Added padding='max_length' for consistency across batches
                text_token = tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=TEXT_MAX_LENGTH,
                    padding="max_length" 
                )
                text_token = text_token.to(device)
                
                # Get CLS token embedding
                text_outputs = text_model(**text_token)
                text_embed = text_outputs.last_hidden_state[:, 0, :][0].cpu()

                # --- Audio Processing (emotion2vec) ---
                audio_file = dataset['audio_file'][idx]
                audio_embed = extract_emotion2vec_features(audio_file)

                if audio_embed is None:
                    print(f"  Failed to extract audio for {audio_file}")
                    failed += 1
                    continue

                # --- Label Processing ---
                # The label is already the correct integer (0-5) from the CSV
                label = dataset['label'][idx]
                label = torch.tensor(int(label))
                
                # Session/Speaker info (useful for analysis later)
                session = dataset['session'][idx]

                processed_data.append({
                    'text_embed': text_embed,
                    'audio_embed': audio_embed,
                    'label': label,
                    'session': session
                })
            except Exception as e:
                print(f"  Error at idx {idx}: {e}")
                failed += 1

    print(f"  Processed: {len(processed_data)}, Failed: {failed}")

    # Save to pickle
    with open(output_file, "wb") as file:
        pickle.dump(processed_data, file)
    print(f"Processed data saved to {output_file}\n")


def main():
    """Main function to process IEMOCAP datasets."""
    
    # Process training set
    process_dataset(
        IEMOCAP_TRAIN_PATH,
        f"{OUTPUT_PATH}IEMOCAP_Common6_train.pkl"
    )

    # Process validation set
    process_dataset(
        IEMOCAP_VAL_PATH,
        f"{OUTPUT_PATH}IEMOCAP_Common6_val.pkl"
    )

    # Process test set
    process_dataset(
        IEMOCAP_TEST_PATH,
        f"{OUTPUT_PATH}IEMOCAP_Common6_test.pkl"
    )


if __name__ == "__main__":
    main()