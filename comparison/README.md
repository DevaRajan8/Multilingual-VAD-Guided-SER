# Per-Dataset Comparison Experiments
## `comparison/` folder

This folder contains scripts to train **one model per dataset** using the
**same ACL2026Model architecture** as the joint multilingual model — mirroring
the evaluation protocol of TIM-Net and EmoFusioNet for a fair paper comparison.

---

## Files

| File | Purpose |
|------|---------|
| `train_per_dataset.py` | Trains 5 independent models (one per dataset), 5 runs each, saves checkpoints under `per_dataset_models/` |
| `evaluate_per_dataset.py` | Loads best checkpoints, generates metrics + confusion matrices + paper comparison table |

---

## Run Order

> **All commands must be run from the project root**, not from inside `comparison/`.

### Step 1 — Train all 5 per-dataset models
```bash
# Train all datasets sequentially (recommended on server with GPU)
python comparison/train_per_dataset.py --dataset ALL --epochs 100 --num_runs 5

# Or train one at a time
python comparison/train_per_dataset.py --dataset IEMOCAP
python comparison/train_per_dataset.py --dataset EmoDB
python comparison/train_per_dataset.py --dataset EMOVO
python comparison/train_per_dataset.py --dataset SUBESCO
python comparison/train_per_dataset.py --dataset RAVDESS
```

If your feature files are in a different location, override the base path:
```bash
python comparison/train_per_dataset.py --dataset ALL \
    --base_path /your/path/to/features_common_6
```

### Step 2 — Evaluate and generate comparison report
```bash
# Basic evaluation (per-dataset only)
python comparison/evaluate_per_dataset.py

# With side-by-side comparison against your joint multilingual model
python comparison/evaluate_per_dataset.py \
    --joint_results results_multilingual/all_metrics.json
```

---

## Outputs

```
comparison/
├── per_dataset_models/
│   ├── IEMOCAP_run0.pth ... IEMOCAP_run4.pth
│   ├── EmoDB_run0.pth   ... EmoDB_run4.pth
│   ├── EMOVO_run0.pth   ... EMOVO_run4.pth
│   ├── SUBESCO_run0.pth ... SUBESCO_run4.pth
│   └── RAVDESS_run0.pth ... RAVDESS_run4.pth
│
└── per_dataset_results/
    ├── IEMOCAP_results.json       ← val mean±std + test metrics per dataset
    ├── EmoDB_results.json
    ├── EMOVO_results.json
    ├── SUBESCO_results.json
    ├── RAVDESS_results.json
    ├── all_per_dataset_results.json  ← combined JSON
    ├── per_dataset_test_metrics.json ← evaluation output
    ├── per_dataset_report.md         ← paper-ready markdown table
    └── confusion_matrices/
        ├── CM_IEMOCAP_PerDataset.png
        ├── CM_EmoDB_PerDataset.png
        ├── CM_EMOVO_PerDataset.png
        ├── CM_SUBESCO_PerDataset.png
        └── CM_RAVDESS_PerDataset.png
```

---

## What These Results Mean for Your Paper

| Training | Expected WA | Purpose |
|----------|-------------|---------|
| **Per-dataset** (this folder) | Higher (dataset-specific upper bound) | Matches TIM-Net / EmoFusioNet protocol |
| **Joint multilingual** (`main.py`) | Lower (shared across 5 languages) | Your novel contribution |

The key paper argument:
> *"Our joint multilingual model achieves [X]% on EmoDB while simultaneously
> handling 4 other languages — competitive with per-dataset specialized models
> that use identical architecture but train exclusively on that dataset."*

---

## Notes

- The **same `ACL2026Model` architecture** is used — no changes.
- The **same hyperparameters** (lr, batch_size, epochs, loss weights) are used.
- Per-dataset models use **5 runs** with mean ± std reporting, matching EmoFusioNet's protocol.
- Results are saved as both JSON (structured) and Markdown (paper-ready table).
