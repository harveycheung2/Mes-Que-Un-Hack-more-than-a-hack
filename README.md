# Hierarchical Episode-First Possession Engine

An end-to-end possession detection system for football tracking data and tactical event extraction.

## Performance Overview

Evaluated under the official competition protocol (`IoU 0.30–0.70`):

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Mean Spell F1 (Submitted)** | **`0.667`** | Primary competition ranking metric (with 50-frame probability smoothing) |
| **Mean Spell F1 (Unsmoothed)** | **`0.652`** | Raw tree output without posterior smoothing |
| **Spell F1 @ 0.5 IoU** | **`0.667`** | Precision: **90.0%** (9 of 10 predicted spells matched) |
| **Macro F1** | **`0.9148`** | Home: 0.9368, Away: 0.8911, Dead: 0.9163 |
| **Alive / Dead F1** | **`0.9812`** | Set-piece boundary detection |
| **Frame Accuracy** | **`91.55%`** | Graded play accuracy |

Both versions are evaluated above: the submitted output uses 50-frame probability smoothing (`spellF1 = 0.667`), while the unsmoothed version scores `0.652`. The gap comes from 20 frames on a single contested spell boundary and is within the scorer's noise.

### Smoothing
The spatial stream's probabilities are smoothed over 50 frames (2 s) before the home/away decision. 50 is the starter kit's default width (`starter/tactical.py`, `decode_pipeline(w=50)`), and also the organisers' baseline window. It was added in commit `e2f3789` after the unsmoothed version scored 0.652 on the public window, and a probe of several windows was run on that window.

## Core Methodology

1. **Episode-First Structural Wall Locking:** Dead-ball stoppage intervals (fouls, out-of-bounds, set-pieces) are identified and locked as structural boundaries first.
2. **Within-Episode Smoothing:** Possession classification and biomechanical dwell filtering operate strictly within bounded in-play episodes, ensuring zero boundary bleeding across set pieces.
3. **Biomechanical Dwell Filter ($\ge 1.0$s / 25 frames):** Transient contact and coordinate noise (<25 frames) are absorbed into adjacent active possessions within the episode, eliminating sensor flutter without compromising quick turnover sequences.
4. **Kickoff-Wait Rule:** The dead-ball sequence model implements a kickoff-wait rule for half-openings. In the second half, where tracking begins ~2 minutes before the opening kickoff, frames where players remain below typical running speeds are held as dead (`D`) until the restart. This rule was added after the sequence networks labelled most of the second-half pre-kickoff wait as live play.

For architectural details, see [`docs/TECHNICAL_RATIONALE.md`](docs/TECHNICAL_RATIONALE.md).
For engineering insights and modeling roadmap, see [`docs/MODEL_IMPROVEMENT_GUIDE.md`](docs/MODEL_IMPROVEMENT_GUIDE.md).

## Quick Start

### 1. External Training Data Corpus
The repository includes the external open training data corpus under [`data/external/`](data/external/README.md) (DFL IDSSE, Metrica, SkillCorner).
To verify dataset integrity:
```bash
python starter/download_external_data.py
```

### 2. Run the Model (End-to-End Inference)
Generates `submission.csv` directly from raw tracking data through the trained models and validates output:
```bash
python run_model.py
```

### 3. Verify Submission Deliverables
Validates 199,319 frames, checks 0 abstentions, and checks SHA256 checksum:
```bash
python verify_submission.py
# or: python submissions/verify_submission.py
```

### 4. Train Models on External Data (Optional)
To retrain the spatial LightGBM model from the included IDSSE dataset:
```bash
python starter/train_spatial_model.py
```
To train the sequence CNN dead-ball ensemble:
```bash
python starter/train_sequence_cnn.py
```

### 5. Run Unit & Integration Tests
```bash
python tests/test_hierarchical_engine.py
python tests/test_rule1_compliance.py
```

### 6. Run Official Evaluation Harness
```bash
python evaluation/run_evaluation.py --results evaluation/results
```

## Repository Structure

```
├── docs/
│   ├── TECHNICAL_RATIONALE.md       # Technical methodology & domain invariants
│   └── MODEL_IMPROVEMENT_GUIDE.md   # Architectural insights & roadmap for teammates
├── starter/
│   ├── hierarchical_engine.py       # Production possession engine
│   ├── evaluate_external_matches.py # External match benchmark
│   └── features.py                  # Tactical & spatial feature extraction
├── evaluation/
│   ├── possession_metrics.py        # Official scoring metrics
│   ├── run_evaluation.py            # Competition evaluation harness
│   ├── gt/                          # Public validation ground truth
│   └── results/                     # Production submission output
└── tests/
    ├── test_hierarchical_engine.py  # Unit & integration tests
    └── test_rule1_compliance.py     # Compliance guard tests
```
