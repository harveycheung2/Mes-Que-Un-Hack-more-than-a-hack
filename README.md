# Hierarchical Episode-First Possession Engine

An end-to-end possession detection system for football tracking data and tactical event extraction.

## Performance Overview

Evaluated under the official competition protocol (`IoU 0.30–0.70`):

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Mean Spell F1** | **`0.6667`** | Primary competition ranking metric (IoU 0.3–0.7) |
| **Spell F1 @ 0.5 IoU** | **`0.6667`** | Precision: **90.0%** (9 of 10 predicted spells matched) |
| **Macro F1** | **`0.9140`** | Home: 0.9352, Away: 0.8906, Dead: 0.9163 |
| **Alive / Dead F1** | **`0.9812`** | High-precision set-piece boundary detection |
| **Frame Accuracy** | **`91.55%`** | Graded play accuracy |
| **Rule Compliance** | **`100%`** | Zero training or tuning on target match ground truth |

## Core Methodology

1. **Episode-First Structural Wall Locking:** Dead-ball stoppage intervals (fouls, out-of-bounds, set-pieces) are identified and locked as structural boundaries first.
2. **Within-Episode Smoothing:** Possession classification and biomechanical dwell filtering operate strictly within bounded in-play episodes, ensuring zero boundary bleeding across set pieces.
3. **Biomechanical Dwell Filter ($\ge 1.0$s / 25 frames):** Transient contact and coordinate noise (<25 frames) are absorbed into adjacent active possessions within the episode, eliminating sensor flutter without compromising genuine quick turnover sequences.

For complete architectural details, see [`docs/TECHNICAL_RATIONALE.md`](docs/TECHNICAL_RATIONALE.md).
For team engineering insights and modeling roadmap, see [`docs/MODEL_IMPROVEMENT_GUIDE.md`](docs/MODEL_IMPROVEMENT_GUIDE.md).

## Quick Start

### 1. Run Unit Tests
```bash
python tests/test_hierarchical_engine.py
```

### 2. Run Official Evaluation
```bash
python evaluation/run_evaluation.py --results evaluation/results
```

### 3. Run Production Engine
```bash
python starter/hierarchical_engine.py
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
    └── test_hierarchical_engine.py  # Unit & integration tests
```
