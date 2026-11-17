# External Open Training Data Corpus

This directory contains the external open datasets used to train the machine learning models of the **Hierarchical Episode-First Possession Engine**.

The training and parameter selection were conducted using open external tracking and event sources without using target match frames (`BAR-ATH`).

---

## 📁 Directory Structure & Included Datasets

```
data/external/
├── README.md                           # This documentation file
├── idsse_7matches_features.csv         # Tabular 78-feature extract for 7 IDSSE Bundesliga matches (18.8 MB)
├── metrica_3games_features.csv         # Tabular feature & label extract for Metrica Games 1–3 (15.8 MB)
├── skillcorner_4matches_features.csv   # Tabular feature extract for SkillCorner tracking matches (49.3 MB)
├── idsse/
│   └── cache/                          # Pre-processed full tracking & label arrays for 7 IDSSE matches (CC BY 4.0)
│       ├── J03WMX.npz (28.3 MB)
│       ├── J03WN1.npz (24.9 MB)
│       ├── J03WOH.npz (26.5 MB)
│       ├── J03WOY.npz (27.7 MB)
│       ├── J03WPY.npz (27.9 MB)
│       ├── J03WQQ.npz (27.2 MB)
│       └── J03WR9.npz (28.2 MB)
├── metrica/                            # Official event streams for Metrica Games 1, 2, and 3
│   ├── Sample_Game_1_events.csv
│   ├── Sample_Game_2_events.csv
│   └── Sample_Game_3_events.json
└── skillcorner/                        # Phases of play & match metadata for 20 A-League matches (MIT)
    ├── *_phases_of_play.csv (20 files, 2.4 MB total)
    └── *_match_data.json
```

---

## 🏆 Model Training Provenance & Dataset Mapping

### 1. Sequence CNN Dead-Ball Ensemble (`models/seq_cnn/`)
- **Architecture:** 8-Network Temporal Sequence CNN (~220k weights, $\pm 12.6$s receptive field, exact team-swap symmetry).
- **Primary Task:** Imposing dead-ball stoppage walls (`D: 73,839` frames) to separate set-piece intervals from open play.
- **Training Corpus:**
  - **Metrica Sports Open Data:** Games 1, 2, and 3 with analyst event-derived ground truth.
  - **DFL Bundesliga IDSSE Dataset:** 7 matches with control-event filtered possession states.
  - **SkillCorner Open Data:** Broadcast tracking with phases of play and one-touch turnover filtering.
- **Training Script:** [`starter/train_sequence_cnn.py`](../../starter/train_sequence_cnn.py)

### 2. Spatial Team LightGBM Model (`models/spatial_model/`)
- **Architecture:** Gradient Boosted Decision Trees trained over 78 permutation-invariant geometric and kinematic features.
- **Primary Task:** Inferring team ownership ($H$ vs $A$) inside disjoint in-play episodes.
- **Training Corpus:**
  - **DFL Bundesliga IDSSE Dataset:** 7 full Bundesliga matches (`J03WMX` through `J03WR9`) evaluated under 7-fold Leave-One-Match-Out (LOMO) cross-validation.
- **Training Script:** [`starter/train_spatial_model.py`](../../starter/train_spatial_model.py)

---

## 📜 Dataset Citations, Attributions & Licenses

### 1. DFL Bundesliga IDSSE Dataset (CC BY 4.0)
- **Attribution & Citation:**
  > Bassek, M., Rein, R., Weber, M., & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data, 12, 195. https://doi.org/10.1038/s41597-025-04505-y
  > Dataset repository: Bassek et al., figshare doi: 10.6084/m9.figshare.28196177.
  > Distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
- **Source:** [huggingface.co/datasets/pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data)

### 2. Metrica Sports Sample Tracking & Event Data
- **Source:** [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)
- **Description:** Public sample tracking and event dataset published by Metrica Sports for development and evaluation. Note: The source repository contains no formal license file; files are used in accordance with their public sample release.

### 3. SkillCorner Open Tracking Data (MIT License)
- **Source:** [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata)
- **License Text:**
  ```text
  MIT License

  Copyright (c) 2024 SkillCorner

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```
