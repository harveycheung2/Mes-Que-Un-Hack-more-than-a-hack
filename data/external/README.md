# External Open Training Data Corpus

This directory contains the **100% external open datasets** used to train the machine learning models of the **Hierarchical Episode-First Possession Engine**.

In strict accordance with **Rule 1** (Open / Inspectable Assets) and **Rule 2** (Zero Competition Target Ground Truth fitting), **zero target match frames (`BAR-ATH`)** were used for training or parameter selection.

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
- **Primary Task:** Imposing impenetrable dead-ball stoppage walls (`D: 73,839` frames, `0.9812` Alive/Dead F1) to eliminate set-piece boundary bleeding.
- **Training Corpus:**
  - **Metrica Sports Open Data:** Games 1, 2, and 3 with analyst event-derived ground truth.
  - **DFL Bundesliga IDSSE Dataset:** 7 matches with control-event filtered possession states.
  - **SkillCorner Open Data:** Broadcast tracking with official phases of play and one-touch turnover filtering.
- **Training Script:** [`starter/train_sequence_cnn.py`](../../starter/train_sequence_cnn.py)

### 2. Spatial Team LightGBM Model (`models/spatial_model/`)
- **Architecture:** Gradient Boosted Decision Trees trained over 78 permutation-invariant geometric and kinematic features.
- **Primary Task:** Inferring team ownership ($H$ vs $A$) inside disjoint in-play episodes.
- **Training Corpus:**
  - **DFL Bundesliga IDSSE Dataset:** 7 full Bundesliga matches (`J03WMX` through `J03WR9`) evaluated under 7-fold Leave-One-Match-Out (LOMO) cross-validation.
- **Training Script:** [`starter/train_spatial_model.py`](../../starter/train_spatial_model.py)

---

## 📜 Dataset Citations & Licenses

1. **DFL Bundesliga IDSSE Dataset (CC BY 4.0):**
   - *Citation:* Bassek, M., Rein, R., Weber, M. et al., *"An integrated dataset of spatiotemporal and event data in elite soccer"*, Scientific Data 12, 195 (2025). [https://doi.org/10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y).
   - *Source Repository:* [huggingface.co/datasets/pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data)

2. **Metrica Sports Sample Tracking & Event Data:**
   - *License:* Open Educational & Research Use.
   - *Source Repository:* [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)

3. **SkillCorner Open Data (MIT License):**
   - *Citation:* SkillCorner Open Data Collection, 2024/2025 A-League tracking dataset.
   - *Source Repository:* [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata)
