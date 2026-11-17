"""Train Spatial Team LightGBM Model on External IDSSE Bundesliga Tracking Data.

Strict Compliance with Nov 3 Rule Mandate (100% External Data, 0 Competition Match Frames).

Trains LightGBM classifier on all 7 IDSSE matches (CC BY 4.0):
  - Advanced multi-scale, boundary, and collective motion features (78 features)
  - Evaluated via 7-fold Leave-One-Match-Out (LOMO) cross-validation
  - Saves trained booster to `models/spatial_model/spatial_lgb_model.txt`
"""

import os
import sys
import time
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import external_pipeline as ep

MODEL_DIR = os.path.join(ROOT, "models", "spatial_model")


def main():
    print("=" * 80)
    print(" TRAINING SPATIAL TEAM POSSESSION MODEL (100% EXTERNAL DATA)")
    print(" Corpus: 7 IDSSE Bundesliga Matches (CC BY 4.0) | 78 Permutation-Invariant Features")
    print("=" * 80)
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[1/2] Loading IDSSE feature sets...")
    sets = ep.load_idsse_sets(ep.ball_unseen_pattern(), advanced=True, boundary=True, motion=True)
    print(f"  Loaded in {time.time()-t0:.2f}s across {len(sets)} matches.")

    print("\n[2/2] Fitting LightGBM spatial possession classifier...")
    t1 = time.time()
    model = ep.fit("lightgbm", sets)
    print(f"  Fitted LightGBM in {time.time()-t1:.2f}s.")

    out_path = os.path.join(MODEL_DIR, "spatial_lgb_model.txt")
    model.booster_.save_model(out_path)
    print(f"\nModel saved to: {out_path} ({os.path.getsize(out_path):,} bytes)")
    print("=" * 80)
    print(f" TRAINING COMPLETE IN {time.time()-t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
