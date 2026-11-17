#!/usr/bin/env python3
"""Run Model: Hierarchical Episode-First Possession Engine.

FC Barcelona Sports Analytics Hackathon 2027
Phase 1 Deliverable: Automated Possession Prediction Pipeline

Usage:
    python run_model.py
    python run_model.py --out submission.csv
    python run_model.py --verify-only

Pipeline Architecture (Tracking -> Weights -> Predictions -> Labels):
  Stage 1: Structural Stoppage Wall Locking (8-Network Sequence CNN Ensemble)
  Stage 2: In-Play Episode Partitioning (Zero Boundary Bleeding)
  Stage 3: Spatial Classification & Biomechanical Dwell (Spatial LightGBM + Dwell=25f)
"""

import os
import sys
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from hierarchical_engine import run_pipeline
import possession_metrics as pm
from games import GAME, GAMES


def print_banner():
    print("=" * 80)
    print(" MORE THAN A HACK 2027 - HIERARCHICAL EPISODE-FIRST POSSESSION ENGINE")
    print("=" * 80)
    print(" Architecture: 3-Stage Hierarchical Decoupled Engine")
    print("   • Stage 1: Structural Stoppage Wall Enforcement (8-Network Sequence CNN)")
    print("   • Stage 2: Disjoint In-Play Episode Partitioning (Zero Boundary Bleeding)")
    print("   • Stage 3: Spatial Classification & Within-Episode Biomechanical Dwell (25f)")
    print(" Model Weights: models/seq_cnn/ (8 CNN Networks) & models/spatial_model/")
    print(" Data Source:   100% External Open Data (Metrica 1-3, DFL IDSSE, SkillCorner)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Hierarchical Episode-First Possession Model")
    parser.add_argument("--out", default=os.path.join(ROOT, "submission.csv"), help="Output CSV path")
    parser.add_argument("--data", default=None, help="Path to data directory containing match tracking")
    parser.add_argument("--models", default=None, help="Directory containing trained models")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing submission file")
    args = parser.parse_args()

    print_banner()

    if args.verify_only:
        print(f"\n[Mode: Verify Only] Checking: {args.out}")
        if not os.path.isfile(args.out):
            print(f"ERROR: File not found: {args.out}")
            sys.exit(1)
        table = pm.read_label_csv(args.out)
        val = pm.validate_submission(table, GAMES[GAME]["n_video_frames"])
        if val:
            print(f"FAILED: {val}")
            sys.exit(1)
        print("PASSED official format validation.")
        return

    labels, file_hash = run_pipeline(
        data_dir=args.data,
        models_dir=args.models,
        out_path=args.out
    )

    print("\n" + "=" * 80)
    print(" REPRODUCTION COMPLETE - Submission verified and ready for upload.")
    print("=" * 80)


if __name__ == "__main__":
    main()
