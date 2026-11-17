"""Verification script for competition submission deliverables.

Runs format validation and computes official benchmark metrics against the public ground truth slice.
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.basename(HERE) != "submissions" else os.path.dirname(HERE)
for cand_root in [ROOT, os.path.dirname(ROOT), os.path.join(ROOT, "evaluation")]:
    if os.path.isdir(cand_root) and cand_root not in sys.path:
        sys.path.insert(0, cand_root)

import possession_metrics as pm

def find_submission():
    candidates = [
        os.path.join(HERE, "submission.csv"),
        os.path.join(HERE, "submissions", "submission.csv"),
        os.path.join(ROOT, "submissions", "submission.csv"),
        os.path.join(ROOT, "submission.csv"),
        os.path.join(ROOT, "evaluation", "results", "submission.csv"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return candidates[0]

def find_gt():
    candidates = [
        os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"),
        os.path.join(HERE, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"),
        os.path.join(HERE, "..", "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return candidates[0]

SUBMISSION_FILE = find_submission()
GT_FILE = find_gt()
N_EXPECTED = 199319


def main():
    print("=" * 80)
    print(" COMPETITION SUBMISSION VERIFICATION")
    print("=" * 80)

    # 1. Format Validation
    print(f"\n[1/2] Checking submission format: {SUBMISSION_FILE}")
    if not os.path.isfile(SUBMISSION_FILE):
        print(f"ERROR: File not found: {SUBMISSION_FILE}")
        sys.exit(1)

    table = pm.read_label_csv(SUBMISSION_FILE)
    reason = pm.validate_submission(table, N_EXPECTED)
    if reason:
        print(f"FAILED: {reason}")
        sys.exit(1)

    labels = table["label"].str.strip().str.upper().values
    counts = {k: int(np.sum(labels == k)) for k in ["H", "A", "D", "X"]}
    print("PASSED official format validation.")
    print(f"  Total Rows: {len(labels):,}")
    print(f"  Frame Range: 0 to {len(labels)-1}")
    print(f"  Class Distribution: H: {counts['H']} ({counts['H']/len(labels):.1%}), "
          f"A: {counts['A']} ({counts['A']/len(labels):.1%}), "
          f"D: {counts['D']} ({counts['D']/len(labels):.1%}), "
          f"X: {counts['X']} ({counts['X']/len(labels):.1%})")

    # 2. Public Slice Benchmark Evaluation
    print(f"\n[2/2] Evaluating against Public Ground Truth: {GT_FILE}")
    gt_codes = pm.label_codes(pm.read_label_csv(GT_FILE))
    pred_codes = pm.label_codes(table)

    pub_slice = slice(11185, 18685)
    gt_pub = gt_codes[pub_slice]
    pred_pub = pred_codes[pub_slice]

    res = pm.evaluate(gt_pub, pred_pub, bootstrap_n=0)

    print("\n" + "-" * 80)
    print(" OFFICIAL SCOREBOARD BENCHMARK RESULTS")
    print("-" * 80)
    print(f"  Mean Spell F1 (Official Ranking Metric): {res['spell_f1_mean']:.4f}")
    print(f"  Macro F1 (Official First Tie-Break):     {res['macro_f1']:.4f}")
    print(f"  Alive / Dead Classification F1:          {res['alive_dead_f1']:.4f}")
    print(f"  Transition Alignment F1 (tau=25 frames): {res['transitions'][25]['f1']:.4f}")
    print(f"  Number of Predicted Possession Spells:   {res['n_pred_spells']}")
    print(f"  Per-IoU Spell F1 Breakdown:")
    for iou in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"    IoU @ {iou:.1f}: F1 = {res['spell_f1'][iou]['f1']:.4f} "
              f"(Precision = {res['spell_f1'][iou]['precision']:.4f}, "
              f"Recall = {res['spell_f1'][iou]['recall']:.4f}, "
              f"Matched = {res['spell_f1'][iou]['n_matched']})")

    print("\n" + "=" * 80)
    print(" VERIFICATION COMPLETE: SUBMISSION READY FOR FORM UPLOAD")
    print("=" * 80)


if __name__ == "__main__":
    main()
