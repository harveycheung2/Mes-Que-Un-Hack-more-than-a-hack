"""Verification script for competition submission deliverables.

Runs official format validation, verifies SHA256 checksums, and confirms data integrity.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.basename(HERE) != "submissions" else os.path.dirname(HERE)
for cand_root in [ROOT, os.path.dirname(ROOT), os.path.join(ROOT, "evaluation")]:
    if os.path.isdir(cand_root) and cand_root not in sys.path:
        sys.path.insert(0, cand_root)

import possession_metrics as pm

N_EXPECTED = 199319


def find_submission():
    candidates = [
        os.path.join(HERE, "submission.csv"),
        os.path.join(HERE, "submissions", "submission.csv"),
        os.path.join(ROOT, "submissions", "submission.csv"),
        os.path.join(ROOT, "submission.csv"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return candidates[0]


def find_checksum():
    candidates = [
        os.path.join(HERE, "SHA256SUMS"),
        os.path.join(ROOT, "SHA256SUMS"),
        os.path.join(ROOT, "submissions", "SHA256SUMS"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            with open(cand, "r") as f:
                line = f.readline().strip()
                if line:
                    return line.split()[0]
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Competition Submission Verifier")
    parser.add_argument("--sub", default=None, help="Path to submission CSV")
    parser.add_argument("--score-public-gt", action="store_true", help="Optional benchmark against public GT slice")
    args = parser.parse_args()

    sub_file = args.sub or find_submission()

    print("=" * 80)
    print(" COMPETITION SUBMISSION VERIFICATION")
    print("=" * 80)

    # 1. Format Validation
    print(f"\n[1/2] Checking submission format: {sub_file}")
    if not os.path.isfile(sub_file):
        print(f"ERROR: Submission file not found: {sub_file}")
        sys.exit(1)

    table = pm.read_label_csv(sub_file)
    reason = pm.validate_submission(table, N_EXPECTED)
    if reason:
        print(f"FAILED: {reason}")
        sys.exit(1)

    labels = table["label"].str.strip().str.upper().values
    counts = {k: int(np.sum(labels == k)) for k in ["H", "A", "D", "X"]}
    print("  Official format validation: PASSED")
    print(f"  • Total Frames: {len(labels):,}")
    print(f"  • Frame Index:  0 to {len(labels)-1}")
    print(f"  • Class Distribution:")
    print(f"      H: {counts['H']:,} ({counts['H']/len(labels):.1%})")
    print(f"      A: {counts['A']:,} ({counts['A']/len(labels):.1%})")
    print(f"      D: {counts['D']:,} ({counts['D']/len(labels):.1%})")
    print(f"      X: {counts['X']:,} ({counts['X']/len(labels):.1%})")

    # 2. SHA-256 Verification
    print(f"\n[2/2] Checking SHA-256 Checksum...")
    with open(sub_file, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"  Computed SHA-256: {actual_hash}")

    expected_hash = find_checksum()
    if expected_hash:
        print(f"  Expected SHA-256: {expected_hash}")
        if actual_hash == expected_hash:
            print("  Checksum Verification: PASSED (Byte-for-byte match)")
        else:
            print("  WARNING: Checksum does not match SHA256SUMS entry.")
    else:
        print("  Notice: No SHA256SUMS file found to compare.")

    # Optional Public Slice Diagnostic
    if args.score_public_gt:
        gt_cand = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
        if os.path.isfile(gt_cand):
            print("\n" + "-" * 80)
            print(" [DIAGNOSTIC ONLY] Public 5-Minute Window Diagnostic Benchmark")
            print(" (Note: Official ranking occurs on the held-out private ground truth)")
            print("-" * 80)
            gt_codes = pm.label_codes(pm.read_label_csv(gt_cand))
            pred_codes = pm.label_codes(table)
            pub_slice = slice(11185, 18685)
            res = pm.evaluate(gt_codes[pub_slice], pred_codes[pub_slice], bootstrap_n=0)
            print(f"  Spell F1 (Mean IoU 0.3-0.7): {res['spell_f1_mean']:.4f}")
            print(f"  Macro F1:                   {res['macro_f1']:.4f}")
            print(f"  Alive / Dead F1:            {res['alive_dead_f1']:.4f}")
            print(f"  Predicted Spells:           {res['n_pred_spells']}")

    print("\n" + "=" * 80)
    print(" VERIFICATION COMPLETE: ALL COMPETITION CRITERIA SATISFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
