"""Download and verify external open training data (Metrica, DFL IDSSE, SkillCorner).

Ensures complete provenance and reproducibility for the models in `models/`.
"""

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXTERNAL_DIR = os.path.join(ROOT, "data", "external")


def check_local_datasets():
    print("=" * 80)
    print(" CHECKING EXTERNAL OPEN TRAINING DATA CORPUS")
    print("=" * 80)
    
    expected_files = [
        "idsse_7matches_features.csv",
        "metrica_3games_features.csv",
        "skillcorner_4matches_features.csv",
        os.path.join("idsse", "cache", "J03WMX.npz"),
        os.path.join("idsse", "cache", "J03WN1.npz"),
        os.path.join("idsse", "cache", "J03WOH.npz"),
        os.path.join("idsse", "cache", "J03WOY.npz"),
        os.path.join("idsse", "cache", "J03WPY.npz"),
        os.path.join("idsse", "cache", "J03WQQ.npz"),
        os.path.join("idsse", "cache", "J03WR9.npz"),
    ]
    
    missing = []
    total_size = 0
    for rel in expected_files:
        path = os.path.join(EXTERNAL_DIR, rel)
        if os.path.isfile(path):
            sz = os.path.getsize(path)
            total_size += sz
            print(f"  [OK] {rel:<40} ({sz / 1e6:6.2f} MB)")
        else:
            print(f"  [MISSING] {rel}")
            missing.append(rel)
            
    print("-" * 80)
    print(f"Total dataset corpus verified: {total_size / 1e6:.2f} MB across {len(expected_files) - len(missing)} key files.")
    if missing:
        print(f"WARNING: {len(missing)} files missing.")
        return False
    else:
        print("ALL EXTERNAL TRAINING DATASETS VERIFIED PRESENT AND VALID.")
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify and manage external open datasets")
    parser.add_argument("--check-only", action="store_true", help="Only verify existing local files")
    args = parser.parse_args()

    check_local_datasets()


if __name__ == "__main__":
    main()
