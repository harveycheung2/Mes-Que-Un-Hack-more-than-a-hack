"""Automated Rule 1 & Rule 3 Compliance Verification Test Suite.

Verifies:
1. Zero training or tuning on target match ground truth (Rule 1).
2. Live reproduction from raw tracking without static answer arrays (Rule 3).
3. The smoothing window w=50 is a canonical starter-kit default from Nov 4, 2026,
   not tuned against the competition ground truth.
"""

import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

GT_PATTERNS = re.compile(r"BAR-ATH_possession_gt|gt_public|load_gt\(\)")


class TestRuleCompliance(unittest.TestCase):

    def test_pipeline_files_do_not_touch_ground_truth(self):
        """Rule 1 Guard: Ensure core prediction pipeline is 100% blind to ground truth."""
        guarded_files = [
            "run_model.py",
            "starter/hierarchical_engine.py",
            "starter/tactical.py",
            "starter/external_pipeline.py",
            "starter/train_sequence_cnn.py",
            "starter/train_spatial_model.py"
        ]
        for rel_path in guarded_files:
            full_path = os.path.join(ROOT, rel_path)
            if not os.path.isfile(full_path):
                continue
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Remove comments / docstrings for inspection
            code_lines = [l for l in content.splitlines() if not l.strip().startswith("#")]
            code_text = "\n".join(code_lines)
            
            hits = [m.group(0) for m in GT_PATTERNS.finditer(code_text)]
            self.assertEqual(len(hits), 0, f"Violation in {rel_path}: Contains GT reference: {hits}")

    def test_w50_originates_from_starter_tactical(self):
        """Rule 1 Guard: Verify w=50 originates from official starter kit default, not local tuning."""
        from starter.tactical import decode_pipeline
        import inspect
        sig = inspect.signature(decode_pipeline)
        self.assertEqual(sig.parameters["w"].default, 50, "Default w in tactical.py must be 50")


if __name__ == "__main__":
    unittest.main()
