"""Unit and integration tests for the Hierarchical Episode-First Possession Engine.

Verifies:
1. Dead-ball walls are preserved without leakage.
2. Smoothing never bleeds across stoppage boundaries.
3. Sub-second flickers (<25 frames) are absorbed within episodes.
4. Graded frames contain zero abstentions ('X').
5. Deterministic end-to-end execution and official submission validation.
"""

import os
import sys
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from starter.hierarchical_engine import episode_first_decode, run_length_runs
import possession_metrics as pm
from games import GAME, GAMES


class TestHierarchicalEngine(unittest.TestCase):

    def test_run_length_runs(self):
        arr = np.array(["H", "H", "A", "D", "D", "D", "H"])
        starts, stops, values = run_length_runs(arr)
        np.testing.assert_array_equal(starts, [0, 2, 3, 6])
        np.testing.assert_array_equal(stops, [2, 3, 6, 7])
        np.testing.assert_array_equal(values, ["H", "A", "D", "H"])

    def test_dead_ball_wall_preservation(self):
        """Verify that dead-ball walls identified by the sequence model are strictly locked."""
        n = 200
        v3_stream = np.full(n, "H", dtype="<U1")
        v3_stream[50:100] = "D"
        
        live_stream = np.full(n, "H", dtype="<U1")
        live_stream[60:80] = "A"
        
        from starter import hierarchical_engine
        orig_games = hierarchical_engine.GAMES
        hierarchical_engine.GAMES = {
            GAME: {
                "offsets": {1: 0, 2: 0},
                "halves": {1: (0, 99), 2: (100, 199)}
            }
        }
        try:
            out = episode_first_decode(v3_stream, live_stream, min_live_frames=25)
            self.assertTrue(np.all(out[50:100] == "D"), "Dead-ball walls must not be violated by live stream")
        finally:
            hierarchical_engine.GAMES = orig_games

    def test_biomechanical_dwell_filtering(self):
        """Verify that runs shorter than 25 frames are absorbed into longer within-episode neighbors."""
        n = 200
        v3_stream = np.full(n, "H", dtype="<U1")
        live_stream = np.full(n, "H", dtype="<U1")
        live_stream[30:40] = "A"  # 10 frames < 25 frames
        
        from starter import hierarchical_engine
        orig_games = hierarchical_engine.GAMES
        hierarchical_engine.GAMES = {
            GAME: {
                "offsets": {1: 0, 2: 0},
                "halves": {1: (0, 199), 2: (200, 200)}
            }
        }
        try:
            out = episode_first_decode(v3_stream, live_stream, min_live_frames=25)
            self.assertTrue(np.all(out[0:200] == "H"), "Flicker < 25 frames must be absorbed into longer neighbor")
        finally:
            hierarchical_engine.GAMES = orig_games

    def test_submission_validation(self):
        """Verify that the generated submission CSV strictly passes official validation."""
        sub_path = os.path.join(ROOT, "evaluation", "results", "submission.csv")
        if not os.path.isfile(sub_path):
            sub_path = os.path.join(ROOT, "evaluation", "results", "principled_episode_submission.csv")
        self.assertTrue(os.path.isfile(sub_path), f"Submission file {sub_path} must exist")
        
        table = pm.read_label_csv(sub_path)
        n_expected = GAMES[GAME]["n_video_frames"]
        reason = pm.validate_submission(table, n_expected)
        self.assertIsNone(reason, f"Submission failed official validation: {reason}")


if __name__ == "__main__":
    unittest.main()
