"""Small deterministic negative-control and execution-model tests."""

from __future__ import annotations

import random
import unittest

from src.evaluation.execution_models import enforce_participation_caps
from src.evaluation.metrics import compute_classification_metrics


class NegativeControlsTest(unittest.TestCase):
    def test_randomized_probabilities_lose_perfect_event_auprc(self) -> None:
        labels = [1] * 20 + [0] * 80
        perfect_probs = [0.9] * 20 + [0.1] * 80
        shuffled_probs = list(perfect_probs)
        random.Random(7).shuffle(shuffled_probs)

        perfect = compute_classification_metrics(labels, perfect_probs)
        shuffled = compute_classification_metrics(labels, shuffled_probs)

        self.assertGreater(perfect.average_precision, 0.99)
        self.assertLess(shuffled.average_precision, perfect.average_precision)

    def test_thin_name_pov_rejection(self) -> None:
        self.assertEqual(enforce_participation_caps(requested_pov=0.10, max_pov=0.03, reject_excess=True), 0.0)
        self.assertAlmostEqual(
            enforce_participation_caps(requested_pov=0.10, max_pov=0.03, reject_excess=False),
            0.30,
        )


if __name__ == "__main__":
    unittest.main()
