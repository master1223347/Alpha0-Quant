import math

import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.metrics import compute_classification_metrics, sigmoid

Y_TRUE = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
Y_PROB = [0.1, 0.8, 0.65, 0.4, 0.4, 0.2, 0.9, 0.55, 0.7, 0.3]


def test_matches_sklearn_reference_values():
    metrics = compute_classification_metrics(Y_TRUE, Y_PROB)
    assert metrics.auc == pytest.approx(roc_auc_score(Y_TRUE, Y_PROB))
    assert metrics.brier_score == pytest.approx(brier_score_loss(Y_TRUE, Y_PROB))
    distinct = [0.1, 0.8, 0.65, 0.4, 0.45, 0.2, 0.9, 0.55, 0.7, 0.3]
    assert compute_classification_metrics(Y_TRUE, distinct).average_precision == pytest.approx(
        average_precision_score(Y_TRUE, distinct)
    )


def test_confusion_counts_at_threshold():
    metrics = compute_classification_metrics(Y_TRUE, Y_PROB, threshold=0.5)
    assert (metrics.tp, metrics.fp, metrics.tn, metrics.fn) == (4, 1, 4, 1)
    assert metrics.accuracy == pytest.approx(0.8)


def test_single_class_auc_is_nan():
    assert math.isnan(compute_classification_metrics([1, 1], [0.2, 0.9]).auc)


def test_input_validation():
    with pytest.raises(ValueError):
        compute_classification_metrics([1], [0.5, 0.5])
    with pytest.raises(ValueError):
        compute_classification_metrics([], [])


def test_sigmoid_is_stable_for_large_inputs():
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0)
