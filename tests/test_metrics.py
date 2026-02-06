from governed_nlp.modeling.metrics import adjacent_accuracy, weighted_kappa # type: ignore

def test_adjacent_accuracy():
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 2, 2, 4, 4]  # off-by-one for indices 1 and 3
    assert adjacent_accuracy(y_true, y_pred) == 1.0

def test_weighted_kappa_range():
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    k = weighted_kappa(y_true, y_pred)
    assert -1.0 <= k <= 1.0
