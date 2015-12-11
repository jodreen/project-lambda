from stat159lambda.classification.random_forest import rf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
try:
    from mock import patch
except:
    from unittest.mock import patch
from numpy.testing import assert_almost_equal, assert_array_equal


def test_prepare():
    X = np.array([[10, 8, 8], [1, 11, 8], [4, 1, 1], [5, 0, 3], [7, 9, 1],
                  [1, 5, 8], [6, 3, 0], [7, 10, 3], [5, 4, 3], [7, 0, 8]])
    y = np.array([1, 0, 1, 1, 0, 2, 1, 1, 0, 0])

    return X, y


@patch.object(RandomForestClassifier, '__init__')
def test_rf(mock_rf_init):
    X, y = test_prepare()
    mock_rf_init.return_value = None
    test_classifier = rf.Classifier(X, y, depth=2)
    mock_rf_init.assert_called_with(max_depth=None, max_features='auto', n_estimators=400, n_jobs=-1, oob_score=False)
    test_classifier.train()
    assert test_classifier.predict(X) is not None
