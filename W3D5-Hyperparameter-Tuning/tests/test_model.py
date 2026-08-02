import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_prep import load_data
from src.tuning import grid_search_svm


def test_model():
    X_train, X_test, y_train, y_test = load_data()

    model = grid_search_svm(X_train, y_train)

    assert model.best_score_ > 0