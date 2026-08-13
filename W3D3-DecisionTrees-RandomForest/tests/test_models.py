import pytest

from src.decision_tree import train_decision_tree
from src.random_forest import train_random_forest



def test_decision_tree_training():

    model = train_decision_tree()

    assert model is not None



def test_random_forest_training():

    model = train_random_forest()

    assert model is not None