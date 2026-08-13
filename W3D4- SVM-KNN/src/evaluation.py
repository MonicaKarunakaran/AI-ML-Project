from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(y_true, predictions):

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(y_true, predictions)

    recall = recall_score(y_true, predictions)

    f1 = f1_score(y_true, predictions)

    return accuracy, precision, recall, f1