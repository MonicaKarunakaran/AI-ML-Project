from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy")
    print(accuracy)

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    return accuracy