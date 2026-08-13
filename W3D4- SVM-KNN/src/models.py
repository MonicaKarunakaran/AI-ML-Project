from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class SVMModel:
    def __init__(self):
        self.model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class KNNModel:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)