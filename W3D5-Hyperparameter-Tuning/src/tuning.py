from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def grid_search_svm(X_train, y_train):

    params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    model = SVC()

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid


def random_search_knn(X_train, y_train):

    params = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    model = KNeighborsClassifier()

    random = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=6,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )

    random.fit(X_train, y_train)

    return random