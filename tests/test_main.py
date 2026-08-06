from tests.conftest import client


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["message"] == "Welcome to FastAPI ML Model Serving API"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "Healthy"


def test_prediction():

    response = client.post(
        "/predict",
        json={
            "sepal_length":5.1,
            "sepal_width":3.5,
            "petal_length":1.4,
            "petal_width":0.2
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "species" in data