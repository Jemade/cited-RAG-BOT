import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import get_db

@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert "embedding_model" in data

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "total_queries" in data
    assert "refusal_rate" in data
