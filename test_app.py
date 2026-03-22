from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_high_risk():
    response = client.post("/predict-risk", json={
        "ticket_7d": 5,
        "ticket_30d": 10,
        "ticket_90d": 15,
        "sentiment_score": -0.5,
        "monthly_change": 20
    })

    assert response.status_code == 200
    assert response.json()["risk_category"] in ["HIGH", "LOW"]