from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_high_risk():
    response = client.post("/predict-risk", json={
        "monthly_charge": 100,
        "previous_monthly_charge": 80,
        "contract_type": "Month-to-Month",
        "tickets": [
            {"category": "complaint", "days_ago": 10},
            {"category": "complaint", "days_ago": 15},
            {"category": "complaint", "days_ago": 20},
            {"category": "complaint", "days_ago": 25},
            {"category": "complaint", "days_ago": 5},
            {"category": "complaint", "days_ago": 2}
        ]
    })

    assert response.status_code == 200
    assert response.json()["risk_category"] == "HIGH"