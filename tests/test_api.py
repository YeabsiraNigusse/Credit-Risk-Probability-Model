import requests

url = "http://localhost:8000/predict"
data = {
    "customer_id": "CUST001",
    "recency_days": 14,
    "frequency": 9,
    "monetary_value": 850.5,
    "avg_transaction_value": 94.5,
    "fraud_count": 2,
    "total_transactions": 11,
    "fraud_ratio": 0.18
}

response = requests.post(url, json=data)
print(response.json())
