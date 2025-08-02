import requests
import os

# 1) Choose the correct URL. 
#    If you run this on your HOST machine: use localhost:8000
#    If you exec inside another container in the same compose network: use api:8000
API_HOST = os.getenv("API_HOST", "http://localhost:8000")
URL = f"{API_HOST}/predict"

payload = {
    "recency_days": 12.5,
    "frequency":    4,
    "monetary":     245.75,
    "fraud_ratio":  0.18
}

resp = requests.post(URL, json=payload)

# 2) Debug output
print("→ URL:", URL)
print("→ Status code:", resp.status_code)
print("→ Raw response text:")
print(resp.text or "<empty>")

# 3) Try to parse JSON, but don’t crash if it isn’t JSON
try:
    data = resp.json()
    print("→ Parsed JSON:", data)
except ValueError:
    print("→ Response was not valid JSON.")
