import requests
import json
import os

API_KEY = os.environ.get("API_KEY", "YOUR_API_KEY")
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": API_KEY
}

payload = {
    "contents": [
        {"parts": [{"text": "Explain how AI works in a few words"}]}
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

data = response.json()
print(data["candidates"][0]["content"]["parts"][0]["text"])