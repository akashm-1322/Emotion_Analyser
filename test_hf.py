from dotenv import load_dotenv
import os
import requests

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
print("HF_TOKEN loaded:", HF_TOKEN is not None)

API_URL = "https://router.huggingface.co/hf-inference/models/AnasAlokla/multilingual_go_emotions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "inputs": "I feel very happy today"
}

response = requests.post(API_URL, headers=headers, json=payload)

print("Status:", response.status_code)
print("Response:", response.text)
