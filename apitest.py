import requests

url = "http://localhost:8000/classify"
payload = {
    "text": "سلام، من در مورد محصول خریداری شده مشکل دارم. لطفا راهنمایی کنید"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(response.json())
