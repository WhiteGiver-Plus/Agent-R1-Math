import requests
import json

url = "http://localhost:5000/api/retrieve"
payload = {
    "queries": ["求证三角形内角和为180度", "计算圆柱体体积"]*20,
    "num": 5
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(payload), headers=headers)

# 打印响应
print(f"Status Code: {response.status_code}")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))