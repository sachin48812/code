import requests
import json

# Configure camera
url = 'http://localhost:5000/api/config'
data = {
    'camera_urls': {
        'north': 'http://192.168.5.41:8080/video',
        'east': '',
        'south': '',
        'west': ''
    }
}

print("Configuring camera...")
try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Verify configuration
    response = requests.get('http://localhost:5000/api/config')
    config = response.json()
    print(f"Current camera URLs: {config['camera_urls']}")
except Exception as e:
    print(f"Error: {e}")