import logging
import logging.handlers
import requests
import json
import csv
import os
import pytz
from datetime import date, timedelta, datetime

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_file_handler = logging.handlers.RotatingFileHandler(
    "status.log",
    maxBytes=1024 * 1024,
    backupCount=1,
    encoding="utf8",
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger_file_handler.setFormatter(formatter)
logger.addHandler(logger_file_handler)

# API endpoint URL
url = "https://portal-datahub-24vn-ews.education.wise-paas.com/api/v1/Command/writeValue"

# Data payload as specified in the curl command
data_payload = [
    {
        "nodeId": "14b290c2-d759-4247-ae6e-d55e64656aba",
        "deviceId": "Device1",
        "tagName": "Predicted_EC_Value",
        "value": "24"
    }
]

# Replace 'your_token_here' with your actual token
header = {'Content-Type': 'application/json'}
body = json.dumps({"username": "10422050@student.vgu.edu.vn", "password": "VGUrangers2024@"})
r = requests.post("https://portal-sso-ensaas.education.wise-paas.com/v4.0/auth/native", headers=header, data=body)

# Get the token
try:
    result = r.json()
    myToken = result["accessToken"]
    logger.info("Successfully retrieved access token.")
except (KeyError, json.JSONDecodeError) as e:
    logger.error("Failed to retrieve access token.", exc_info=True)
    raise

# Headers including the authorization token
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {myToken}"
}

# Make the POST request
response = requests.post(url, headers=headers, json=data_payload)

# Check the response status
if response.status_code == 200:
    print("Data successfully uploaded to the platform.")
    # Print response content if needed
    print("Response:", response.json())
else:
    print(f"Request failed with status code: {response.status_code}")
    print("Error:", response.text)
