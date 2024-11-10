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

url = "https://portal-datahub-24vn-ews.education.wise-paas.com/api/v1/HistData/raw"

# startTs = (date.today() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
targetDate = (date.today() - timedelta(days=1))
startTs = (targetDate - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
endTs = date.today().strftime("%Y-%m-%dT%H:%M:%S.000Z")

params = {
    "tags": [
        {
            "nodeId": "14b290c2-d759-4247-ae6e-d55e64656aba",
            "deviceId": "Device1",
            "tagName": "EC_value",
        }
    ],
    "startTs": startTs,
    "endTs": endTs,
    "desc": False,
    "count": 2000
}

try:
    USERNAME_LOGIN = os.environ["USERNAME_LOGIN"]
    PASSWORD_LOGIN = os.environ["PASSWORD_LOGIN"]
    print(USERNAME_LOGIN)
except KeyError:
    logger.info("Environment variables not set!")
    #raise

header = {'Content-Type': 'application/json'}
body = json.dumps({"username": str(USERNAME_LOGIN), "password": str(PASSWORD_LOGIN)})
r = requests.post("https://portal-sso-ensaas.education.wise-paas.com/v4.0/auth/native", headers=header, data=body)

# Get the token
try:
    result = r.json()
    myToken = result["accessToken"]
    logger.info("Successfully retrieved access token.")
except (KeyError, json.JSONDecodeError) as e:
    logger.error("Failed to retrieve access token.", exc_info=True)
    raise

headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {myToken}'
}

response = requests.post(url, headers=headers, json=params)

while (response.status_code != 200):
    response = requests.post(url, headers=headers, json=params)

if response.status_code == 200:
    try:
        data = response.json()
        logger.info("Data successfully retrieved from API.")

        # Timezones: UTC and target timezone GMT+7
        utc_tz = pytz.timezone('UTC')
        gmt_plus_7_tz = pytz.timezone('Asia/Bangkok')

        # Extract values, convert timestamps to GMT+7, and filter for start_date_plus_one
        extracted_values = [
            {
                'timestamp': gmt_plus_7_time.strftime('%Y-%m-%d %H:%M:%S'),
                'value': entry['value']
            }
            for tag in data
            for entry in tag.get('values', [])
            if (gmt_plus_7_time := datetime.strptime(entry['ts'], '%Y-%m-%dT%H:%M:%S.%fZ')
                .replace(tzinfo=utc_tz)
                .astimezone(gmt_plus_7_tz)).date() == targetDate
        ]
        # Specify the folder where files are saved
        folder_path = 'EC_DataHub'

        # Generate a dynamic filename by the current day
        csv_filename = os.path.join(folder_path, f'{targetDate.strftime("%Y-%m-%d")}.csv')

        output_file = 'output_values.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['value', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(extracted_values)

        logger.info(f"Data of datahub successfully written to {csv_filename}")

    except json.JSONDecodeError:
        logger.error("Response is not valid JSON.", exc_info=True)
        print("Response is not valid JSON.")
else:
    logger.error(f"Request failed with status code: {response.status_code}")
    print(f"Request failed with status code: {response.status_code}")
