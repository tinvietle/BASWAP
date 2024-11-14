from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import logging.handlers
import requests
import json
import csv
import os
import pytz
from datetime import date, timedelta, datetime
from datahub import sendMessage
import pandas as pd

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

try:
    USERNAME_DATAHUB = os.environ["USERNAME_DATAHUB"]
    
    PASSWORD_DATAHUB = os.environ["PASSWORD_DATAHUB"]
    
except KeyError:
    logger.info("Environment variables not set!")
    #raise

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_dir = "./saved_model"
# device = "cpu"  # or "cuda" for GPU

# # Load the model and tokenizer from the saved directory
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

# Directory containing the CSV files
directory = 'EC_DataHub'

# Get the list of all files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Get the latest file based on the modification time
latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))

# Read the latest CSV file
df = pd.read_csv(os.path.join(directory, latest_file))

# Calculate the average value
average_value = round(df['value'].mean())

print(f'The average value is: {average_value}')

context = """
    You are a helpful data analysis assistant specializing in water quality for farmers. Your main job is to help farmers who have limited knowledge of data analysis understand when to open or close a gate to protect their paddy fields from saline water.

    The paddy field is next to a river where salinity levels are recorded. Here is the rule:
    - If the salinity level is above 300, the water is saline, so close the gate.
    - If the salinity level is 300 or below, the water is suitable, so open the gate.

    <EXAMPLES>
    User: The salinity level is 300, open or close the gate? - Assistant: The water is suitable, open the gate.
    User: The salinity level is 500, open or close the gate? - Assistant: The water is saline, close the gate.
    User: The salinity level is 200, open or close the gate? - Assistant: The water is suitable, open the gate.
    User: The salinity level is 450, open or close the gate? - Assistant: The water is saline, close the gate.
    User: The salinity level is 550, open or close the gate? - Assistant: The water is saline, close the gate.
    User: The salinity level is 100, open or close the gate? - Assistant: The water is suitable, open the gate.
    </EXAMPLES>
"""

request = f"""
    The salinity level is {average_value}, should we open or close the gate?
"""

messages = [
    {"role": "system", "content": f"{context}"},
    {"role": "user", "content": f"{request}"}
]

input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=256, temperature=0.01, top_p=1, do_sample=True)
response = tokenizer.decode(outputs[0])


# Extract content between the specific tokens
start_token = "<|im_start|>assistant"
end_token = "<|im_end|>"
start_idx = response.rfind(start_token) + len(start_token)
end_idx = response.find(end_token, start_idx)
assistant_response = response[start_idx:end_idx]

print(assistant_response)
logger.info(assistant_response)
sendMessage(USERNAME_DATAHUB, PASSWORD_DATAHUB, "Prompt", assistant_response)

# print(tokenizer.decode(outputs[0]))
# logger.info(f'{tokenizer.decode(outputs[0])}')