from transformers import AutoModelForCausalLM, AutoTokenizer
import os

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
model_dir = "./saved_model"

# Load the model and tokenizer
device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Save the model and tokenizer
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)