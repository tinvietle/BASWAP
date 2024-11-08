from transformers import AutoModelForCausalLM, AutoTokenizer
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

context = """
    As a friendly data analysist asisstant who specializes in water technology, analyzing water quality mesurements.
    Your main audience is the farmers who do not have adept knowledge in data analysis.
    The farmer has a paddy field and a river next to it. The salinity level is recorded in the river.
    You job is to protect the field and help the farmer decide when to open or close the gate
    If the outside water is saline, open the gate will harm the plant. Only open the gate if the outside water is not saline.
    - If the salinity level of the outside water is higher than 0.5, the water is saline so close the gate.
    - If the salinity level of the outside water is less than 0.5, the water is not saline so open the gate.
    <EXAMPLE>
    The salinity is 0.4, open the gate. Because the water
    The salinity is 0.6, close the gate.
    The salinity is 0.3, open the gate.
    The salinity is 0.7, close the gate.
    </EXAMPLE>
"""

request = f"""
    If the salinity is 0, open or close the gate? Exlain why.
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
print(tokenizer.decode(outputs[0]))