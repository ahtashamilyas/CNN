# Import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Falcon model and tokenizer from Hugging Face
model_name = "tiiuae/falcon-7b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


prompt = "In a distant future, humanity has reached the stars"

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text using the model
output = model.generate(inputs['input_ids'], max_length=100)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
