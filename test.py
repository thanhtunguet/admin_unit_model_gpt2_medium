# test_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the locally saved model
model_path = "./admin_unit_model_gpt2_medium/checkpoint-14136"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define the function to generate output from the model
def generate_address_standardization(input_text, max_length=512):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)  # Explicitly pass attention mask
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Pass attention mask here
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the generated text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Example input text for testing
input_address = "Provide details for the administrative unit Tam Hồng in Yên Lạc, Vĩnh Phúc."
output = generate_address_standardization(input_address)

print("Input Address:", input_address)
print("Standardized Address:", output)
