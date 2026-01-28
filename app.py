from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load model and tokenizer
model_name = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.get("/")
def read_root():
    return {"message": "Gemma 3 270M API is running"}

@app.post("/generate")
def generate_text(prompt: str, max_length: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}
