import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_name = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter your prompt", lines=3),
        gr.Slider(minimum=50, maximum=500, value=100, label="Max Length")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=5),
    title="Gemma 3 270M Text Generator",
    description="Generate text using Google's Gemma 3 270M model"
)

iface.launch()
