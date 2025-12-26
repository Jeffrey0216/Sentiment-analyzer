import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict(text):
    return classifier(text)[0]["label"]

with gr.Blocks() as demo:
    gr.Markdown("# Sentiment Analysis")
    inp = gr.Textbox(label="Enter text")
    out = gr.Textbox(label="Sentiment")
    btn = gr.Button("Analyze")
    btn.click(predict, inp, out)

demo.launch(server_name="0.0.0.0", server_port=7860)

