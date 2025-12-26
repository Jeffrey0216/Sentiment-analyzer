import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict(text):
    result = classifier(text)
    return result[0]["label"]

with gr.Blocks() as demo:
    gr.Markdown("# Sentiment Analysis")
    text = gr.Textbox(label="Enter text")
    output = gr.Textbox(label="Sentiment")
    btn = gr.Button("Analyze")
    btn.click(fn=predict, inputs=text, outputs=output)

demo.launch()
