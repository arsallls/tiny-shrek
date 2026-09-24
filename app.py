"""Gradio demo. Scene generator, not a chatbot - it continues dialogue, it doesn't answer."""
import gradio as gr
import torch

from sample import load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, STOI, ITOS = load("shrek.pt", DEVICE)
CHARACTERS = ["SHREK", "DONKEY", "FIONA", "FARQUAAD", "PUSS"]


def generate(character, setup, temperature, tokens):
    prompt = (setup.strip() + "\n" if setup.strip() else "") + f"{character}: "
    ids = [STOI[c] for c in prompt if c in STOI] or [0]
    idx = torch.tensor([ids], device=DEVICE)
    out = MODEL.generate(idx, int(tokens), float(temperature))
    return "".join(ITOS[int(t)] for t in out[0])


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Dropdown(CHARACTERS, value="SHREK", label="Speaker"),
        gr.Textbox(label="Scene setup (optional)", placeholder="DONKEY: Are we there yet?"),
        gr.Slider(0.2, 1.4, value=0.8, label="Temperature"),
        gr.Slider(100, 1000, value=400, step=50, label="Length"),
    ],
    outputs=gr.Textbox(label="Generated scene", lines=18),
    title="Tiny-Shrek: a 14M-param transformer trained from scratch",
    description="Pretrained on 617 movies, finetuned on Shrek. Generates dialogue "
                "in character - it does not answer questions.",
)

if __name__ == "__main__":
    demo.launch()
