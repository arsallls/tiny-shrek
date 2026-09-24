"""Gradio demo. Scene generator, not a chatbot - it continues dialogue, it doesn't answer."""
import gradio as gr
import torch

from sample import load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, STOI, ITOS = load("finetune.pt", DEVICE)
CHARACTERS = ["SHREK", "DONKEY", "FIONA", "FARQUAAD", "PUSS"]

# Context is load-bearing, not decoration. Stage 1 saw ~131M tokens of general movie
# dialogue and stage 2 only ~278k of Shrek, so an unseeded prompt drifts back to the
# pretraining distribution within a few turns. A couple of real lines pin the mode.
SEED = ("SHREK: What are you doing in my swamp?\n"
        "DONKEY: I'm all alone, there's no one here beside me.")


def generate(character, setup, temperature, tokens):
    prompt = (setup.strip() or SEED) + "\n" + f"{character}: "
    ids = [STOI[c] for c in prompt if c in STOI] or [0]
    idx = torch.tensor([ids], device=DEVICE)
    out = MODEL.generate(idx, int(tokens), float(temperature))
    return "".join(ITOS[int(t)] for t in out[0])


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Dropdown(CHARACTERS, value="SHREK", label="Speaker"),
        gr.Textbox(label="Scene setup (a line or two of dialogue anchors the voice)",
                   value=SEED, lines=3),
        gr.Slider(0.2, 1.4, value=0.8, label="Temperature"),
        gr.Slider(100, 1000, value=400, step=50, label="Length"),
    ],
    outputs=gr.Textbox(label="Generated scene", lines=18),
    title="Tiny-Shrek: a 14M-param transformer trained from scratch",
    description="Pretrained from scratch on 617 movies, then finetuned on Shrek. "
                "Generates in-character dialogue - it is a scene generator, not a "
                "chatbot, and will not answer questions. Keep a line or two of setup: "
                "without context it drifts back to generic movie dialogue.",
)

if __name__ == "__main__":
    demo.launch()
