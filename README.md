# Tiny-Shrek

A 14M-parameter Llama-style transformer trained from scratch on movie dialogue,
then finetuned to a single character. No pretrained weights anywhere.

## Result

| model | held-out Shrek BPC | ppl |
|---|---|---|
| pretrained on 617 movies, no finetune | _fill in_ | |
| **pretrained + finetuned on Shrek** | _fill in_ | |
| Shrek-only, from scratch (no pretrain) | _fill in_ | |

The third row is the point: same architecture, same Shrek data, no stage-1
pretraining. The gap is what pretraining bought.

## Architecture (`model.py`)

RoPE · RMSNorm · SwiGLU · no biases · tied embeddings · flash attention via SDPA.
8 layers, 6 heads, 384 embd, 512 context, character-level (~100 vocab).

## Pipeline

```bash
python prepare.py cornell              # stage-1 corpus, whole movies held out for val
python train.py --stage pretrain --out_dir $CKPT --compile
python prepare.py shrek --inspect      # eyeball the screenplay parse
python prepare.py shrek
python train.py --stage finetune --out_dir $CKPT
python train.py --stage scratch --out_dir out/ablation   # the baseline
python eval.py --ckpt $CKPT/pretrain.pt --ckpt $CKPT/finetune.pt --split shrek_val
python eval.py --ckpt $CKPT/finetune.pt --memorize
python sample.py --ckpt $CKPT/finetune.pt --bench
```

## Notes

- **Whole-movie val split.** Random-line splits leak context across the boundary.
- **Replay mixing.** The finetune blends 15% stage-1 data; without it the model
  forgets general English and collapses into Shrek-flavoured noise.
- **Memorization audit.** 14M params on 25MB recites. `eval.py --memorize` reports
  verbatim k-gram overlap against the training corpus.
- **KV cache.** `sample.py --bench` reports tokens/sec with and without.

## Limits

Generates in-character dialogue; it is not a chatbot and will not answer questions.
At this scale output is locally coherent and stylistically right, not semantically
grounded.

The corpora are not committed: Cornell Movie-Dialogs has its own distribution terms
and the scripts are copyrighted.
