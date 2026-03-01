# microgpt-rs

The most atomic way to train and run inference for a GPT — in pure Rust, in a single file.

Inspired by [Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). Same educational spirit, idiomatic Rust. Tape-based autograd instead of pointer graph.

## What's inside

- **Tape-based autograd** (Wengert list) with reverse-mode differentiation
- **Transformer**: multi-head attention, RMSNorm (optionally with learnable gamma), SiLU or ReLU activation
- **AdamW** optimizer with cosine or linear LR schedule, warmup, and gradient clipping
- **Char-level tokenizer** with BOS token
- **Checkpoint** save/load for trained models (architecture stored in file)
- **Python compat mode**: reproduce the original microgpt.py algorithm with the right flags
- No frameworks, no BLAS, no GPU — just `rand` crate and arithmetic

## Quick start

```bash
# Download a dataset (e.g. Karpathy's names dataset)
curl -sL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt

# Train with defaults (2L/48E/8H, 7000 steps, ~59K params)
cargo run --release

# Train and save checkpoint
cargo run --release -- --save model.mgpt

# Load checkpoint and generate (no training)
cargo run --release -- --load model.mgpt --samples 50

# Custom architecture
cargo run --release -- --layers 4 --embd 64 --heads 8 --steps 5000 --save big.mgpt
```

## Usage

```
Usage: microgpt [OPTIONS]

Model:
  --layers N      number of transformer layers [default: 2]
  --embd N        embedding dimension [default: 48]
  --heads N       number of attention heads [default: 8]
  --block N       context window size [default: 16]
  --activation S  activation function: silu, relu [default: silu]
  --no-final-norm skip final RMSNorm before lm_head
  --no-learnable-gamma  freeze RMSNorm gamma at 1.0

Training:
  --steps N       training steps [default: 7000]
  --warmup N      LR warmup steps [default: 20]
  --lr F          learning rate [default: 0.002]
  --beta1 F       Adam beta1 [default: 0.9]
  --wd F          weight decay, 0=pure Adam [default: 0.01]
  --grad-clip F   gradient clipping norm, 0=disabled [default: 1]
  --lr-schedule S lr schedule: cosine, linear [default: cosine]
  --seed N        random seed [default: random]

Inference:
  --temp F        sampling temperature [default: 0.5]
  --samples N     number of samples to generate [default: 20]

Data:
  --input PATH    dataset file [default: input.txt]

Checkpoint:
  --save PATH     save trained model to file
  --load PATH     load model and skip training (inference only)
```

## Example output

```
docs: 32033 | vocab: 27 | params: 58944 | layers: 2 | embd: 48 | heads: 8
step 7000 / 7000 | loss 2.0583 | avg100 2.1805
--- generated samples ---
sample  1: kais
sample  2: amiilee
sample  3: chelean
sample  4: laison
sample  5: tanali
sample  6: shari
sample  7: sheleon
sample  8: marac
sample  9: alini
sample 10: maria
```

## Architecture

| Component | Details |
|-----------|---------|
| Autograd | Tape-based (Wengert list), reverse-mode AD |
| Normalization | RMSNorm with optional learnable per-dimension gamma |
| Activation | SiLU (default) or ReLU, selectable via `--activation` |
| Loss | Fused log-softmax for numerical stability |
| Optimizer | AdamW with decoupled weight decay (`--wd 0` for pure Adam) |
| LR schedule | Cosine decay with warmup (default) or linear decay |
| Gradient clipping | Global norm (default max=1.0, `--grad-clip 0` to disable) |
| Sampling | Temperature-scaled softmax |
| Inference | Tape-free f64 forward pass (no autograd overhead) |

### Differences from microgpt.py

| | microgpt.py (original) | microgpt-rs (default) |
|---|---|---|
| Activation | ReLU | SiLU |
| Optimizer | Adam (beta1=0.85) | AdamW (beta1=0.9, wd=0.01) |
| LR schedule | Linear decay | Cosine with warmup |
| Gradient clipping | None | Global norm, max=1.0 |
| Final RMSNorm | Absent | Present |
| RMSNorm gamma | Not learnable | Learnable |
| Default config | 1L/16E/4H, 1000 steps | 2L/48E/8H, 7000 steps |

Reproduce the original microgpt.py algorithm (ReLU, pure Adam, linear LR decay, no gradient clipping, no learnable gamma, no final norm):

```bash
cargo run --release -- --layers 1 --embd 16 --heads 4 --block 16 \
    --steps 1000 --lr 0.01 --warmup 0 --wd 0 --seed 42 \
    --activation relu --beta1 0.85 --lr-schedule linear \
    --grad-clip 0 --no-final-norm --no-learnable-gamma
```

Note: Python and Rust use different PRNGs, so initial weights differ and loss trajectories won't be bit-identical. The algorithm and hyperparameters are identical.

## Dataset format

One entry per line, plain text. The model learns character-level patterns and generates new entries.

```
emma
olivia
ava
isabella
...
```

Works with any line-based text: names, words, short phrases.

## Performance

Training and inference are optimized without sacrificing correctness:

- **Tape-free inference**: `generate_sample()` uses pure f64 arithmetic — no tape, no autograd graph. Orders of magnitude faster than running forward pass through the tape.
- **Pre-allocated tape**: tape capacity is estimated from model config, eliminating reallocations during training.
- **Zero-alloc dot product**: `dot()` accumulates in-place instead of collecting into intermediate `Vec`.
- **Shared constants**: frequently used leaf nodes (`-1.0`, `1.0`, `eps`) are created once per step and reused across all operations.
- **One-time model index build**: `GptModel::build()` creates the parameter index template once; each training step only loads parameter values.

## Checkpoint format

Binary file (`.mgpt`): magic header + model config + architectural flags (activation, norm settings) + tokenizer vocabulary + raw f64 parameters. Full architecture is stored in the file, so `--load` doesn't need any model flags.
