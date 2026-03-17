# microgpt-rs

The most atomic way to train and run inference for a GPT — in pure Rust.

Inspired by [Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). Same educational spirit, idiomatic Rust. Tape-based autograd instead of pointer graph.

## What's inside

- **Tape-based autograd** (Wengert list) with reverse-mode differentiation
- **Transformer**: multi-head attention, RMSNorm (optionally with learnable gamma), SiLU/ReLU/SwiGLU activation, optional dropout
- **AdamW** optimizer with cosine or linear LR schedule, warmup, gradient clipping, and gradient accumulation
- **Tokenizer**: char-level (default) or BPE with configurable vocabulary size
- **Checkpoint** save/load for trained models (architecture stored in file)
- **Train/validation split** with tape-free validation loss
- **Epoch-based data re-shuffling** for better generalization
- **OOV-safe tokenizer**: unknown characters are silently skipped during encoding
- No frameworks, no BLAS, no GPU — just `rand` and `serde` crates

## Quick start

```bash
# Download a dataset (e.g. Karpathy's names dataset)
curl -sL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt

# Train with defaults (microgpt.py compatible: 1L/16E/4H, 1000 steps, ~2K params)
cargo run --release

# Train and save checkpoint
cargo run --release -- --save model.mgpt

# Load checkpoint and generate (no training)
cargo run --release -- --load model.mgpt --steps 0 --samples 50

# Resume training from checkpoint
cargo run --release -- --load model.mgpt --steps 500

# Custom architecture
cargo run --release -- --layers 4 --embd 64 --heads 8 --steps 5000 --save big.mgpt

# Train with BPE tokenizer (subword-level)
cargo run --release -- --tokenizer bpe --vocab-size 64 --steps 1000 --save bpe.mgpt
```

## Usage

```
Usage: microgpt [OPTIONS]

Model:
  --layers N      number of transformer layers [default: 1]
  --embd N        embedding dimension [default: 16]
  --heads N       number of attention heads [default: 4]
  --block N       context window size [default: 16]
  --activation S  activation function: silu, relu, swiglu [default: relu]
  --init-scale S  weight init: flat, scaled [default: flat]
  --no-final-norm skip final RMSNorm before lm_head (default: on)
  --final-norm    enable final RMSNorm before lm_head
  --no-learnable-gamma  freeze RMSNorm gamma at 1.0 (default: on)
  --learnable-gamma     enable learnable RMSNorm gamma

Training:
  --steps N       training steps [default: 1000]
  --warmup N      LR warmup steps [default: 0]
  --lr F          learning rate [default: 0.01]
  --beta1 F       Adam beta1 [default: 0.85]
  --beta2 F       Adam beta2 [default: 0.99]
  --wd F          weight decay, 0=pure Adam [default: 0]
  --grad-clip F   gradient clipping norm, 0=disabled [default: 0]
  --lr-schedule S lr schedule: cosine, linear [default: linear]
  --batch N       gradient accumulation batch size [default: 1]
  --dropout F     dropout probability, 0=disabled [default: 0]
  --val-split F   validation split fraction, 0=disabled [default: 0]
  --seed N        random seed [default: random]

Inference:
  --temp F        sampling temperature [default: 0.5]
  --samples N     number of samples to generate [default: 20]

Data:
  --input PATH    dataset file [default: input.txt]

Tokenizer:
  --tokenizer S   tokenizer type: char, bpe [default: char]
  --vocab-size N  BPE target vocabulary size [default: 256]

Checkpoint:
  --save PATH     save model checkpoint after training
  --load PATH     load checkpoint and resume training (--steps 0 for inference only)
```

## Recommended settings for better training

```bash
cargo run --release -- --layers 2 --embd 48 --heads 8 --steps 7000 \
    --lr 0.002 --beta1 0.9 --wd 0.01 --grad-clip 1.0 \
    --activation silu --lr-schedule cosine --warmup 20 \
    --init-scale scaled --dropout 0.1 --batch 4 \
    --final-norm --learnable-gamma
```

Why each flag improves over the default:

| Flag | Default | Recommended | Why it helps |
|------|---------|-------------|--------------|
| `--layers 2 --embd 48 --heads 8` | 1L/16E/4H (~2K params) | 2L/48E/8H (~59K params) | Larger model has more capacity to learn complex patterns; 2 layers allow compositional features |
| `--steps 7000` | 1000 | 7000 | More training steps let the model converge further, especially with a larger architecture |
| `--lr 0.002` | 0.01 | 0.002 | Lower LR gives smoother, more stable optimization — less risk of overshooting minima |
| `--beta1 0.9` | 0.85 | 0.9 | Higher momentum smooths out noisy gradients, standard value for Adam across most tasks |
| `--wd 0.01` | 0 (pure Adam) | 0.01 (AdamW) | Weight decay regularizes by penalizing large weights, reducing overfitting |
| `--grad-clip 1.0` | 0 (disabled) | 1.0 | Prevents gradient explosions that can destabilize training, especially with deeper models |
| `--activation silu` | relu | silu | SiLU (Swish) is smooth and non-monotonic — avoids dead neurons and generally trains better than ReLU |
| `--activation swiglu` | relu | swiglu | SwiGLU gated MLP (as in LLaMA) — adds a gate projection `SiLU(xW_gate) ⊙ xW_up` before the down projection. More params per layer but better quality |
| `--lr-schedule cosine` | linear | cosine | Cosine annealing decays slowly at first, then quickly — more time at productive learning rates |
| `--warmup 20` | 0 | 20 | Gradual LR ramp-up prevents large early updates when Adam statistics are not yet calibrated |
| `--init-scale scaled` | flat (0.08 std) | scaled (GPT-2 style) | Per-layer scaling (1/√depth for residual projections) prevents signal explosion in deeper models |
| `--dropout 0.1` | 0 (disabled) | 0.1 | Randomly drops 10% of activations during training — forces redundant representations, reduces overfitting |
| `--batch 4` | 1 | 4 | Averaging gradients over 4 documents reduces noise, giving a more reliable optimization signal per step |
| `--final-norm` | off | on | Extra RMSNorm before the output projection stabilizes logit scale, especially helpful with multiple layers |
| `--learnable-gamma` | off | on | Lets each RMSNorm learn a per-dimension scaling factor — more expressive normalization at negligible parameter cost |

### Scaling up: larger models and datasets

When working with bigger datasets or deeper architectures, a few additional flags become important:

```bash
# Larger model with overfitting protection and progress tracking
cargo run --release -- --layers 4 --embd 64 --heads 8 --block 64 --steps 10000 \
    --lr 0.001 --beta1 0.9 --wd 0.02 --grad-clip 1.0 \
    --activation silu --lr-schedule cosine --warmup 100 \
    --init-scale scaled --dropout 0.2 --batch 8 \
    --final-norm --learnable-gamma \
    --tokenizer bpe --vocab-size 128 \
    --val-split 0.1 --save big.mgpt
```

Key considerations when scaling:

- **`--val-split 0.1`** — hold out 10% of data for validation. Validation loss is printed at every epoch boundary, letting you detect overfitting early (train loss drops but val loss starts rising). Without this, you're flying blind.
- **`--dropout 0.2`** — increase dropout for deeper models. More layers = more parameters = more risk of memorizing the training data. 0.1 is fine for 2L, bump to 0.2–0.3 for 4L+.
- **`--init-scale scaled`** — becomes critical with depth. Flat init with 4+ layers often leads to exploding activations in early training. Scaled init keeps signal magnitude stable across layers.
- **`--batch 8`** — larger batches smooth gradients further, especially useful with big noisy datasets. Trade-off: each step takes N× longer, but converges in fewer steps.
- **`--lr 0.001`** — lower the learning rate as model size grows. Larger models are more sensitive to large weight updates.
- **`--warmup 100`** — more warmup steps for larger models. Adam's second moment estimates need more samples to stabilize when there are more parameters.
- **`--wd 0.02`** — slightly stronger weight decay for bigger models helps prevent weights from growing unbounded.
- **`--steps`** — scale roughly with dataset size. A good heuristic: at least 3–5 epochs (steps ≥ 3–5 × dataset_size / batch_size).
- **`--tokenizer bpe --vocab-size 128`** — for datasets with longer entries (sentences, phrases, code snippets), BPE compresses sequences into fewer tokens, letting the model see more content within the fixed `--block` context window. For short entries like names, char-level is fine — BPE has little to merge and adds vocabulary overhead that a small model can't utilize well. Rule of thumb: if your average entry is longer than `--block` characters, switch to BPE.
- **`--block 64`** — context window size, fixed at training time and baked into the checkpoint (positional embeddings are `block_size × n_embd`). The model can only see and generate sequences up to this length. Default 16 is enough for short names, but for longer text you need more. Trade-offs: attention cost is O(block²) per position — doubling block roughly quadruples memory and time per training step. Combine with BPE to stretch the effective context: BPE compresses text so the same `--block` covers more characters. Practical guideline: set `--block` ≥ median entry length in tokens (check `docs` count vs raw line count to estimate compression).

## Example output

```
docs: 32033 | vocab: 27 (char) | params: 4256 | layers: 1 | embd: 16 | heads: 4
step 1000 / 1000 | loss 2.1570 | avg100 2.3604 | eta 0s
--- generated samples ---
karay
salinen
aranile
doren
kahar
```

## Architecture

| Component | Details |
|-----------|---------|
| Autograd | Tape-based (Wengert list), reverse-mode AD |
| Normalization | RMSNorm with optional learnable per-dimension gamma |
| Activation | ReLU (default), SiLU, or SwiGLU, selectable via `--activation` |
| Dropout | Training-only, applied after attention softmax, Wo, and fc2 |
| Loss | Fused log-softmax for numerical stability |
| Optimizer | AdamW with decoupled weight decay (`--wd 0` for pure Adam) |
| LR schedule | Linear decay (default) or cosine decay with warmup |
| Gradient clipping | Global norm (`--grad-clip 0` to disable, default) |
| Gradient accumulation | `--batch N` averages gradients over N documents |
| Weight init | Flat (0.08 std) or scaled (GPT-2 style per-layer scaling) |
| Validation | Tape-free f64 forward pass, loss printed at epoch boundaries |
| Sampling | Temperature-scaled softmax |
| Inference | Tape-free f64 forward pass (no autograd overhead) |

Defaults match the original microgpt.py exactly (1L/16E/4H, ReLU, pure Adam, linear LR, no gradient clipping, no final norm, no learnable gamma). All improvements are opt-in via CLI flags.

## Dataset format

One entry per line, plain text. With `--tokenizer char` (default) the model learns character-level patterns. With `--tokenizer bpe` it learns subword patterns using Byte Pair Encoding, which produces shorter token sequences for the same text.

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
- **BPE swap-buffer**: BPE encode reuses a single buffer across merge iterations instead of allocating per merge.

## Checkpoint format

Binary file (`.mgpt`) using MessagePack serialization: magic header (`MGPT`) + model config + optimizer config (AdamW hyperparameters) + tokenizer (char vocabulary or BPE vocabulary with merge rules) + model parameters + optimizer state (Adam m/v vectors) + training progress (completed step, batch size). Full architecture and training state are stored in the file, so `--load` doesn't need any model flags and can resume training seamlessly.
