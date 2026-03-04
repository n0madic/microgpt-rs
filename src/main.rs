//! microgpt-rs — train and run a GPT in pure Rust.
//!
//! This file contains only the CLI, argument parsing, data loading, and
//! the main training/inference orchestration loop.
//! The algorithmic core lives in `microgpt.rs`.

mod microgpt;
pub use microgpt::*;

use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::VecDeque;
use std::fs;
use std::io::{self, Write};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Section 8: Data Loading & Main
// ---------------------------------------------------------------------------

fn load_dataset(path: &str) -> Vec<String> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: cannot read '{path}': {e}");
            std::process::exit(1);
        }
    };
    let docs: Vec<String> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .collect();
    if docs.is_empty() {
        eprintln!("error: '{path}' contains no data");
        std::process::exit(1);
    }
    docs
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let raw: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--help" | "-h" => {
                eprintln!(
                    "Usage: microgpt [OPTIONS]\n\n\
                     Model:\n  \
                       --layers N      number of transformer layers [default: {}]\n  \
                       --embd N        embedding dimension [default: {}]\n  \
                       --heads N       number of attention heads [default: {}]\n  \
                       --block N       context window size [default: {}]\n  \
                       --activation S  activation function: silu, relu [default: relu]\n  \
                       --init-scale S  weight init: flat, scaled [default: flat]\n  \
                       --no-final-norm skip final RMSNorm before lm_head (default: on)\n  \
                       --final-norm    enable final RMSNorm before lm_head\n  \
                       --no-learnable-gamma  freeze RMSNorm gamma at 1.0 (default: on)\n  \
                       --learnable-gamma     enable learnable RMSNorm gamma\n\n\
                     Training:\n  \
                       --steps N       training steps [default: {}]\n  \
                       --warmup N      LR warmup steps [default: {}]\n  \
                       --lr F          learning rate [default: {}]\n  \
                       --beta1 F       Adam beta1 [default: {}]\n  \
                       --beta2 F       Adam beta2 [default: {}]\n  \
                       --wd F          weight decay, 0=pure Adam [default: {}]\n  \
                       --grad-clip F   gradient clipping norm, 0=disabled [default: {}]\n  \
                       --lr-schedule S lr schedule: cosine, linear [default: linear]\n  \
                       --batch N       gradient accumulation batch size [default: {}]\n  \
                       --dropout F     dropout probability, 0=disabled [default: {}]\n  \
                       --val-split F   validation split fraction, 0=disabled [default: {}]\n  \
                       --seed N        random seed [default: random]\n\n\
                     Inference:\n  \
                       --temp F        sampling temperature [default: {}]\n  \
                       --samples N     number of samples to generate [default: {}]\n\n\
                     Data:\n  \
                       --input PATH    dataset file [default: {}]\n\n\
                     Tokenizer:\n  \
                       --tokenizer S   tokenizer type: char, bpe [default: char]\n  \
                       --vocab-size N  BPE target vocabulary size [default: 256]\n\n\
                     Checkpoint:\n  \
                       --save PATH     save model checkpoint after training\n  \
                       --load PATH     load checkpoint and resume training (--steps 0 for inference only)",
                    args.n_layer,
                    args.n_embd,
                    args.n_head,
                    args.block_size,
                    args.steps,
                    args.warmup,
                    args.lr,
                    args.beta1,
                    args.beta2,
                    args.weight_decay,
                    args.grad_clip,
                    args.batch_size,
                    args.dropout,
                    args.val_split,
                    args.temperature,
                    args.samples,
                    args.input,
                );
                std::process::exit(0);
            }
            "--layers" => {
                i += 1;
                args.n_layer = parse_val(&raw, i, "--layers");
            }
            "--embd" => {
                i += 1;
                args.n_embd = parse_val(&raw, i, "--embd");
            }
            "--heads" => {
                i += 1;
                args.n_head = parse_val(&raw, i, "--heads");
            }
            "--block" => {
                i += 1;
                args.block_size = parse_val(&raw, i, "--block");
            }
            "--steps" => {
                i += 1;
                args.steps = parse_val(&raw, i, "--steps");
            }
            "--warmup" => {
                i += 1;
                args.warmup = parse_val(&raw, i, "--warmup");
            }
            "--lr" => {
                i += 1;
                args.lr = parse_val(&raw, i, "--lr");
            }
            "--wd" => {
                i += 1;
                args.weight_decay = parse_val(&raw, i, "--wd");
            }
            "--beta1" => {
                i += 1;
                args.beta1 = parse_val(&raw, i, "--beta1");
            }
            "--grad-clip" => {
                i += 1;
                args.grad_clip = parse_val(&raw, i, "--grad-clip");
            }
            "--lr-schedule" => {
                i += 1;
                let s = parse_str(&raw, i, "--lr-schedule");
                args.lr_schedule = match s.as_str() {
                    "cosine" => LrSchedule::Cosine,
                    "linear" => LrSchedule::Linear,
                    _ => {
                        eprintln!("error: --lr-schedule must be 'cosine' or 'linear', got '{s}'");
                        std::process::exit(1);
                    }
                };
            }
            "--activation" => {
                i += 1;
                let s = parse_str(&raw, i, "--activation");
                args.activation = match s.as_str() {
                    "silu" => Activation::Silu,
                    "relu" => Activation::Relu,
                    _ => {
                        eprintln!("error: --activation must be 'silu' or 'relu', got '{s}'");
                        std::process::exit(1);
                    }
                };
            }
            "--no-final-norm" => {
                args.no_final_norm = true;
            }
            "--final-norm" => {
                args.no_final_norm = false;
            }
            "--no-learnable-gamma" => {
                args.no_learnable_gamma = true;
            }
            "--learnable-gamma" => {
                args.no_learnable_gamma = false;
            }
            "--beta2" => {
                i += 1;
                args.beta2 = parse_val(&raw, i, "--beta2");
            }
            "--init-scale" => {
                i += 1;
                let s = parse_str(&raw, i, "--init-scale");
                args.init_scale = match s.as_str() {
                    "flat" => InitScale::Flat,
                    "scaled" => InitScale::Scaled,
                    _ => {
                        eprintln!("error: --init-scale must be 'flat' or 'scaled', got '{s}'");
                        std::process::exit(1);
                    }
                };
            }
            "--dropout" => {
                i += 1;
                args.dropout = parse_val(&raw, i, "--dropout");
            }
            "--batch" => {
                i += 1;
                args.batch_size = parse_val(&raw, i, "--batch");
            }
            "--val-split" => {
                i += 1;
                args.val_split = parse_val(&raw, i, "--val-split");
            }
            "--seed" => {
                i += 1;
                args.seed = Some(parse_val(&raw, i, "--seed"));
            }
            "--temp" => {
                i += 1;
                args.temperature = parse_val(&raw, i, "--temp");
            }
            "--samples" => {
                i += 1;
                args.samples = parse_val(&raw, i, "--samples");
            }
            "--input" => {
                i += 1;
                args.input = parse_str(&raw, i, "--input");
            }
            "--save" => {
                i += 1;
                args.save = Some(parse_str(&raw, i, "--save"));
            }
            "--load" => {
                i += 1;
                args.load = Some(parse_str(&raw, i, "--load"));
            }
            "--tokenizer" => {
                i += 1;
                let s = parse_str(&raw, i, "--tokenizer");
                args.tokenizer_type = match s.as_str() {
                    "char" => TokenizerType::Char,
                    "bpe" => TokenizerType::Bpe,
                    _ => {
                        eprintln!("error: --tokenizer must be 'char' or 'bpe', got '{s}'");
                        std::process::exit(1);
                    }
                };
            }
            "--vocab-size" => {
                i += 1;
                args.bpe_vocab_size = parse_val(&raw, i, "--vocab-size");
            }
            other => {
                eprintln!("unknown option: {other}\nrun with --help for usage");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    validate_args(&mut args);
    args
}

fn parse_val<T: std::str::FromStr>(raw: &[String], i: usize, flag: &str) -> T {
    raw.get(i)
        .unwrap_or_else(|| {
            eprintln!("{flag} requires a value");
            std::process::exit(1);
        })
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("{flag}: invalid value '{}'", raw[i]);
            std::process::exit(1);
        })
}

fn parse_str(raw: &[String], i: usize, flag: &str) -> String {
    raw.get(i)
        .unwrap_or_else(|| {
            eprintln!("{flag} requires a value");
            std::process::exit(1);
        })
        .clone()
}

fn validate_args(args: &mut Args) {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // --- hard errors: invalid values ---
    if args.n_layer == 0 {
        errors.push("--layers must be >= 1".into());
    }
    if args.n_embd < 2 {
        errors.push("--embd must be >= 2".into());
    }
    if args.n_head == 0 {
        errors.push("--heads must be >= 1".into());
    }
    if args.block_size < 2 {
        errors
            .push("--block must be >= 2 (need at least 2 tokens for next-token prediction)".into());
    }
    if args.steps == 0 && args.load.is_none() {
        errors
            .push("--steps must be >= 1 (or use --load with --steps 0 for inference only)".into());
    }
    if args.lr <= 0.0 {
        errors.push("--lr must be positive".into());
    }
    if args.weight_decay < 0.0 {
        errors.push("--wd must be >= 0".into());
    }
    if args.beta1 <= 0.0 || args.beta1 >= 1.0 {
        errors.push("--beta1 must be in (0, 1)".into());
    }
    if args.beta2 <= 0.0 || args.beta2 >= 1.0 {
        errors.push("--beta2 must be in (0, 1)".into());
    }
    if args.grad_clip < 0.0 {
        errors.push("--grad-clip must be >= 0".into());
    }
    if args.temperature <= 0.0 {
        errors.push("--temp must be positive".into());
    }
    if args.tokenizer_type == TokenizerType::Bpe && args.bpe_vocab_size < 2 {
        errors.push("--vocab-size must be >= 2 for BPE tokenizer".into());
    }
    if args.dropout < 0.0 || args.dropout >= 1.0 {
        errors.push("--dropout must be in [0, 1)".into());
    }
    if args.batch_size == 0 {
        errors.push("--batch must be >= 1".into());
    }
    if args.val_split < 0.0 || args.val_split >= 1.0 {
        errors.push("--val-split must be in [0, 1)".into());
    }

    // --- hard errors: conflicting values ---
    if args.n_embd > 0 && args.n_head > 0 && !args.n_embd.is_multiple_of(args.n_head) {
        // try to auto-fix: find nearest valid n_head
        let orig = args.n_head;
        let candidates: Vec<usize> = (1..=args.n_embd)
            .filter(|h| args.n_embd.is_multiple_of(*h))
            .collect();
        let best = candidates
            .iter()
            .min_by_key(|&&h| (h as isize - orig as isize).unsigned_abs())
            .unwrap();
        warnings.push(format!(
            "--embd {} not divisible by --heads {}, adjusted heads to {}",
            args.n_embd, orig, best
        ));
        args.n_head = *best;
    }
    if args.n_embd > 0 && args.n_head > 0 && args.n_embd / args.n_head < 2 {
        errors.push(format!(
            "head_dim = embd/heads = {}/{} = {} is too small, need >= 2",
            args.n_embd,
            args.n_head,
            args.n_embd / args.n_head
        ));
    }

    // --- warnings: suboptimal but allowed ---
    if args.steps > 0 && args.steps <= args.warmup {
        warnings.push(format!(
            "--steps {} <= warmup ({}): model won't reach full learning rate",
            args.steps, args.warmup
        ));
    }
    if args.lr > 0.05 {
        warnings.push(format!(
            "--lr {} is high, may cause instability (try <= 0.05)",
            args.lr
        ));
    }
    if args.lr_schedule == LrSchedule::Linear && args.warmup > 0 {
        warnings.push("--lr-schedule linear ignores --warmup (linear decay has no warmup)".into());
    }
    if args.weight_decay >= 0.1 {
        warnings.push(format!(
            "--wd {} is high, may prevent learning (try 0.001-0.05)",
            args.weight_decay
        ));
    }
    if args.temperature > 1.5 {
        warnings.push(format!(
            "--temp {} is very high, output will be near-random",
            args.temperature
        ));
    }
    if args.n_layer > 1 && args.n_embd < 16 {
        warnings.push(format!(
            "--layers {} with --embd {}: deep+narrow is suboptimal, increase embd",
            args.n_layer, args.n_embd
        ));
    }
    if args.n_layer > 4 {
        warnings.push(format!(
            "--layers {}: tape-based autograd gets slow with many layers, expect long training",
            args.n_layer
        ));
    }

    if !errors.is_empty() {
        for e in &errors {
            eprintln!("error: {e}");
        }
        std::process::exit(1);
    }
    for w in &warnings {
        eprintln!("warning: {w}");
    }
}

#[derive(Debug)]
struct Args {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    block_size: usize,
    steps: usize,
    warmup: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    weight_decay: f64,
    grad_clip: f64,
    lr_schedule: LrSchedule,
    activation: Activation,
    no_final_norm: bool,
    no_learnable_gamma: bool,
    seed: Option<u64>,
    temperature: f64,
    samples: usize,
    input: String,
    save: Option<String>,
    load: Option<String>,
    tokenizer_type: TokenizerType,
    bpe_vocab_size: usize,
    init_scale: InitScale,
    dropout: f64,
    batch_size: usize,
    val_split: f64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            block_size: 16,
            steps: 1000,
            warmup: 0,
            lr: 0.01,
            beta1: 0.85,
            beta2: 0.99,
            weight_decay: 0.0,
            grad_clip: 0.0,
            lr_schedule: LrSchedule::Linear,
            activation: Activation::Relu,
            no_final_norm: true,
            no_learnable_gamma: true,
            seed: None,
            temperature: 0.5,
            samples: 20,
            input: "input.txt".to_string(),
            save: None,
            load: None,
            tokenizer_type: TokenizerType::Char,
            bpe_vocab_size: 256,
            init_scale: InitScale::Flat,
            dropout: 0.0,
            batch_size: 1,
            val_split: 0.0,
        }
    }
}

fn fmt_duration(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

fn main() {
    let args = parse_args();
    let seed = args.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    let mut rng = StdRng::seed_from_u64(seed);

    // Determine model source: checkpoint or fresh build
    let (config, tokenizer, mut params, step_offset, adam, batch_size, docs) =
        if let Some(ref path) = args.load {
            let ckpt = load_checkpoint(path);
            // Load dataset only if we'll continue training
            let docs = if args.steps > 0 {
                let mut d = load_dataset(&args.input);
                shuffle(&mut d, &mut rng);
                d
            } else {
                Vec::new()
            };
            (
                ckpt.config,
                ckpt.tokenizer,
                ckpt.params,
                ckpt.completed_step,
                ckpt.adam,
                ckpt.batch_size,
                docs,
            )
        } else {
            let mut docs = load_dataset(&args.input);
            shuffle(&mut docs, &mut rng);

            let tokenizer = match args.tokenizer_type {
                TokenizerType::Char => Tokenizer::from_docs_char(&docs),
                TokenizerType::Bpe => Tokenizer::from_docs_bpe(&docs, args.bpe_vocab_size),
            };
            let config = GptConfig {
                n_layer: args.n_layer,
                n_embd: args.n_embd,
                block_size: args.block_size,
                n_head: args.n_head,
                head_dim: args.n_embd / args.n_head,
                vocab_size: tokenizer.vocab_size(),
                activation: args.activation,
                no_final_norm: args.no_final_norm,
                no_learnable_gamma: args.no_learnable_gamma,
                dropout: args.dropout,
            };
            let params = Params::new(&config, &mut rng, args.init_scale);
            let tok_label = match args.tokenizer_type {
                TokenizerType::Char => "char",
                TokenizerType::Bpe => "bpe",
            };
            println!(
                "docs: {} | vocab: {} ({}) | params: {} | layers: {} | embd: {} | heads: {}",
                docs.len(),
                tokenizer.vocab_size(),
                tok_label,
                params.len(),
                config.n_layer,
                config.n_embd,
                config.n_head
            );

            let adam = AdamConfig {
                lr: args.lr,
                beta1: args.beta1,
                beta2: args.beta2,
                eps: 1e-8,
                weight_decay: args.weight_decay,
                warmup_steps: args.warmup,
                schedule: args.lr_schedule,
                grad_clip: args.grad_clip,
            };
            (config, tokenizer, params, 0, adam, args.batch_size, docs)
        };

    // Training (skipped when --steps 0)
    if args.steps > 0 {
        let mut docs = docs;

        // Split into train/val
        let val_count = (docs.len() as f64 * args.val_split) as usize;
        let val_docs: Vec<String> = docs.drain(docs.len() - val_count..).collect();
        let train_docs = docs;

        let model = GptModel::build(&config);
        let nt = n_trainable(&config, params.len());
        let mut grads_buf = vec![0.0; nt];
        let mut recent_losses = VecDeque::with_capacity(101);
        let train_start = Instant::now();
        let mut last_print = Instant::now() - Duration::from_secs(2);
        let mut doc_idx = 0usize;
        let train_len = train_docs.len();
        let mut train_order: Vec<usize> = (0..train_len).collect();
        let mut docs_seen = 0usize;
        let mut epoch = 0usize;

        // Total steps for LR schedule: previous + new
        let total_steps = step_offset + args.steps;

        for i in 0..args.steps {
            let step = step_offset + i;

            for g in grads_buf.iter_mut() {
                *g = 0.0;
            }

            let mut batch_loss = 0.0;
            for _ in 0..batch_size {
                if docs_seen > 0 && docs_seen.is_multiple_of(train_len) {
                    epoch += 1;
                    shuffle(&mut train_order, &mut rng);
                    if !val_docs.is_empty() {
                        let vl = compute_val_loss(&params, &config, &tokenizer, &val_docs);
                        println!("\nepoch {} | val_loss {:.4}", epoch, vl);
                    }
                }
                let di = train_order[doc_idx % train_len];
                doc_idx += 1;
                docs_seen += 1;
                let doc = &train_docs[di];
                let tokens = tokenizer.encode(doc);
                batch_loss +=
                    forward_backward(&params, &config, &model, &tokens, &mut grads_buf, &mut rng);
            }

            let inv_batch = 1.0 / batch_size as f64;
            for g in grads_buf.iter_mut() {
                *g *= inv_batch;
            }
            batch_loss *= inv_batch;

            adamw_step(&mut params, &config, &adam, &grads_buf, step, total_steps);

            recent_losses.push_back(batch_loss);
            if recent_losses.len() > 100 {
                recent_losses.pop_front();
            }
            let avg: f64 = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
            let now = Instant::now();
            if now.duration_since(last_print) >= Duration::from_secs(1) || i + 1 == args.steps {
                last_print = now;
                let elapsed = train_start.elapsed().as_secs_f64();
                let steps_done = i + 1;
                let eta_str = if steps_done >= 10 {
                    let secs_per_step = elapsed / steps_done as f64;
                    let remaining = ((args.steps - steps_done) as f64 * secs_per_step) as u64;
                    format!(" | eta {}", fmt_duration(remaining))
                } else {
                    String::new()
                };
                print!(
                    "\rstep {:4} / {:4} | loss {:.4} | avg100 {:.4}{}",
                    step + 1,
                    total_steps,
                    batch_loss,
                    avg,
                    eta_str
                );
                let _ = io::stdout().flush();
            }
        }
        println!();

        if !val_docs.is_empty() {
            let vl = compute_val_loss(&params, &config, &tokenizer, &val_docs);
            println!("final val_loss {:.4}", vl);
        }

        if let Some(ref path) = args.save {
            save_checkpoint(
                path,
                &config,
                &adam,
                &tokenizer,
                &params,
                step_offset + args.steps,
                batch_size,
            );
        }
    }

    // Inference
    println!("--- generated samples ---");
    for _ in 0..args.samples {
        let name = generate_sample(&params, &config, &tokenizer, args.temperature, &mut rng);
        println!("{}", name);
    }
}

// ---------------------------------------------------------------------------
// Section 10: Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
