use super::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

// --- (a) Autograd numerical gradient check ---

fn numerical_grad(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

#[test]
fn test_autograd_add() {
    let mut tape = Tape::with_capacity(16);
    let a = tape.leaf(2.0);
    let b = tape.leaf(3.0);
    let c = tape.add(a, b);
    tape.backward(c);
    assert!((tape.grad(a) - 1.0).abs() < 1e-10);
    assert!((tape.grad(b) - 1.0).abs() < 1e-10);
}

#[test]
fn test_autograd_mul() {
    let mut tape = Tape::with_capacity(16);
    let a = tape.leaf(2.0);
    let b = tape.leaf(3.0);
    let c = tape.mul(a, b);
    tape.backward(c);
    assert!((tape.grad(a) - 3.0).abs() < 1e-10);
    assert!((tape.grad(b) - 2.0).abs() < 1e-10);
}

#[test]
fn test_autograd_complex_expr() {
    // (a * b + c).exp().log() — tests Add, Mul, Exp, Log
    let a_val = 1.5;
    let b_val = 2.0;
    let c_val = -0.5;
    let h = 1e-6;

    let mut tape = Tape::with_capacity(32);
    let a = tape.leaf(a_val);
    let b = tape.leaf(b_val);
    let c = tape.leaf(c_val);
    let ab = tape.mul(a, b);
    let abc = tape.add(ab, c);
    let e = tape.exp(abc);
    let result = tape.log(e);
    tape.backward(result);

    // Numerical gradients
    let f_a = |x: f64| (x * b_val + c_val).exp().ln();
    let f_b = |x: f64| (a_val * x + c_val).exp().ln();
    let f_c = |x: f64| (a_val * b_val + x).exp().ln();

    let num_a = numerical_grad(f_a, a_val, h);
    let num_b = numerical_grad(f_b, b_val, h);
    let num_c = numerical_grad(f_c, c_val, h);

    assert!((tape.grad(a) - num_a).abs() / num_a.abs().max(1e-10) < 1e-5);
    assert!((tape.grad(b) - num_b).abs() / num_b.abs().max(1e-10) < 1e-5);
    assert!((tape.grad(c) - num_c).abs() / num_c.abs().max(1e-10) < 1e-5);
}

#[test]
fn test_autograd_pow() {
    let a_val = 2.5;
    let exp = 3.0;
    let h = 1e-6;

    let mut tape = Tape::with_capacity(16);
    let a = tape.leaf(a_val);
    let result = tape.pow(a, exp);
    tape.backward(result);

    let num = numerical_grad(|x| x.powf(exp), a_val, h);
    assert!((tape.grad(a) - num).abs() / num.abs().max(1e-10) < 1e-5);
}

// --- (b) log_softmax vs softmax + log ---

#[test]
fn test_log_softmax_vs_softmax_log() {
    let mut tape = Tape::with_capacity(256);
    let c = TapeConstants::new(&mut tape, 4);

    let vals = [1.0, 2.0, 3.0, 0.5];
    let logits: Vec<Idx> = vals.iter().map(|&v| tape.leaf(v)).collect();

    let log_sm = log_softmax(&mut tape, &logits, &c);
    let sm = softmax(&mut tape, &logits, &c);

    for i in 0..vals.len() {
        let log_sm_val = tape.val(log_sm[i]);
        let sm_log_val = tape.val(sm[i]).ln();
        assert!(
            (log_sm_val - sm_log_val).abs() < 1e-10,
            "mismatch at {i}: log_softmax={log_sm_val} vs log(softmax)={sm_log_val}"
        );
    }
}

// --- (c) Checkpoint round-trip ---

#[test]
fn test_checkpoint_roundtrip() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abc".to_string(), "bca".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Silu,
        no_final_norm: false,
        no_learnable_gamma: false,
        dropout: 0.0,
    };
    let params = Params::new(&config, &mut rng, InitScale::Flat);

    let dir = std::env::temp_dir();
    let path = dir.join("microgpt_test_checkpoint.mgpt");
    let path_str = path.to_str().unwrap();

    let adam = AdamConfig {
        lr: 0.01,
        beta1: 0.85,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        warmup_steps: 0,
        schedule: LrSchedule::Linear,
        grad_clip: 0.0,
    };

    save_checkpoint(path_str, &config, &adam, &tokenizer, &params, 42, 1);
    let ckpt = load_checkpoint(path_str);

    // Config
    assert_eq!(config.n_layer, ckpt.config.n_layer);
    assert_eq!(config.n_embd, ckpt.config.n_embd);
    assert_eq!(config.n_head, ckpt.config.n_head);
    assert_eq!(config.block_size, ckpt.config.block_size);
    assert_eq!(config.head_dim, ckpt.config.head_dim);
    assert_eq!(config.vocab_size, ckpt.config.vocab_size);

    // Tokenizer
    assert_eq!(tokenizer.vocab_size(), ckpt.tokenizer.vocab_size());
    assert_eq!(tokenizer.bos(), ckpt.tokenizer.bos());

    // Params
    assert_eq!(params.data.len(), ckpt.params.data.len());
    for (&a, &b) in params.data.iter().zip(ckpt.params.data.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "param mismatch: {a} vs {b}");
    }

    // Optimizer state (m/v) round-trip
    for (&a, &b) in params.m.iter().zip(ckpt.params.m.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "m mismatch");
    }
    for (&a, &b) in params.v.iter().zip(ckpt.params.v.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "v mismatch");
    }

    // Training state
    assert_eq!(ckpt.completed_step, 42);
    assert_eq!(ckpt.batch_size, 1);
    assert_eq!(ckpt.adam.lr.to_bits(), adam.lr.to_bits());
    assert_eq!(ckpt.adam.beta1.to_bits(), adam.beta1.to_bits());
    assert_eq!(ckpt.adam.beta2.to_bits(), adam.beta2.to_bits());

    // Cleanup
    let _ = std::fs::remove_file(path_str);
}

// --- (d) Tokenizer encode/decode ---

#[test]
fn test_tokenizer_encode_decode() {
    let docs = vec!["abc".to_string(), "def".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);

    let tokens = tokenizer.encode("abc");
    // Should be [BOS, a_id, b_id, c_id, BOS]
    assert_eq!(tokens.len(), 5);
    assert_eq!(tokens[0], tokenizer.bos());
    assert_eq!(tokens[4], tokenizer.bos());

    // Decode non-BOS tokens back
    assert_eq!(tokenizer.decode(tokens[1]), Some("a".to_string()));
    assert_eq!(tokenizer.decode(tokens[2]), Some("b".to_string()));
    assert_eq!(tokenizer.decode(tokens[3]), Some("c".to_string()));

    // BOS decodes to None
    assert_eq!(tokenizer.decode(tokenizer.bos()), None);
}

// --- (d2) BPE tokenizer ---

#[test]
fn test_bpe_encode_decode_roundtrip() {
    let docs = vec!["aaabdaaabac".to_string(), "aabdaab".to_string()];
    let tokenizer = Tokenizer::from_docs_bpe(&docs, 10);

    assert!(tokenizer.vocab_size() <= 10);
    assert!(tokenizer.vocab_size() > 0);

    // Roundtrip: encode then decode should reconstruct original
    for doc in &docs {
        let tokens = tokenizer.encode(doc);
        assert_eq!(tokens[0], tokenizer.bos());
        assert_eq!(tokens[tokens.len() - 1], tokenizer.bos());
        let decoded: String = tokens[1..tokens.len() - 1]
            .iter()
            .filter_map(|&t| tokenizer.decode(t))
            .collect();
        assert_eq!(&decoded, doc);
    }

    assert_eq!(tokenizer.decode(tokenizer.bos()), None);
}

#[test]
fn test_bpe_merges_reduce_token_count() {
    let docs = vec!["aaaa".to_string()];

    // Char-level: "aaaa" -> [BOS, a, a, a, a, BOS] = 6 tokens
    let char_tok = Tokenizer::from_docs_char(&docs);
    let char_tokens = char_tok.encode("aaaa");
    assert_eq!(char_tokens.len(), 6);

    // BPE with enough vocab: should merge "aa" and produce fewer tokens
    let bpe_tok = Tokenizer::from_docs_bpe(&docs, 5);
    let bpe_tokens = bpe_tok.encode("aaaa");
    assert!(bpe_tokens.len() < char_tokens.len());
}

#[test]
fn test_bpe_checkpoint_roundtrip() {
    let docs = vec!["hello".to_string(), "help".to_string()];
    let tokenizer = Tokenizer::from_docs_bpe(&docs, 15);

    let mut rng = StdRng::seed_from_u64(42);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Silu,
        no_final_norm: false,
        no_learnable_gamma: false,
        dropout: 0.0,
    };
    let params = Params::new(&config, &mut rng, InitScale::Flat);

    let dir = std::env::temp_dir();
    let path = dir.join("microgpt_test_bpe_checkpoint.mgpt");
    let path_str = path.to_str().unwrap();

    let adam = AdamConfig {
        lr: 0.01,
        beta1: 0.85,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        warmup_steps: 0,
        schedule: LrSchedule::Linear,
        grad_clip: 0.0,
    };

    save_checkpoint(path_str, &config, &adam, &tokenizer, &params, 0, 1);
    let ckpt = load_checkpoint(path_str);

    assert_eq!(config.n_layer, ckpt.config.n_layer);
    assert_eq!(config.vocab_size, ckpt.config.vocab_size);
    assert_eq!(tokenizer.vocab_size(), ckpt.tokenizer.vocab_size());
    assert_eq!(tokenizer.bos(), ckpt.tokenizer.bos());

    // Encode roundtrip with loaded tokenizer
    let tokens1 = tokenizer.encode("hello");
    let tokens2 = ckpt.tokenizer.encode("hello");
    assert_eq!(tokens1, tokens2);

    assert_eq!(params.data.len(), ckpt.params.data.len());
    for (&a, &b) in params.data.iter().zip(ckpt.params.data.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let _ = std::fs::remove_file(path_str);
}

#[test]
fn test_char_tokenizer_decode_returns_string() {
    let docs = vec!["abc".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let tokens = tokenizer.encode("abc");
    assert_eq!(tokenizer.decode(tokens[1]), Some("a".to_string()));
    assert_eq!(tokenizer.decode(tokens[2]), Some("b".to_string()));
    assert_eq!(tokenizer.decode(tokenizer.bos()), None);
}

// --- (e) Div backward correctness ---

#[test]
fn test_div_backward() {
    let a_val = 6.0;
    let b_val = 3.0;
    let h = 1e-6;

    let mut tape = Tape::with_capacity(16);
    let a = tape.leaf(a_val);
    let b = tape.leaf(b_val);
    let result = tape.div(a, b);
    tape.backward(result);

    // Forward check
    assert!((tape.val(result) - 2.0).abs() < 1e-10);

    // Gradient check: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    let num_a = numerical_grad(|x| x / b_val, a_val, h);
    let num_b = numerical_grad(|x| a_val / x, b_val, h);

    assert!(
        (tape.grad(a) - num_a).abs() < 1e-5,
        "da: analytical={} numerical={}",
        tape.grad(a),
        num_a
    );
    assert!(
        (tape.grad(b) - num_b).abs() < 1e-5,
        "db: analytical={} numerical={}",
        tape.grad(b),
        num_b
    );
}

#[test]
fn test_div_in_expression() {
    // Test div in a more complex expression: (a / b) * c
    let a_val = 4.0;
    let b_val = 2.0;
    let c_val = 3.0;
    let h = 1e-6;

    let mut tape = Tape::with_capacity(32);
    let a = tape.leaf(a_val);
    let b = tape.leaf(b_val);
    let c = tape.leaf(c_val);
    let d = tape.div(a, b);
    let result = tape.mul(d, c);
    tape.backward(result);

    let num_a = numerical_grad(|x| (x / b_val) * c_val, a_val, h);
    let num_b = numerical_grad(|x| (a_val / x) * c_val, b_val, h);
    let num_c = numerical_grad(|x| (a_val / b_val) * x, c_val, h);

    assert!((tape.grad(a) - num_a).abs() < 1e-5);
    assert!((tape.grad(b) - num_b).abs() < 1e-5);
    assert!((tape.grad(c) - num_c).abs() < 1e-5);
}

// --- (g) ReLU backward correctness ---

#[test]
fn test_autograd_relu() {
    let h = 1e-6;

    // Positive input: relu(x) = x, grad = 1
    let mut tape = Tape::with_capacity(16);
    let a = tape.leaf(3.0);
    let r = tape.relu(a);
    tape.backward(r);
    assert!((tape.val(r) - 3.0).abs() < 1e-10);
    let num = numerical_grad(|x| x.max(0.0), 3.0, h);
    assert!(
        (tape.grad(a) - num).abs() < 1e-5,
        "positive: analytical={} numerical={}",
        tape.grad(a),
        num
    );

    // Zero input: relu(0) = 0, grad = 0 by convention
    let mut tape = Tape::with_capacity(16);
    let a = tape.leaf(0.0);
    let r = tape.relu(a);
    tape.backward(r);
    assert!((tape.val(r)).abs() < 1e-10);
    assert!(tape.grad(a).abs() < 1e-10, "zero: grad should be 0");

    // Negative input: relu(x) = 0, grad = 0
    let mut tape = Tape::with_capacity(16);
    let a = tape.leaf(-2.0);
    let r = tape.relu(a);
    tape.backward(r);
    assert!((tape.val(r) - 0.0).abs() < 1e-10);
    assert!(
        tape.grad(a).abs() < 1e-10,
        "negative: grad should be 0, got {}",
        tape.grad(a)
    );

    // ReLU in expression: relu(a) * b
    let a_val = 2.0;
    let b_val = 5.0;
    let mut tape = Tape::with_capacity(16);
    let a = tape.leaf(a_val);
    let b = tape.leaf(b_val);
    let r = tape.relu(a);
    let result = tape.mul(r, b);
    tape.backward(result);
    let num_a = numerical_grad(|x| x.max(0.0) * b_val, a_val, h);
    let num_b = numerical_grad(|x| a_val.max(0.0) * x, b_val, h);
    assert!((tape.grad(a) - num_a).abs() < 1e-5);
    assert!((tape.grad(b) - num_b).abs() < 1e-5);
}

// --- (h) Dropout tests ---

#[test]
fn test_dropout_zero_is_identity() {
    let mut tape = Tape::with_capacity(64);
    let mut rng = StdRng::seed_from_u64(42);
    let vals = [1.0, 2.0, 3.0, 4.0];
    let x: Vec<Idx> = vals.iter().map(|&v| tape.leaf(v)).collect();
    let y = dropout(&mut tape, &x, 0.0, &mut rng);
    for (i, &yi) in y.iter().enumerate() {
        assert!(
            (tape.val(yi) - vals[i]).abs() < 1e-10,
            "dropout p=0 should be identity"
        );
    }
}

#[test]
fn test_dropout_backward() {
    let mut tape = Tape::with_capacity(64);
    let mut rng = StdRng::seed_from_u64(42);
    let p = 0.5;
    let vals = [1.0, 2.0, 3.0, 4.0];
    let x: Vec<Idx> = vals.iter().map(|&v| tape.leaf(v)).collect();
    let y = dropout(&mut tape, &x, p, &mut rng);
    let s = tape.sum(&y);
    tape.backward(s);
    let scale = 1.0 / (1.0 - p);
    for (i, &xi) in x.iter().enumerate() {
        let g = tape.grad(xi);
        // Gradient should be either 0 (dropped) or 1/(1-p) (kept)
        assert!(
            g.abs() < 1e-10 || (g - scale).abs() < 1e-10,
            "dropout grad at {i} should be 0 or {scale}, got {g}"
        );
    }
}

// --- (i) Scaled init test ---

#[test]
fn test_scaled_init_different_from_flat() {
    let docs = vec!["abc".to_string(), "def".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Relu,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let mut rng1 = StdRng::seed_from_u64(42);
    let mut rng2 = StdRng::seed_from_u64(42);
    let flat = Params::new(&config, &mut rng1, InitScale::Flat);
    let scaled = Params::new(&config, &mut rng2, InitScale::Scaled);
    assert_eq!(flat.data.len(), scaled.data.len());
    // They should differ because different std_devs are used
    let differs = flat
        .data
        .iter()
        .zip(scaled.data.iter())
        .any(|(a, b)| (a - b).abs() > 1e-15);
    assert!(
        differs,
        "scaled and flat init should produce different weights"
    );
}

// --- (j) Validation loss test ---

#[test]
fn test_val_loss_is_finite() {
    let docs = vec!["abc".to_string(), "def".to_string(), "ghi".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Relu,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let mut rng = StdRng::seed_from_u64(42);
    let params = Params::new(&config, &mut rng, InitScale::Flat);
    let val_loss = compute_val_loss(&params, &config, &tokenizer, &docs);
    assert!(
        val_loss.is_finite(),
        "val loss should be finite, got {val_loss}"
    );
    assert!(
        val_loss > 0.0,
        "val loss should be positive, got {val_loss}"
    );
}

// --- (k) End-to-end training: loss decreases after 1 step ---

#[test]
fn test_one_step_training_loss_decreases() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abcabc".to_string(), "defdef".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Relu,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let mut params = Params::new(&config, &mut rng, InitScale::Flat);
    let model = GptModel::build(&config);
    let nt = n_trainable(&config, params.len());
    let adam = AdamConfig {
        lr: 0.01,
        beta1: 0.85,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        warmup_steps: 0,
        schedule: LrSchedule::Linear,
        grad_clip: 0.0,
    };
    let tokens = tokenizer.encode(&docs[0]);

    let mut grads = vec![0.0; nt];
    let loss0 = forward_backward(&params, &config, &model, &tokens, &mut grads, &mut rng);
    adamw_step(&mut params, &config, &adam, &grads, 0, 100);

    grads.iter_mut().for_each(|g| *g = 0.0);
    let loss1 = forward_backward(&params, &config, &model, &tokens, &mut grads, &mut rng);
    assert!(
        loss1 < loss0,
        "loss should decrease after one step: {loss0} -> {loss1}"
    );
}

// --- (k2) SwiGLU: training loss decreases + inference produces valid tokens ---

#[test]
fn test_swiglu_training_loss_decreases() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abcabc".to_string(), "defdef".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::SwiGLU,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let mut params = Params::new(&config, &mut rng, InitScale::Flat);
    let model = GptModel::build(&config);
    let nt = n_trainable(&config, params.len());
    let adam = AdamConfig {
        lr: 0.01,
        beta1: 0.85,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        warmup_steps: 0,
        schedule: LrSchedule::Linear,
        grad_clip: 0.0,
    };
    let tokens = tokenizer.encode(&docs[0]);

    let mut grads = vec![0.0; nt];
    let loss0 = forward_backward(&params, &config, &model, &tokens, &mut grads, &mut rng);
    adamw_step(&mut params, &config, &adam, &grads, 0, 100);

    grads.iter_mut().for_each(|g| *g = 0.0);
    let loss1 = forward_backward(&params, &config, &model, &tokens, &mut grads, &mut rng);
    assert!(
        loss1 < loss0,
        "SwiGLU loss should decrease after one step: {loss0} -> {loss1}"
    );
}

#[test]
fn test_swiglu_generate_sample_valid_tokens() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abc".to_string(), "def".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::SwiGLU,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let params = Params::new(&config, &mut rng, InitScale::Flat);
    let sample = generate_sample(&params, &config, &tokenizer, 0.5, &mut rng);

    let vocab_chars: std::collections::HashSet<char> =
        docs.iter().flat_map(|d| d.chars()).collect();
    for ch in sample.chars() {
        assert!(
            vocab_chars.contains(&ch),
            "SwiGLU generated char '{ch}' not in vocabulary"
        );
    }
}

#[test]
fn test_swiglu_has_more_params_than_silu() {
    let docs = vec!["abc".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let base = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Silu,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let swiglu_config = GptConfig {
        activation: Activation::SwiGLU,
        ..base.clone()
    };
    let mut rng1 = StdRng::seed_from_u64(42);
    let mut rng2 = StdRng::seed_from_u64(42);
    let silu_params = Params::new(&base, &mut rng1, InitScale::Flat);
    let swiglu_params = Params::new(&swiglu_config, &mut rng2, InitScale::Flat);
    // SwiGLU adds one extra gate matrix (4*e*e) per layer
    let e = base.n_embd;
    assert_eq!(
        swiglu_params.len() - silu_params.len(),
        base.n_layer * 4 * e * e,
        "SwiGLU should have exactly 4*e*e more params per layer"
    );
}

#[test]
fn test_swiglu_checkpoint_roundtrip() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abc".to_string(), "bca".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::SwiGLU,
        no_final_norm: false,
        no_learnable_gamma: false,
        dropout: 0.0,
    };
    let params = Params::new(&config, &mut rng, InitScale::Flat);

    let dir = std::env::temp_dir();
    let path = dir.join("microgpt_test_swiglu_checkpoint.mgpt");
    let path_str = path.to_str().unwrap();

    let adam = AdamConfig {
        lr: 0.01,
        beta1: 0.85,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
        warmup_steps: 0,
        schedule: LrSchedule::Linear,
        grad_clip: 0.0,
    };

    save_checkpoint(path_str, &config, &adam, &tokenizer, &params, 10, 1);
    let ckpt = load_checkpoint(path_str);

    assert_eq!(config.activation, ckpt.config.activation);
    assert_eq!(params.data.len(), ckpt.params.data.len());
    for (&a, &b) in params.data.iter().zip(ckpt.params.data.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "param mismatch: {a} vs {b}");
    }

    let _ = std::fs::remove_file(path_str);
}

// --- (k3) SwiGLU: tape forward matches f64 inference ---

#[test]
fn test_swiglu_tape_matches_f64() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abcabc".to_string(), "defdef".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::SwiGLU,
        no_final_norm: false,
        no_learnable_gamma: false,
        dropout: 0.0,
    };
    let params = Params::new(&config, &mut rng, InitScale::Flat);
    let model = GptModel::build(&config);
    let tokens = tokenizer.encode(&docs[0]);
    let token_id = tokens[0];

    // Tape forward
    let mut tape = Tape::with_capacity(config.estimate_tape_nodes());
    tape.load_params(&params.data);
    let c = TapeConstants::new(&mut tape, config.n_embd);
    let mut keys_tape: Vec<Vec<Vec<Idx>>> = (0..config.n_layer).map(|_| Vec::new()).collect();
    let mut vals_tape: Vec<Vec<Vec<Idx>>> = (0..config.n_layer).map(|_| Vec::new()).collect();
    let logits_tape = gpt_forward(
        &mut tape, &model, &config, &c, token_id, 0,
        &mut keys_tape, &mut vals_tape, &mut rng,
    );
    let tape_vals: Vec<f64> = logits_tape.iter().map(|&idx| tape.val(idx)).collect();

    // f64 forward
    let mut keys_f64: Vec<Vec<Vec<f64>>> = (0..config.n_layer).map(|_| Vec::new()).collect();
    let mut vals_f64: Vec<Vec<Vec<f64>>> = (0..config.n_layer).map(|_| Vec::new()).collect();
    let logits_f64 = gpt_forward_f64(
        &params.data, &config, token_id, 0,
        &mut keys_f64, &mut vals_f64,
    );

    assert_eq!(tape_vals.len(), logits_f64.len());
    for (i, (&a, &b)) in tape_vals.iter().zip(logits_f64.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "SwiGLU tape vs f64 mismatch at logit {i}: tape={a} f64={b}"
        );
    }
}

// --- (k4) SwiGLU: numerical gradient check for gate path ---

#[test]
fn test_swiglu_gate_gradient_numerical() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abcabc".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::SwiGLU,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let params = Params::new(&config, &mut rng, InitScale::Flat);
    let model = GptModel::build(&config);
    let tokens = tokenizer.encode(&docs[0]);

    // Get analytical gradients via backward pass
    let nt = n_trainable(&config, params.len());
    let mut grads = vec![0.0; nt];
    forward_backward(&params, &config, &model, &tokens, &mut grads, &mut rng);

    // Spot-check gate weight gradients via finite differences.
    // Gate weights start after wq+wk+wv+wo = 4*e*e attention params,
    // offset by the global embeddings (wte + wpe + lm_head).
    let e = config.n_embd;
    let global_off = config.vocab_size * e + config.block_size * e + config.vocab_size * e;
    let gate_start = global_off + 4 * e * e; // start of gate weights in layer 0
    let h = 1e-5;

    // Check a few gate weight indices
    for &idx in &[gate_start, gate_start + 1, gate_start + 4 * e * e - 1] {
        if idx >= nt {
            continue;
        }
        let mut p_plus = params.clone();
        p_plus.data[idx] += h;
        let mut g_dummy = vec![0.0; nt];
        let loss_plus = forward_backward(&p_plus, &config, &model, &tokens, &mut g_dummy, &mut rng);

        let mut p_minus = params.clone();
        p_minus.data[idx] -= h;
        g_dummy.iter_mut().for_each(|g| *g = 0.0);
        let loss_minus =
            forward_backward(&p_minus, &config, &model, &tokens, &mut g_dummy, &mut rng);

        let numerical = (loss_plus - loss_minus) / (2.0 * h);
        let analytical = grads[idx];
        let denom = analytical.abs().max(numerical.abs()).max(1e-8);
        let rel_err = (analytical - numerical).abs() / denom;
        assert!(
            rel_err < 1e-4,
            "SwiGLU gate grad mismatch at param {idx}: analytical={analytical:.8} numerical={numerical:.8} rel_err={rel_err:.6}"
        );
    }
}

// --- (l) AdamW step updates params, m, v ---

#[test]
fn test_adamw_step_updates_params() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abc".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Relu,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let mut params = Params::new(&config, &mut rng, InitScale::Flat);
    let before = params.data.clone();
    let nt = n_trainable(&config, params.len());
    let grads: Vec<f64> = (0..nt).map(|i| (i as f64) * 0.01).collect();
    let adam = AdamConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        warmup_steps: 0,
        schedule: LrSchedule::Linear,
        grad_clip: 0.0,
    };
    adamw_step(&mut params, &config, &adam, &grads, 0, 100);

    let changed = before
        .iter()
        .zip(params.data.iter())
        .take(nt)
        .any(|(a, b)| (a - b).abs() > 1e-15);
    assert!(changed, "params should change after adamw_step");
    assert!(
        params.m[1].abs() > 0.0,
        "m should be non-zero for non-zero gradient"
    );
    assert!(
        params.v[1] > 0.0,
        "v should be positive for non-zero gradient"
    );
}

// --- (m) LR schedule produces different updates at different steps ---

#[test]
fn test_lr_schedule_values() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abc".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Relu,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let base_params = Params::new(&config, &mut rng, InitScale::Flat);
    let nt = n_trainable(&config, base_params.len());
    let grads: Vec<f64> = vec![0.1; nt];
    let adam = AdamConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        warmup_steps: 0,
        schedule: LrSchedule::Linear,
        grad_clip: 0.0,
    };

    let mut params_early = base_params.clone();
    let mut params_late = base_params.clone();
    adamw_step(&mut params_early, &config, &adam, &grads, 0, 100);
    adamw_step(&mut params_late, &config, &adam, &grads, 99, 100);

    let delta: f64 = params_early
        .data
        .iter()
        .zip(params_late.data.iter())
        .take(nt)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        delta > 0.0,
        "different LR schedule positions should produce different updates"
    );
}

// --- (n) generate_sample produces valid tokens ---

#[test]
fn test_generate_sample_valid_tokens() {
    let mut rng = StdRng::seed_from_u64(42);
    let docs = vec!["abc".to_string(), "def".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Relu,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let params = Params::new(&config, &mut rng, InitScale::Flat);
    let sample = generate_sample(&params, &config, &tokenizer, 0.5, &mut rng);

    let vocab_chars: std::collections::HashSet<char> =
        docs.iter().flat_map(|d| d.chars()).collect();
    for ch in sample.chars() {
        assert!(
            vocab_chars.contains(&ch),
            "generated char '{ch}' not in vocabulary"
        );
    }
}

// --- (o) Edge cases ---

#[test]
fn test_encode_empty_doc() {
    let docs = vec!["abc".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let tokens = tokenizer.encode("");
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], tokenizer.bos());
    assert_eq!(tokens[1], tokenizer.bos());
}

#[test]
fn test_encode_single_char_doc() {
    let docs = vec!["a".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let tokens = tokenizer.encode("a");
    assert_eq!(tokens.len(), 3); // [BOS, a, BOS]
}

#[test]
fn test_encode_unknown_chars_skipped() {
    let docs = vec!["abc".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let tokens = tokenizer.encode("azb");
    // 'z' is not in the vocabulary — should be skipped
    assert_eq!(tokens.len(), 4); // [BOS, a, b, BOS]
    assert_eq!(tokens[0], tokenizer.bos());
    assert_eq!(tokens[3], tokenizer.bos());
}

#[test]
fn test_val_loss_with_short_doc() {
    let docs = vec!["a".to_string()];
    let tokenizer = Tokenizer::from_docs_char(&docs);
    let config = GptConfig {
        n_layer: 1,
        n_embd: 4,
        block_size: 4,
        n_head: 2,
        head_dim: 2,
        vocab_size: tokenizer.vocab_size(),
        activation: Activation::Relu,
        no_final_norm: true,
        no_learnable_gamma: true,
        dropout: 0.0,
    };
    let mut rng = StdRng::seed_from_u64(42);
    let params = Params::new(&config, &mut rng, InitScale::Flat);
    let val_loss = compute_val_loss(&params, &config, &tokenizer, &docs);
    assert!(
        val_loss.is_finite(),
        "val loss should be finite for short doc"
    );
}

// --- (p) Shuffle helper ---

#[test]
fn test_shuffle_preserves_elements() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data: Vec<usize> = (0..10).collect();
    let original: Vec<usize> = data.clone();
    shuffle(&mut data, &mut rng);

    let mut sorted = data.clone();
    sorted.sort();
    assert_eq!(sorted, original);
    assert_ne!(
        data, original,
        "shuffle should change order (may rarely fail)"
    );
}
