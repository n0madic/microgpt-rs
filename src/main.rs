//! The most atomic way to train and run inference for a GPT in pure Rust.
//! This file is the complete algorithm. Everything else is just efficiency.
//!
//! Rewritten from Karpathy's microgpt.py — same architecture, idiomatic Rust.
//! Tape-based autograd instead of pointer graph. One file = complete algorithm.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::collections::{HashMap, VecDeque};
use std::fs;

// ---------------------------------------------------------------------------
// Section 1: Autograd Engine — tape-based (Wengert list)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Idx(usize);

enum Op {
    Leaf,
    Add(Idx, Idx),
    Mul(Idx, Idx),
    Pow(Idx, f64),
    Log(Idx),
    Exp(Idx),
    Div(Idx, Idx),
    Relu(Idx),
}

struct Node {
    data: f64,
    grad: f64,
    op: Op,
}

struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    fn with_capacity(cap: usize) -> Self {
        Tape {
            nodes: Vec::with_capacity(cap),
        }
    }

    fn leaf(&mut self, data: f64) -> Idx {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Leaf,
        });
        Idx(idx)
    }

    fn add(&mut self, a: Idx, b: Idx) -> Idx {
        let data = self.nodes[a.0].data + self.nodes[b.0].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Add(a, b),
        });
        Idx(idx)
    }

    fn mul(&mut self, a: Idx, b: Idx) -> Idx {
        let data = self.nodes[a.0].data * self.nodes[b.0].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Mul(a, b),
        });
        Idx(idx)
    }

    fn pow(&mut self, a: Idx, exp: f64) -> Idx {
        let data = self.nodes[a.0].data.powf(exp);
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Pow(a, exp),
        });
        Idx(idx)
    }

    fn log(&mut self, a: Idx) -> Idx {
        let data = self.nodes[a.0].data.ln();
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Log(a),
        });
        Idx(idx)
    }

    fn exp(&mut self, a: Idx) -> Idx {
        let data = self.nodes[a.0].data.exp();
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Exp(a),
        });
        Idx(idx)
    }

    fn div(&mut self, a: Idx, b: Idx) -> Idx {
        let data = self.nodes[a.0].data / self.nodes[b.0].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Div(a, b),
        });
        Idx(idx)
    }

    fn relu(&mut self, a: Idx) -> Idx {
        let data = self.nodes[a.0].data.max(0.0);
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Relu(a),
        });
        Idx(idx)
    }

    fn sum(&mut self, xs: &[Idx]) -> Idx {
        assert!(!xs.is_empty());
        let mut acc = xs[0];
        for &x in &xs[1..] {
            acc = self.add(acc, x);
        }
        acc
    }

    fn load_params(&mut self, data: &[f64]) {
        debug_assert!(self.nodes.is_empty());
        self.nodes.extend(data.iter().map(|&d| Node {
            data: d,
            grad: 0.0,
            op: Op::Leaf,
        }));
    }

    fn val(&self, idx: Idx) -> f64 {
        self.nodes[idx.0].data
    }

    fn grad(&self, idx: Idx) -> f64 {
        self.nodes[idx.0].grad
    }

    fn backward(&mut self, root: Idx) {
        for node in &mut self.nodes {
            node.grad = 0.0;
        }
        self.nodes[root.0].grad = 1.0;

        // Reverse pass — nodes are already in topological order by construction
        for i in (0..=root.0).rev() {
            let g = self.nodes[i].grad;
            if g == 0.0 {
                continue;
            }
            match self.nodes[i].op {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    self.nodes[a.0].grad += g;
                    self.nodes[b.0].grad += g;
                }
                Op::Mul(a, b) => {
                    let a_data = self.nodes[a.0].data;
                    let b_data = self.nodes[b.0].data;
                    self.nodes[a.0].grad += b_data * g;
                    self.nodes[b.0].grad += a_data * g;
                }
                Op::Pow(a, exp) => {
                    let a_data = self.nodes[a.0].data;
                    self.nodes[a.0].grad += exp * a_data.powf(exp - 1.0) * g;
                }
                Op::Log(a) => {
                    let a_data = self.nodes[a.0].data;
                    self.nodes[a.0].grad += g / a_data;
                }
                Op::Exp(a) => {
                    let out_data = self.nodes[i].data; // exp(a) is stored as this node's data
                    self.nodes[a.0].grad += out_data * g;
                }
                Op::Div(a, b) => {
                    let b_data = self.nodes[b.0].data;
                    self.nodes[a.0].grad += g / b_data;
                    self.nodes[b.0].grad -= g * self.nodes[a.0].data / (b_data * b_data);
                }
                Op::Relu(a) => {
                    if self.nodes[a.0].data > 0.0 {
                        self.nodes[a.0].grad += g;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Section 2: Tokenizer — char-level and BPE with BOS token
// ---------------------------------------------------------------------------

struct CharTokenizer {
    chars: Vec<char>,
    char_to_id: HashMap<char, usize>,
}

struct BpeTokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, usize>,
    merges: Vec<(usize, usize)>,
}

enum Tokenizer {
    Char(CharTokenizer),
    Bpe(BpeTokenizer),
}

impl Tokenizer {
    fn from_docs_char(docs: &[String]) -> Self {
        let mut chars_set = std::collections::BTreeSet::new();
        for doc in docs {
            for ch in doc.chars() {
                chars_set.insert(ch);
            }
        }
        let chars: Vec<char> = chars_set.into_iter().collect();
        let char_to_id: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Tokenizer::Char(CharTokenizer { chars, char_to_id })
    }

    fn from_docs_bpe(docs: &[String], target_vocab_size: usize) -> Self {
        // Step 1: Build initial char-level vocabulary (sorted)
        let mut chars_set = std::collections::BTreeSet::new();
        for doc in docs {
            for ch in doc.chars() {
                chars_set.insert(ch);
            }
        }
        let initial_chars: Vec<char> = chars_set.into_iter().collect();
        let mut vocab: Vec<String> = initial_chars.iter().map(|c| c.to_string()).collect();
        let mut token_to_id: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        // Step 2: Tokenize all docs into initial char-level token IDs
        let mut tokenized_docs: Vec<Vec<usize>> = docs
            .iter()
            .map(|doc| doc.chars().map(|ch| token_to_id[&ch.to_string()]).collect())
            .collect();

        let mut merges: Vec<(usize, usize)> = Vec::new();

        // Step 3: Iteratively merge most frequent pair until target vocab size
        // target_vocab_size includes BOS, so stop when vocab.len() + 1 >= target
        while vocab.len() + 1 < target_vocab_size {
            let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
            for tokens in &tokenized_docs {
                for window in tokens.windows(2) {
                    *pair_counts.entry((window[0], window[1])).or_insert(0) += 1;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            let &best_pair = pair_counts
                .iter()
                .max_by_key(|&(_, &count)| count)
                .unwrap()
                .0;

            let new_token = format!("{}{}", vocab[best_pair.0], vocab[best_pair.1]);
            let new_id = vocab.len();
            vocab.push(new_token.clone());
            token_to_id.insert(new_token, new_id);
            merges.push(best_pair);

            // Apply merge to all tokenized docs
            for tokens in &mut tokenized_docs {
                let mut new_tokens = Vec::with_capacity(tokens.len());
                let mut i = 0;
                while i < tokens.len() {
                    if i + 1 < tokens.len()
                        && tokens[i] == best_pair.0
                        && tokens[i + 1] == best_pair.1
                    {
                        new_tokens.push(new_id);
                        i += 2;
                    } else {
                        new_tokens.push(tokens[i]);
                        i += 1;
                    }
                }
                *tokens = new_tokens;
            }
        }

        Tokenizer::Bpe(BpeTokenizer {
            vocab,
            token_to_id,
            merges,
        })
    }

    fn bos(&self) -> usize {
        match self {
            Tokenizer::Char(ct) => ct.chars.len(),
            Tokenizer::Bpe(bt) => bt.vocab.len(),
        }
    }

    fn vocab_size(&self) -> usize {
        self.bos() + 1
    }

    fn encode(&self, doc: &str) -> Vec<usize> {
        let bos = self.bos();
        match self {
            Tokenizer::Char(ct) => {
                let mut tokens = Vec::with_capacity(doc.len() + 2);
                tokens.push(bos);
                for ch in doc.chars() {
                    tokens.push(ct.char_to_id[&ch]);
                }
                tokens.push(bos);
                tokens
            }
            Tokenizer::Bpe(bt) => {
                // Start with char-level tokens
                let mut tokens: Vec<usize> = doc
                    .chars()
                    .map(|ch| bt.token_to_id[&ch.to_string()])
                    .collect();
                // Apply merges in priority order
                for &(a, b) in &bt.merges {
                    let merged_str = format!("{}{}", bt.vocab[a], bt.vocab[b]);
                    let merged_id = bt.token_to_id[&merged_str];
                    let mut new_tokens = Vec::with_capacity(tokens.len());
                    let mut i = 0;
                    while i < tokens.len() {
                        if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
                            new_tokens.push(merged_id);
                            i += 2;
                        } else {
                            new_tokens.push(tokens[i]);
                            i += 1;
                        }
                    }
                    tokens = new_tokens;
                }
                let mut result = Vec::with_capacity(tokens.len() + 2);
                result.push(bos);
                result.extend(tokens);
                result.push(bos);
                result
            }
        }
    }

    fn decode(&self, token: usize) -> Option<String> {
        let bos = self.bos();
        if token == bos {
            return None;
        }
        match self {
            Tokenizer::Char(ct) => Some(ct.chars[token].to_string()),
            Tokenizer::Bpe(bt) => Some(bt.vocab[token].clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Section 3: Model Config & Parameters
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Activation {
    Silu,
    Relu,
}

#[derive(Clone, Copy, PartialEq)]
enum LrSchedule {
    Cosine,
    Linear,
}

#[derive(Clone, Copy, PartialEq)]
enum TokenizerType {
    Char,
    Bpe,
}

#[derive(Clone, Copy, PartialEq)]
enum InitScale {
    Flat,
    Scaled,
}

struct GptConfig {
    n_layer: usize,
    n_embd: usize,
    block_size: usize,
    n_head: usize,
    head_dim: usize,
    vocab_size: usize,
    activation: Activation,
    no_final_norm: bool,
    no_learnable_gamma: bool,
    dropout: f64,
}

impl GptConfig {
    fn estimate_tape_nodes(&self) -> usize {
        let e = self.n_embd;
        let n_params = self.vocab_size * e
            + self.block_size * e
            + self.vocab_size * e
            + self.n_layer * (4 * e * e + 4 * e * e + e * 4 * e)
            + (2 + 2 * self.n_layer) * e;
        // Per position: embedding add + rmsnorm + per-layer (attn + mlp) + final norm + lm_head
        // Rough estimate: ~130K nodes per position for 2L/48E/8H
        let per_pos = e                              // tok+pos add
            + 3 * e                                  // initial rmsnorm (sq + sum + mul*2 per elem)
            + self.n_layer * (
                3 * e                                // attn_norm rmsnorm
                + 3 * (e * e + e)                    // Q, K, V linear (mul + sum per row)
                + self.n_head * (
                    self.block_size * (self.head_dim + 1)  // attn logits (dot + scale)
                    + self.block_size * 3             // softmax (sub + exp + div)
                    + self.block_size * 2             // dropout on attn weights
                    + self.head_dim * self.block_size  // weighted sum
                )
                + e * e + e                          // Wo linear
                + 2 * e                              // dropout (worst case: scale per elem)
                + e                                  // residual add
                + 3 * e                              // mlp_norm rmsnorm
                + 4 * e * e + 4 * e                  // fc1 linear
                + 4 * e * 6                          // silu (neg, exp, add, pow, mul per elem)
                + e * 4 * e + e                      // fc2 linear
                + 2 * e                              // dropout
                + e                                  // residual add
            )
            + 3 * e                                  // final rmsnorm
            + self.vocab_size * e + self.vocab_size; // lm_head + log_softmax
        n_params + per_pos * self.block_size
    }
}

struct Params {
    data: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
}

impl Params {
    fn new(config: &GptConfig, rng: &mut StdRng, init_scale: InitScale) -> Self {
        let e = config.n_embd;
        let n_layer = config.n_layer;
        // Random params: wte + wpe + lm_head + layer weights
        let n_random = config.vocab_size * e   // wte
            + config.block_size * e            // wpe
            + config.vocab_size * e            // lm_head
            + n_layer * (4 * e * e + 4 * e * e + e * 4 * e); // layers
        // Gamma params (init=1.0): initial_norm + final_norm + per-layer (pre-attn + pre-mlp)
        let n_gamma = (2 + 2 * n_layer) * e;
        let n = n_random + n_gamma;

        let mut data: Vec<f64> = Vec::with_capacity(n);

        match init_scale {
            InitScale::Flat => {
                let std_dev = 0.08;
                data.extend((0..n_random).map(|_| rng.sample::<f64, _>(StandardNormal) * std_dev));
            }
            InitScale::Scaled => {
                let emb_std = 0.02;
                let qkv_std = 1.0 / (e as f64).sqrt();
                let residual_std = 1.0 / (2.0 * n_layer as f64 * e as f64).sqrt();

                // wte: vocab_size * e
                for _ in 0..config.vocab_size * e {
                    data.push(rng.sample::<f64, _>(StandardNormal) * emb_std);
                }
                // wpe: block_size * e
                for _ in 0..config.block_size * e {
                    data.push(rng.sample::<f64, _>(StandardNormal) * emb_std);
                }
                // lm_head: vocab_size * e
                for _ in 0..config.vocab_size * e {
                    data.push(rng.sample::<f64, _>(StandardNormal) * emb_std);
                }
                // Per-layer weights
                for _ in 0..n_layer {
                    // wq: e*e — QKV projection
                    for _ in 0..e * e {
                        data.push(rng.sample::<f64, _>(StandardNormal) * qkv_std);
                    }
                    // wk: e*e
                    for _ in 0..e * e {
                        data.push(rng.sample::<f64, _>(StandardNormal) * qkv_std);
                    }
                    // wv: e*e
                    for _ in 0..e * e {
                        data.push(rng.sample::<f64, _>(StandardNormal) * qkv_std);
                    }
                    // wo: e*e — residual projection
                    for _ in 0..e * e {
                        data.push(rng.sample::<f64, _>(StandardNormal) * residual_std);
                    }
                    // fc1: 4*e*e — standard
                    for _ in 0..4 * e * e {
                        data.push(rng.sample::<f64, _>(StandardNormal) * qkv_std);
                    }
                    // fc2: e*4*e — residual projection
                    for _ in 0..e * 4 * e {
                        data.push(rng.sample::<f64, _>(StandardNormal) * residual_std);
                    }
                }
            }
        }

        data.extend(std::iter::repeat_n(1.0, n_gamma));

        let m = vec![0.0; n];
        let v = vec![0.0; n];
        Params { data, m, v }
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Section 4: Model on Tape — load params as leaf nodes
// ---------------------------------------------------------------------------

struct LayerModel {
    attn_wq: Vec<Vec<Idx>>,
    attn_wk: Vec<Vec<Idx>>,
    attn_wv: Vec<Vec<Idx>>,
    attn_wo: Vec<Vec<Idx>>,
    mlp_fc1: Vec<Vec<Idx>>,
    mlp_fc2: Vec<Vec<Idx>>,
    attn_norm_gamma: Vec<Idx>,
    mlp_norm_gamma: Vec<Idx>,
}

struct GptModel {
    wte: Vec<Vec<Idx>>,
    wpe: Vec<Vec<Idx>>,
    lm_head: Vec<Vec<Idx>>,
    layers: Vec<LayerModel>,
    initial_norm_gamma: Vec<Idx>,
    final_norm_gamma: Vec<Idx>,
}

impl GptModel {
    /// Build index template once — indices are 0..n_params matching Params::data layout.
    fn build(config: &GptConfig) -> Self {
        let mut offset = 0;

        let mut make_matrix = |rows: usize, cols: usize| -> Vec<Vec<Idx>> {
            let mut mat = Vec::with_capacity(rows);
            for _ in 0..rows {
                let mut row = Vec::with_capacity(cols);
                for _ in 0..cols {
                    row.push(Idx(offset));
                    offset += 1;
                }
                mat.push(row);
            }
            mat
        };

        let e = config.n_embd;
        let wte = make_matrix(config.vocab_size, e);
        let wpe = make_matrix(config.block_size, e);
        let lm_head = make_matrix(config.vocab_size, e);

        let mut layers = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            let attn_wq = make_matrix(e, e);
            let attn_wk = make_matrix(e, e);
            let attn_wv = make_matrix(e, e);
            let attn_wo = make_matrix(e, e);
            let mlp_fc1 = make_matrix(4 * e, e);
            let mlp_fc2 = make_matrix(e, 4 * e);
            layers.push(LayerModel {
                attn_wq,
                attn_wk,
                attn_wv,
                attn_wo,
                mlp_fc1,
                mlp_fc2,
                attn_norm_gamma: Vec::new(),
                mlp_norm_gamma: Vec::new(),
            });
        }

        let initial_norm_gamma = make_matrix(1, e).remove(0);
        let final_norm_gamma = make_matrix(1, e).remove(0);
        for layer in &mut layers {
            layer.attn_norm_gamma = make_matrix(1, e).remove(0);
            layer.mlp_norm_gamma = make_matrix(1, e).remove(0);
        }

        GptModel {
            wte,
            wpe,
            lm_head,
            layers,
            initial_norm_gamma,
            final_norm_gamma,
        }
    }
}

// ---------------------------------------------------------------------------
// Section 5: Forward Pass
// ---------------------------------------------------------------------------

struct TapeConstants {
    minus_one: Idx,
    one: Idx,
    eps: Idx,
    inv_n_embd: Idx,
}

impl TapeConstants {
    fn new(tape: &mut Tape, n_embd: usize) -> Self {
        TapeConstants {
            minus_one: tape.leaf(-1.0),
            one: tape.leaf(1.0),
            eps: tape.leaf(1e-5),
            inv_n_embd: tape.leaf(1.0 / n_embd as f64),
        }
    }
}

fn linear(tape: &mut Tape, x: &[Idx], w: &[Vec<Idx>]) -> Vec<Idx> {
    w.iter().map(|wo| dot(tape, wo, x)).collect()
}

fn softmax(tape: &mut Tape, logits: &[Idx], c: &TapeConstants) -> Vec<Idx> {
    let max_val = logits
        .iter()
        .map(|&l| tape.val(l))
        .fold(f64::NEG_INFINITY, f64::max);
    let max_node = tape.leaf(max_val);
    let neg_max = tape.mul(max_node, c.minus_one);
    let exps: Vec<Idx> = logits
        .iter()
        .map(|&l| {
            let shifted = tape.add(l, neg_max);
            tape.exp(shifted)
        })
        .collect();
    let total = tape.sum(&exps);
    exps.iter().map(|&e| tape.div(e, total)).collect()
}

fn log_softmax(tape: &mut Tape, logits: &[Idx], c: &TapeConstants) -> Vec<Idx> {
    let max_val = logits
        .iter()
        .map(|&l| tape.val(l))
        .fold(f64::NEG_INFINITY, f64::max);
    let max_node = tape.leaf(max_val);
    let neg_max = tape.mul(max_node, c.minus_one);
    let shifted: Vec<Idx> = logits.iter().map(|&l| tape.add(l, neg_max)).collect();
    let exps: Vec<Idx> = shifted.iter().map(|&s| tape.exp(s)).collect();
    let sum_exp = tape.sum(&exps);
    let log_sum = tape.log(sum_exp);
    let neg_log = tape.mul(log_sum, c.minus_one);
    shifted.iter().map(|&s| tape.add(s, neg_log)).collect()
}

fn silu(tape: &mut Tape, x: Idx, c: &TapeConstants) -> Idx {
    let neg_x = tape.mul(x, c.minus_one);
    let e = tape.exp(neg_x);
    let denom = tape.add(c.one, e);
    let sig = tape.pow(denom, -1.0);
    tape.mul(x, sig)
}

fn rmsnorm(tape: &mut Tape, x: &[Idx], gamma: &[Idx], c: &TapeConstants) -> Vec<Idx> {
    let sq: Vec<Idx> = x.iter().map(|&xi| tape.mul(xi, xi)).collect();
    let ms_sum = tape.sum(&sq);
    let ms = tape.mul(ms_sum, c.inv_n_embd);
    let ms_eps = tape.add(ms, c.eps);
    let scale = tape.pow(ms_eps, -0.5);
    x.iter()
        .zip(gamma.iter())
        .map(|(&xi, &gi)| {
            let normed = tape.mul(xi, scale);
            tape.mul(normed, gi)
        })
        .collect()
}

fn dot(tape: &mut Tape, a: &[Idx], b: &[Idx]) -> Idx {
    let mut acc = tape.mul(a[0], b[0]);
    for i in 1..a.len() {
        let prod = tape.mul(a[i], b[i]);
        acc = tape.add(acc, prod);
    }
    acc
}

fn dropout(tape: &mut Tape, x: &[Idx], p: f64, rng: &mut StdRng) -> Vec<Idx> {
    if p == 0.0 {
        return x.to_vec();
    }
    let scale = 1.0 / (1.0 - p);
    x.iter()
        .map(|&xi| {
            if rng.random::<f64>() < p {
                tape.leaf(0.0)
            } else {
                let s = tape.leaf(scale);
                tape.mul(xi, s)
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn gpt_forward(
    tape: &mut Tape,
    model: &GptModel,
    config: &GptConfig,
    c: &TapeConstants,
    token_id: usize,
    pos_id: usize,
    keys: &mut [Vec<Vec<Idx>>],
    values: &mut [Vec<Vec<Idx>>],
    rng: &mut StdRng,
) -> Vec<Idx> {
    // Token + position embedding
    let tok_emb = &model.wte[token_id];
    let pos_emb = &model.wpe[pos_id];
    let mut x: Vec<Idx> = tok_emb
        .iter()
        .zip(pos_emb.iter())
        .map(|(&t, &p)| tape.add(t, p))
        .collect();
    x = rmsnorm(tape, &x, &model.initial_norm_gamma, c);

    for li in 0..config.n_layer {
        let layer = &model.layers[li];

        // 1) Multi-head Attention
        let x_residual = x.clone();
        x = rmsnorm(tape, &x, &layer.attn_norm_gamma, c);
        let q = linear(tape, &x, &layer.attn_wq);
        let k = linear(tape, &x, &layer.attn_wk);
        let v = linear(tape, &x, &layer.attn_wv);
        keys[li].push(k);
        values[li].push(v);

        let mut x_attn = Vec::with_capacity(config.n_embd);
        let scale_val = (config.head_dim as f64).sqrt();
        let scale_node = tape.leaf(1.0 / scale_val);

        for h in 0..config.n_head {
            let hs = h * config.head_dim;
            let he = hs + config.head_dim;
            let q_h = &q[hs..he];

            // Compute attention logits for all cached keys
            let n_cached = keys[li].len();
            let attn_logits: Vec<Idx> = (0..n_cached)
                .map(|t| {
                    let k_h = &keys[li][t][hs..he];
                    let d = dot(tape, q_h, k_h);
                    tape.mul(d, scale_node)
                })
                .collect();

            let attn_weights = softmax(tape, &attn_logits, c);
            let attn_weights = dropout(tape, &attn_weights, config.dropout, rng);

            // Weighted sum of values
            for j in 0..config.head_dim {
                let terms: Vec<Idx> = (0..n_cached)
                    .map(|t| {
                        let v_tj = values[li][t][hs + j];
                        tape.mul(attn_weights[t], v_tj)
                    })
                    .collect();
                x_attn.push(tape.sum(&terms));
            }
        }

        x = linear(tape, &x_attn, &layer.attn_wo);
        x = dropout(tape, &x, config.dropout, rng);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(&a, &b)| tape.add(a, b))
            .collect();

        // 2) MLP block
        let x_residual = x.clone();
        x = rmsnorm(tape, &x, &layer.mlp_norm_gamma, c);
        x = linear(tape, &x, &layer.mlp_fc1);
        x = match config.activation {
            Activation::Silu => x.iter().map(|&xi| silu(tape, xi, c)).collect(),
            Activation::Relu => x.iter().map(|&xi| tape.relu(xi)).collect(),
        };
        x = linear(tape, &x, &layer.mlp_fc2);
        x = dropout(tape, &x, config.dropout, rng);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(&a, &b)| tape.add(a, b))
            .collect();
    }

    // Final norm before output projection
    if !config.no_final_norm {
        x = rmsnorm(tape, &x, &model.final_norm_gamma, c);
    }
    linear(tape, &x, &model.lm_head)
}

// ---------------------------------------------------------------------------
// Section 6: Training — AdamW optimizer with cosine LR & gradient clipping
// ---------------------------------------------------------------------------

struct AdamConfig {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    warmup_steps: usize,
    schedule: LrSchedule,
    grad_clip: f64,
}

fn n_trainable(config: &GptConfig, n_params: usize) -> usize {
    if config.no_learnable_gamma {
        let e = config.n_embd;
        config.vocab_size * e
            + config.block_size * e
            + config.vocab_size * e
            + config.n_layer * (4 * e * e + 4 * e * e + e * 4 * e)
    } else {
        n_params
    }
}

fn forward_backward(
    params: &Params,
    config: &GptConfig,
    model: &GptModel,
    tokens: &[usize],
    grads_buf: &mut [f64],
    rng: &mut StdRng,
) -> f64 {
    let n = tokens.len().saturating_sub(1).min(config.block_size);
    if n == 0 {
        return 0.0;
    }

    let mut tape = Tape::with_capacity(config.estimate_tape_nodes());
    tape.load_params(&params.data);
    let c = TapeConstants::new(&mut tape, config.n_embd);

    let mut keys: Vec<Vec<Vec<Idx>>> = (0..config.n_layer).map(|_| Vec::new()).collect();
    let mut values: Vec<Vec<Vec<Idx>>> = (0..config.n_layer).map(|_| Vec::new()).collect();

    let mut losses = Vec::with_capacity(n);
    for pos_id in 0..n {
        let token_id = tokens[pos_id];
        let target_id = tokens[pos_id + 1];
        let logits = gpt_forward(
            &mut tape,
            model,
            config,
            &c,
            token_id,
            pos_id,
            &mut keys,
            &mut values,
            rng,
        );
        let log_probs = log_softmax(&mut tape, &logits, &c);
        let neg_log_prob = tape.mul(log_probs[target_id], c.minus_one);
        losses.push(neg_log_prob);
    }

    let loss_sum = tape.sum(&losses);
    let inv_n = tape.leaf(1.0 / n as f64);
    let loss = tape.mul(loss_sum, inv_n);
    let loss_val = tape.val(loss);

    tape.backward(loss);

    // Accumulate gradients into grads_buf
    let nt = n_trainable(config, params.len());
    for (i, g) in grads_buf.iter_mut().enumerate().take(nt) {
        *g += tape.grad(Idx(i));
    }

    loss_val
}

fn adamw_step(
    params: &mut Params,
    config: &GptConfig,
    adam: &AdamConfig,
    grads: &[f64],
    step: usize,
    num_steps: usize,
) {
    let nt = n_trainable(config, params.len());

    // Gradient clipping (global norm), disabled when grad_clip <= 0
    let clip_coef = if adam.grad_clip > 0.0 {
        let grad_norm_sq: f64 = grads[..nt].iter().map(|g| g * g).sum();
        let grad_norm = grad_norm_sq.sqrt();
        (adam.grad_clip / (grad_norm + 1e-6)).min(1.0)
    } else {
        1.0
    };

    // LR schedule
    let lr_t = match adam.schedule {
        LrSchedule::Cosine => {
            let warmup_steps = adam.warmup_steps;
            if step < warmup_steps {
                adam.lr * (step as f64 / warmup_steps as f64)
            } else if num_steps <= warmup_steps {
                adam.lr
            } else {
                let progress = (step - warmup_steps) as f64 / (num_steps - warmup_steps) as f64;
                adam.lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            }
        }
        LrSchedule::Linear => adam.lr * (1.0 - step as f64 / num_steps as f64),
    };

    // AdamW update
    let step_i = (step + 1) as i32;
    for (i, &gi) in grads.iter().enumerate().take(nt) {
        let g = gi * clip_coef;
        params.m[i] = adam.beta1 * params.m[i] + (1.0 - adam.beta1) * g;
        params.v[i] = adam.beta2 * params.v[i] + (1.0 - adam.beta2) * g * g;
        let m_hat = params.m[i] / (1.0 - adam.beta1.powi(step_i));
        let v_hat = params.v[i] / (1.0 - adam.beta2.powi(step_i));
        params.data[i] -=
            lr_t * (m_hat / (v_hat.sqrt() + adam.eps) + adam.weight_decay * params.data[i]);
    }
}

// ---------------------------------------------------------------------------
// Section 7: Inference — tape-free f64 forward pass
// ---------------------------------------------------------------------------

fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / total).collect()
}

fn weighted_sample(probs: &[f64], rng: &mut StdRng) -> usize {
    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

fn linear_f64(x: &[f64], w: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = x.len();
    debug_assert_eq!(w.len(), out_dim * in_dim);
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum()
        })
        .collect()
}

fn rmsnorm_f64(x: &[f64], gamma: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let ms: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>() / n;
    let scale = 1.0 / (ms + 1e-5).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(&xi, &gi)| xi * scale * gi)
        .collect()
}

fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn relu_f64(x: f64) -> f64 {
    x.max(0.0)
}

fn gpt_forward_f64(
    params: &[f64],
    config: &GptConfig,
    token_id: usize,
    pos_id: usize,
    keys: &mut [Vec<Vec<f64>>],
    values: &mut [Vec<Vec<f64>>],
) -> Vec<f64> {
    let e = config.n_embd;
    let mut off = 0;

    // wte: vocab_size × e
    let wte_off = off;
    off += config.vocab_size * e;
    // wpe: block_size × e
    let wpe_off = off;
    off += config.block_size * e;
    // lm_head: vocab_size × e
    let lm_head_off = off;
    off += config.vocab_size * e;

    // Per-layer weight offsets: wq, wk, wv, wo, fc1, fc2
    let layer_random_size = 4 * e * e + 4 * e * e + e * 4 * e;
    let layers_off = off;
    off += config.n_layer * layer_random_size;

    // Gamma offsets: initial_norm, final_norm, then per-layer (attn_norm, mlp_norm)
    let initial_norm_off = off;
    off += e;
    let final_norm_off = off;
    off += e;
    let layer_gamma_off = off;
    // each layer: 2 * e gamma values

    // Token + position embedding
    let tok_base = wte_off + token_id * e;
    let pos_base = wpe_off + pos_id * e;
    let mut x: Vec<f64> = (0..e)
        .map(|j| params[tok_base + j] + params[pos_base + j])
        .collect();

    // Initial RMSNorm
    x = rmsnorm_f64(&x, &params[initial_norm_off..initial_norm_off + e]);

    for li in 0..config.n_layer {
        let l_off = layers_off + li * layer_random_size;
        let wq_off = l_off;
        let wk_off = wq_off + e * e;
        let wv_off = wk_off + e * e;
        let wo_off = wv_off + e * e;
        let fc1_off = wo_off + e * e;
        let fc2_off = fc1_off + 4 * e * e;

        let lg_off = layer_gamma_off + li * 2 * e;
        let attn_gamma = &params[lg_off..lg_off + e];
        let mlp_gamma = &params[lg_off + e..lg_off + 2 * e];

        // 1) Multi-head Attention
        let x_residual = x.clone();
        x = rmsnorm_f64(&x, attn_gamma);
        let q = linear_f64(&x, &params[wq_off..wk_off], e);
        let k = linear_f64(&x, &params[wk_off..wv_off], e);
        let v = linear_f64(&x, &params[wv_off..wo_off], e);
        keys[li].push(k);
        values[li].push(v);

        let mut x_attn = Vec::with_capacity(e);
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        for h in 0..config.n_head {
            let hs = h * config.head_dim;
            let he = hs + config.head_dim;
            let q_h = &q[hs..he];

            let n_cached = keys[li].len();
            let attn_logits: Vec<f64> = (0..n_cached)
                .map(|t| {
                    let k_h = &keys[li][t][hs..he];
                    q_h.iter()
                        .zip(k_h.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
                        * scale
                })
                .collect();

            let attn_weights = softmax_f64(&attn_logits);

            for j in 0..config.head_dim {
                let val: f64 = (0..n_cached)
                    .map(|t| attn_weights[t] * values[li][t][hs + j])
                    .sum();
                x_attn.push(val);
            }
        }

        x = linear_f64(&x_attn, &params[wo_off..fc1_off], e);
        for j in 0..e {
            x[j] += x_residual[j];
        }

        // 2) MLP block
        let x_residual = x.clone();
        x = rmsnorm_f64(&x, mlp_gamma);
        x = linear_f64(&x, &params[fc1_off..fc2_off], 4 * e);
        match config.activation {
            Activation::Silu => {
                for v in &mut x {
                    *v = silu_f64(*v);
                }
            }
            Activation::Relu => {
                for v in &mut x {
                    *v = relu_f64(*v);
                }
            }
        }
        x = linear_f64(&x, &params[fc2_off..fc2_off + e * 4 * e], e);
        for j in 0..e {
            x[j] += x_residual[j];
        }
    }

    // Final norm + lm_head
    if !config.no_final_norm {
        x = rmsnorm_f64(&x, &params[final_norm_off..final_norm_off + e]);
    }
    linear_f64(
        &x,
        &params[lm_head_off..lm_head_off + config.vocab_size * e],
        config.vocab_size,
    )
}

fn generate_sample(
    params: &Params,
    config: &GptConfig,
    tokenizer: &Tokenizer,
    temperature: f64,
    rng: &mut StdRng,
) -> String {
    let mut keys: Vec<Vec<Vec<f64>>> = (0..config.n_layer).map(|_| Vec::new()).collect();
    let mut values: Vec<Vec<Vec<f64>>> = (0..config.n_layer).map(|_| Vec::new()).collect();

    let mut token_id = tokenizer.bos();
    let mut sample = String::new();

    for pos_id in 0..config.block_size {
        let logits = gpt_forward_f64(
            &params.data,
            config,
            token_id,
            pos_id,
            &mut keys,
            &mut values,
        );

        let logit_vals: Vec<f64> = logits.iter().map(|&l| l / temperature).collect();
        let probs = softmax_f64(&logit_vals);

        token_id = weighted_sample(&probs, rng);
        if token_id == tokenizer.bos() {
            break;
        }
        if let Some(s) = tokenizer.decode(token_id) {
            sample.push_str(&s);
        }
    }

    sample
}

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
                       --save PATH     save trained model to file\n  \
                       --load PATH     load model and skip training (inference only)",
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
    if args.steps == 0 {
        errors.push("--steps must be >= 1".into());
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

// ---------------------------------------------------------------------------
// Section 9: Model checkpoint — save/load
// ---------------------------------------------------------------------------

const CHECKPOINT_MAGIC: &[u8; 4] = b"MGPT";
const CHECKPOINT_VERSION: u64 = 1;

fn save_checkpoint(path: &str, config: &GptConfig, tokenizer: &Tokenizer, params: &Params) {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(CHECKPOINT_MAGIC);
    for &v in &[
        CHECKPOINT_VERSION,
        config.n_layer as u64,
        config.n_embd as u64,
        config.n_head as u64,
        config.block_size as u64,
    ] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    // Architectural flags
    buf.push(match config.activation {
        Activation::Silu => 0,
        Activation::Relu => 1,
    });
    buf.push(config.no_final_norm as u8);
    buf.push(config.no_learnable_gamma as u8);
    // Tokenizer type + data
    match tokenizer {
        Tokenizer::Char(ct) => {
            buf.push(0u8);
            buf.extend_from_slice(&(ct.chars.len() as u64).to_le_bytes());
            for &ch in &ct.chars {
                let mut utf8_buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut utf8_buf);
                buf.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
                buf.extend_from_slice(encoded.as_bytes());
            }
        }
        Tokenizer::Bpe(bt) => {
            buf.push(1u8);
            buf.extend_from_slice(&(bt.vocab.len() as u64).to_le_bytes());
            for token_str in &bt.vocab {
                let bytes = token_str.as_bytes();
                buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(bytes);
            }
            buf.extend_from_slice(&(bt.merges.len() as u64).to_le_bytes());
            for &(a, b) in &bt.merges {
                buf.extend_from_slice(&(a as u64).to_le_bytes());
                buf.extend_from_slice(&(b as u64).to_le_bytes());
            }
        }
    }
    // Params
    buf.extend_from_slice(&(params.data.len() as u64).to_le_bytes());
    for &v in &params.data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, &buf).unwrap_or_else(|e| panic!("failed to save {path}: {e}"));
    println!("saved checkpoint to {path} ({} bytes)", buf.len());
}

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, n: usize, path: &str) -> &'a [u8] {
    if *pos + n > data.len() {
        eprintln!("error: checkpoint '{path}' is truncated at offset {}", *pos);
        std::process::exit(1);
    }
    let slice = &data[*pos..*pos + n];
    *pos += n;
    slice
}

fn read_u64(data: &[u8], pos: &mut usize, path: &str) -> u64 {
    u64::from_le_bytes(read_bytes(data, pos, 8, path).try_into().unwrap())
}

fn read_f64(data: &[u8], pos: &mut usize, path: &str) -> f64 {
    f64::from_le_bytes(read_bytes(data, pos, 8, path).try_into().unwrap())
}

fn read_u32(data: &[u8], pos: &mut usize, path: &str) -> u32 {
    u32::from_le_bytes(read_bytes(data, pos, 4, path).try_into().unwrap())
}

fn load_checkpoint(path: &str) -> (GptConfig, Tokenizer, Params) {
    let data = fs::read(path).unwrap_or_else(|e| {
        eprintln!("error: failed to read '{path}': {e}");
        std::process::exit(1);
    });
    let mut pos = 0;

    // Magic
    let magic = read_bytes(&data, &mut pos, 4, path);
    if magic != CHECKPOINT_MAGIC {
        eprintln!("error: '{path}' is not a valid checkpoint (bad magic)");
        std::process::exit(1);
    }

    let version = read_u64(&data, &mut pos, path);
    if version != CHECKPOINT_VERSION {
        eprintln!("error: unsupported checkpoint version {version} in '{path}'");
        std::process::exit(1);
    }
    let n_layer = read_u64(&data, &mut pos, path) as usize;
    let n_embd = read_u64(&data, &mut pos, path) as usize;
    let n_head = read_u64(&data, &mut pos, path) as usize;
    let block_size = read_u64(&data, &mut pos, path) as usize;

    if n_head == 0 {
        eprintln!("error: checkpoint '{path}' has n_head=0");
        std::process::exit(1);
    }

    // Architectural flags
    let act_byte = read_bytes(&data, &mut pos, 1, path)[0];
    let activation = match act_byte {
        0 => Activation::Silu,
        1 => Activation::Relu,
        _ => {
            eprintln!("error: unknown activation {act_byte} in checkpoint '{path}'");
            std::process::exit(1);
        }
    };
    let no_final_norm = read_bytes(&data, &mut pos, 1, path)[0] != 0;
    let no_learnable_gamma = read_bytes(&data, &mut pos, 1, path)[0] != 0;

    // Tokenizer
    let tok_type = read_bytes(&data, &mut pos, 1, path)[0];
    let tokenizer = match tok_type {
        0 => {
            // Char tokenizer
            let n_chars = read_u64(&data, &mut pos, path) as usize;
            let mut chars = Vec::with_capacity(n_chars);
            for _ in 0..n_chars {
                let len = read_u32(&data, &mut pos, path) as usize;
                let bytes = read_bytes(&data, &mut pos, len, path);
                let s = std::str::from_utf8(bytes).unwrap_or_else(|_| {
                    eprintln!("error: invalid utf8 in checkpoint '{path}'");
                    std::process::exit(1);
                });
                chars.push(s.chars().next().unwrap());
            }
            let char_to_id: HashMap<char, usize> =
                chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
            Tokenizer::Char(CharTokenizer { chars, char_to_id })
        }
        1 => {
            // BPE tokenizer
            let n_tokens = read_u64(&data, &mut pos, path) as usize;
            let mut vocab = Vec::with_capacity(n_tokens);
            for _ in 0..n_tokens {
                let len = read_u32(&data, &mut pos, path) as usize;
                let bytes = read_bytes(&data, &mut pos, len, path);
                let s = std::str::from_utf8(bytes).unwrap_or_else(|_| {
                    eprintln!("error: invalid utf8 in checkpoint '{path}'");
                    std::process::exit(1);
                });
                vocab.push(s.to_string());
            }
            let n_merges = read_u64(&data, &mut pos, path) as usize;
            let mut merges = Vec::with_capacity(n_merges);
            for _ in 0..n_merges {
                let a = read_u64(&data, &mut pos, path) as usize;
                let b = read_u64(&data, &mut pos, path) as usize;
                merges.push((a, b));
            }
            let token_to_id: HashMap<String, usize> = vocab
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), i))
                .collect();
            Tokenizer::Bpe(BpeTokenizer {
                vocab,
                token_to_id,
                merges,
            })
        }
        _ => {
            eprintln!("error: unknown tokenizer type {tok_type} in checkpoint '{path}'");
            std::process::exit(1);
        }
    };

    let vocab_size = tokenizer.vocab_size();
    let config = GptConfig {
        n_layer,
        n_embd,
        block_size,
        n_head,
        head_dim: n_embd / n_head,
        vocab_size,
        activation,
        no_final_norm,
        no_learnable_gamma,
        dropout: 0.0,
    };

    // Params
    let n_params = read_u64(&data, &mut pos, path) as usize;

    // Validate n_params matches config expectations
    let e = n_embd;
    let expected_params = vocab_size * e
        + block_size * e
        + vocab_size * e
        + n_layer * (4 * e * e + 4 * e * e + e * 4 * e)
        + (2 + 2 * n_layer) * e;
    if n_params != expected_params {
        eprintln!(
            "error: checkpoint '{path}' has {} params but config expects {}",
            n_params, expected_params
        );
        std::process::exit(1);
    }

    let mut param_data = Vec::with_capacity(n_params);
    for _ in 0..n_params {
        param_data.push(read_f64(&data, &mut pos, path));
    }
    let params = Params {
        data: param_data,
        m: vec![0.0; n_params],
        v: vec![0.0; n_params],
    };

    let tok_label = match tokenizer {
        Tokenizer::Char(_) => "char",
        Tokenizer::Bpe(_) => "bpe",
    };
    println!(
        "loaded checkpoint from {path}: layers={} embd={} heads={} params={} tokenizer={}",
        n_layer, n_embd, n_head, n_params, tok_label
    );

    (config, tokenizer, params)
}

fn compute_val_loss(
    params: &Params,
    config: &GptConfig,
    tokenizer: &Tokenizer,
    val_docs: &[String],
) -> f64 {
    let mut total_loss = 0.0;
    let mut total_tokens = 0usize;
    for doc in val_docs {
        let tokens = tokenizer.encode(doc);
        let n = tokens.len().saturating_sub(1).min(config.block_size);
        if n == 0 {
            continue;
        }
        let mut keys: Vec<Vec<Vec<f64>>> = (0..config.n_layer).map(|_| Vec::new()).collect();
        let mut values: Vec<Vec<Vec<f64>>> = (0..config.n_layer).map(|_| Vec::new()).collect();
        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt_forward_f64(
                &params.data,
                config,
                token_id,
                pos_id,
                &mut keys,
                &mut values,
            );
            // log-softmax for the target
            let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = logits
                .iter()
                .map(|&l| (l - max_val).exp())
                .sum::<f64>()
                .ln()
                + max_val;
            let neg_log_prob = -(logits[target_id] - log_sum_exp);
            total_loss += neg_log_prob;
            total_tokens += 1;
        }
    }
    if total_tokens == 0 {
        return 0.0;
    }
    total_loss / total_tokens as f64
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

    let (config, tokenizer, params) = if let Some(ref path) = args.load {
        // Load from checkpoint — skip training
        load_checkpoint(path)
    } else {
        // Build from dataset
        let mut docs = load_dataset(&args.input);
        let n = docs.len();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            docs.swap(i, j);
        }

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
        let mut params = Params::new(&config, &mut rng, args.init_scale);
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

        // Split into train/val
        let val_count = (docs.len() as f64 * args.val_split) as usize;
        let val_docs: Vec<String> = docs.drain(docs.len() - val_count..).collect();
        let train_docs = docs;

        // Training
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
        let model = GptModel::build(&config);
        let nt = n_trainable(&config, params.len());
        let mut grads_buf = vec![0.0; nt];
        let mut recent_losses = VecDeque::with_capacity(101);
        let mut doc_idx = 0usize;
        let batch_size = args.batch_size;
        let train_len = train_docs.len();
        let mut train_order: Vec<usize> = (0..train_len).collect();
        // Track epoch boundary for re-shuffling and val loss
        let mut docs_seen = 0usize;
        let mut epoch = 0usize;

        for step in 0..args.steps {
            // Zero gradient buffer
            for g in grads_buf.iter_mut() {
                *g = 0.0;
            }

            let mut batch_loss = 0.0;
            for _ in 0..batch_size {
                // Epoch-based re-shuffling
                if docs_seen > 0 && docs_seen.is_multiple_of(train_len) {
                    epoch += 1;
                    // Fisher-Yates shuffle
                    for i in (1..train_len).rev() {
                        let j = rng.random_range(0..=i);
                        train_order.swap(i, j);
                    }
                    // Validation loss at epoch boundary
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
            // Average gradients and loss
            let inv_batch = 1.0 / batch_size as f64;
            for g in grads_buf.iter_mut() {
                *g *= inv_batch;
            }
            batch_loss *= inv_batch;

            adamw_step(&mut params, &config, &adam, &grads_buf, step, args.steps);

            recent_losses.push_back(batch_loss);
            if recent_losses.len() > 100 {
                recent_losses.pop_front();
            }
            let avg: f64 = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
            print!(
                "\rstep {:4} / {:4} | loss {:.4} | avg100 {:.4}",
                step + 1,
                args.steps,
                batch_loss,
                avg
            );
        }
        println!();

        // Final val loss
        if !val_docs.is_empty() {
            let vl = compute_val_loss(&params, &config, &tokenizer, &val_docs);
            println!("final val_loss {:.4}", vl);
        }

        if let Some(ref path) = args.save {
            save_checkpoint(path, &config, &tokenizer, &params);
        }

        (config, tokenizer, params)
    };

    // Inference
    println!("--- generated samples ---");
    for i in 0..args.samples {
        let name = generate_sample(&params, &config, &tokenizer, args.temperature, &mut rng);
        println!("sample {:2}: {}", i + 1, name);
    }
}

// ---------------------------------------------------------------------------
// Section 10: Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
