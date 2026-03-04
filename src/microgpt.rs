//! Algorithmic core: autograd, tokenizer, model, training, inference, checkpoint.
//!
//! Rewritten from Karpathy's microgpt.py — same architecture, idiomatic Rust.
//! Tape-based autograd instead of pointer graph.

use rand::Rng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

/// Epsilon for RMSNorm numerical stability (used in both training and inference).
const RMSNORM_EPS: f64 = 1e-5;

/// Epsilon for gradient norm clipping to avoid division by zero.
const GRAD_CLIP_EPS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Section 1: Autograd Engine — tape-based (Wengert list)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Idx(usize);

#[derive(Debug)]
pub enum Op {
    Leaf,
    Add(Idx, Idx),
    Mul(Idx, Idx),
    Pow(Idx, f64),
    Log(Idx),
    Exp(Idx),
    Div(Idx, Idx),
    Relu(Idx),
}

#[derive(Debug)]
pub struct Node {
    pub data: f64,
    pub grad: f64,
    pub op: Op,
}

#[derive(Debug)]
pub struct Tape {
    pub nodes: Vec<Node>,
}

impl Tape {
    pub fn with_capacity(cap: usize) -> Self {
        Tape {
            nodes: Vec::with_capacity(cap),
        }
    }

    pub fn leaf(&mut self, data: f64) -> Idx {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Leaf,
        });
        Idx(idx)
    }

    pub fn add(&mut self, a: Idx, b: Idx) -> Idx {
        let data = self.nodes[a.0].data + self.nodes[b.0].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Add(a, b),
        });
        Idx(idx)
    }

    pub fn mul(&mut self, a: Idx, b: Idx) -> Idx {
        let data = self.nodes[a.0].data * self.nodes[b.0].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Mul(a, b),
        });
        Idx(idx)
    }

    pub fn pow(&mut self, a: Idx, exp: f64) -> Idx {
        let data = self.nodes[a.0].data.powf(exp);
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Pow(a, exp),
        });
        Idx(idx)
    }

    pub fn log(&mut self, a: Idx) -> Idx {
        let data = self.nodes[a.0].data.ln();
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Log(a),
        });
        Idx(idx)
    }

    pub fn exp(&mut self, a: Idx) -> Idx {
        let data = self.nodes[a.0].data.exp();
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Exp(a),
        });
        Idx(idx)
    }

    pub fn div(&mut self, a: Idx, b: Idx) -> Idx {
        let data = self.nodes[a.0].data / self.nodes[b.0].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Div(a, b),
        });
        Idx(idx)
    }

    pub fn relu(&mut self, a: Idx) -> Idx {
        let data = self.nodes[a.0].data.max(0.0);
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Relu(a),
        });
        Idx(idx)
    }

    pub fn sum(&mut self, xs: &[Idx]) -> Idx {
        assert!(!xs.is_empty());
        let mut acc = xs[0];
        for &x in &xs[1..] {
            acc = self.add(acc, x);
        }
        acc
    }

    pub fn load_params(&mut self, data: &[f64]) {
        debug_assert!(self.nodes.is_empty());
        self.nodes.extend(data.iter().map(|&d| Node {
            data: d,
            grad: 0.0,
            op: Op::Leaf,
        }));
    }

    pub fn val(&self, idx: Idx) -> f64 {
        self.nodes[idx.0].data
    }

    pub fn grad(&self, idx: Idx) -> f64 {
        self.nodes[idx.0].grad
    }

    pub fn backward(&mut self, root: Idx) {
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

#[derive(Clone, Debug)]
pub struct CharTokenizer {
    pub chars: Vec<char>,
    pub char_to_id: HashMap<char, usize>,
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, usize>,
    pub merges: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub enum Tokenizer {
    Char(CharTokenizer),
    Bpe(BpeTokenizer),
}

impl Tokenizer {
    pub fn from_docs_char(docs: &[String]) -> Self {
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

    pub fn from_docs_bpe(docs: &[String], target_vocab_size: usize) -> Self {
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

    pub fn bos(&self) -> usize {
        match self {
            Tokenizer::Char(ct) => ct.chars.len(),
            Tokenizer::Bpe(bt) => bt.vocab.len(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.bos() + 1
    }

    pub fn encode(&self, doc: &str) -> Vec<usize> {
        let bos = self.bos();
        match self {
            Tokenizer::Char(ct) => {
                let mut tokens = Vec::with_capacity(doc.len() + 2);
                tokens.push(bos);
                for ch in doc.chars() {
                    if let Some(&id) = ct.char_to_id.get(&ch) {
                        tokens.push(id);
                    }
                }
                tokens.push(bos);
                tokens
            }
            Tokenizer::Bpe(bt) => {
                // Start with char-level tokens, skipping unknown characters
                let mut tokens: Vec<usize> = doc
                    .chars()
                    .filter_map(|ch| bt.token_to_id.get(&ch.to_string()).copied())
                    .collect();
                // Apply merges in priority order, reusing buffer to avoid per-merge allocation
                let mut buf = Vec::with_capacity(tokens.len());
                for &(a, b) in &bt.merges {
                    let merged_str = format!("{}{}", bt.vocab[a], bt.vocab[b]);
                    let merged_id = bt.token_to_id[&merged_str];
                    buf.clear();
                    let mut i = 0;
                    while i < tokens.len() {
                        if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
                            buf.push(merged_id);
                            i += 2;
                        } else {
                            buf.push(tokens[i]);
                            i += 1;
                        }
                    }
                    std::mem::swap(&mut tokens, &mut buf);
                }
                let mut result = Vec::with_capacity(tokens.len() + 2);
                result.push(bos);
                result.extend(tokens);
                result.push(bos);
                result
            }
        }
    }

    pub fn decode(&self, token: usize) -> Option<String> {
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    Silu,
    Relu,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LrSchedule {
    Cosine,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TokenizerType {
    Char,
    Bpe,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitScale {
    Flat,
    Scaled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GptConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub activation: Activation,
    pub no_final_norm: bool,
    pub no_learnable_gamma: bool,
    pub dropout: f64,
}

impl GptConfig {
    pub fn estimate_tape_nodes(&self) -> usize {
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

#[derive(Clone, Debug)]
pub struct Params {
    pub data: Vec<f64>,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl Params {
    pub fn new(config: &GptConfig, rng: &mut StdRng, init_scale: InitScale) -> Self {
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

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Section 4: Model on Tape — load params as leaf nodes
// ---------------------------------------------------------------------------

pub struct LayerModel {
    pub attn_wq: Vec<Vec<Idx>>,
    pub attn_wk: Vec<Vec<Idx>>,
    pub attn_wv: Vec<Vec<Idx>>,
    pub attn_wo: Vec<Vec<Idx>>,
    pub mlp_fc1: Vec<Vec<Idx>>,
    pub mlp_fc2: Vec<Vec<Idx>>,
    pub attn_norm_gamma: Vec<Idx>,
    pub mlp_norm_gamma: Vec<Idx>,
}

pub struct GptModel {
    pub wte: Vec<Vec<Idx>>,
    pub wpe: Vec<Vec<Idx>>,
    pub lm_head: Vec<Vec<Idx>>,
    pub layers: Vec<LayerModel>,
    pub initial_norm_gamma: Vec<Idx>,
    pub final_norm_gamma: Vec<Idx>,
}

impl GptModel {
    /// Build index template once — indices are 0..n_params matching Params::data layout.
    pub fn build(config: &GptConfig) -> Self {
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

#[derive(Debug)]
pub struct TapeConstants {
    pub minus_one: Idx,
    pub one: Idx,
    pub eps: Idx,
    pub inv_n_embd: Idx,
}

impl TapeConstants {
    pub fn new(tape: &mut Tape, n_embd: usize) -> Self {
        TapeConstants {
            minus_one: tape.leaf(-1.0),
            one: tape.leaf(1.0),
            eps: tape.leaf(RMSNORM_EPS),
            inv_n_embd: tape.leaf(1.0 / n_embd as f64),
        }
    }
}

pub fn linear(tape: &mut Tape, x: &[Idx], w: &[Vec<Idx>]) -> Vec<Idx> {
    w.iter().map(|wo| dot(tape, wo, x)).collect()
}

pub fn softmax(tape: &mut Tape, logits: &[Idx], c: &TapeConstants) -> Vec<Idx> {
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

pub fn log_softmax(tape: &mut Tape, logits: &[Idx], c: &TapeConstants) -> Vec<Idx> {
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

pub fn silu(tape: &mut Tape, x: Idx, c: &TapeConstants) -> Idx {
    let neg_x = tape.mul(x, c.minus_one);
    let e = tape.exp(neg_x);
    let denom = tape.add(c.one, e);
    let sig = tape.pow(denom, -1.0);
    tape.mul(x, sig)
}

pub fn rmsnorm(tape: &mut Tape, x: &[Idx], gamma: &[Idx], c: &TapeConstants) -> Vec<Idx> {
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

pub fn dot(tape: &mut Tape, a: &[Idx], b: &[Idx]) -> Idx {
    let mut acc = tape.mul(a[0], b[0]);
    for i in 1..a.len() {
        let prod = tape.mul(a[i], b[i]);
        acc = tape.add(acc, prod);
    }
    acc
}

pub fn dropout(tape: &mut Tape, x: &[Idx], p: f64, rng: &mut StdRng) -> Vec<Idx> {
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
pub fn gpt_forward(
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub warmup_steps: usize,
    pub schedule: LrSchedule,
    pub grad_clip: f64,
}

pub fn n_trainable(config: &GptConfig, n_params: usize) -> usize {
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

pub fn forward_backward(
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

pub fn adamw_step(
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
        (adam.grad_clip / (grad_norm + GRAD_CLIP_EPS)).min(1.0)
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

    // AdamW update (use f64 exponent to avoid i32 overflow at >2.1B steps)
    let step_f = (step + 1) as f64;
    for (i, &gi) in grads.iter().enumerate().take(nt) {
        let g = gi * clip_coef;
        params.m[i] = adam.beta1 * params.m[i] + (1.0 - adam.beta1) * g;
        params.v[i] = adam.beta2 * params.v[i] + (1.0 - adam.beta2) * g * g;
        let m_hat = params.m[i] / (1.0 - adam.beta1.powf(step_f));
        let v_hat = params.v[i] / (1.0 - adam.beta2.powf(step_f));
        params.data[i] -=
            lr_t * (m_hat / (v_hat.sqrt() + adam.eps) + adam.weight_decay * params.data[i]);
    }
}

// ---------------------------------------------------------------------------
// Section 7: Inference — tape-free f64 forward pass
// ---------------------------------------------------------------------------

/// Fisher-Yates shuffle of a mutable slice.
pub fn shuffle<T>(slice: &mut [T], rng: &mut StdRng) {
    for i in (1..slice.len()).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

pub fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / total).collect()
}

pub fn weighted_sample(probs: &[f64], rng: &mut StdRng) -> usize {
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

pub fn linear_f64(x: &[f64], w: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = x.len();
    debug_assert_eq!(w.len(), out_dim * in_dim);
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum()
        })
        .collect()
}

pub fn rmsnorm_f64(x: &[f64], gamma: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let ms: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>() / n;
    let scale = 1.0 / (ms + RMSNORM_EPS).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(&xi, &gi)| xi * scale * gi)
        .collect()
}

pub fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

pub fn relu_f64(x: f64) -> f64 {
    x.max(0.0)
}

pub fn gpt_forward_f64(
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

    // Per-layer weight count: wq(e²) + wk(e²) + wv(e²) + wo(e²) = 4e²,
    // fc1(4e·e) = 4e², fc2(e·4e) = 4e² → total 12e²
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

pub fn generate_sample(
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
// Section 9: Model checkpoint — save/load (MessagePack via serde + rmp-serde)
// ---------------------------------------------------------------------------

const CHECKPOINT_MAGIC: &[u8; 4] = b"MGPT";

#[derive(Debug)]
pub struct Checkpoint {
    pub config: GptConfig,
    pub tokenizer: Tokenizer,
    pub params: Params,
    pub completed_step: usize,
    pub adam: AdamConfig,
    pub batch_size: usize,
}

#[derive(Serialize, Deserialize)]
enum TokenizerData {
    Char {
        chars: Vec<char>,
    },
    Bpe {
        vocab: Vec<String>,
        merges: Vec<(usize, usize)>,
    },
}

impl TokenizerData {
    fn from_tokenizer(tok: &Tokenizer) -> Self {
        match tok {
            Tokenizer::Char(ct) => TokenizerData::Char {
                chars: ct.chars.clone(),
            },
            Tokenizer::Bpe(bt) => TokenizerData::Bpe {
                vocab: bt.vocab.clone(),
                merges: bt.merges.clone(),
            },
        }
    }

    fn into_tokenizer(self) -> Tokenizer {
        match self {
            TokenizerData::Char { chars } => {
                let char_to_id = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
                Tokenizer::Char(CharTokenizer { chars, char_to_id })
            }
            TokenizerData::Bpe { vocab, merges } => {
                let token_to_id = vocab
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
        }
    }
}

#[derive(Serialize, Deserialize)]
struct CheckpointData {
    config: GptConfig,
    adam: AdamConfig,
    tokenizer: TokenizerData,
    params_data: Vec<f64>,
    params_m: Vec<f64>,
    params_v: Vec<f64>,
    completed_step: usize,
    batch_size: usize,
}

pub fn save_checkpoint(
    path: &str,
    config: &GptConfig,
    adam: &AdamConfig,
    tokenizer: &Tokenizer,
    params: &Params,
    completed_step: usize,
    batch_size: usize,
) {
    let data = CheckpointData {
        config: config.clone(),
        adam: adam.clone(),
        tokenizer: TokenizerData::from_tokenizer(tokenizer),
        params_data: params.data.clone(),
        params_m: params.m.clone(),
        params_v: params.v.clone(),
        completed_step,
        batch_size,
    };
    let mut buf = CHECKPOINT_MAGIC.to_vec();
    buf.extend(rmp_serde::to_vec_named(&data).expect("failed to serialize checkpoint"));
    fs::write(path, &buf).unwrap_or_else(|e| panic!("failed to save {path}: {e}"));
    println!("saved checkpoint to {path} ({} bytes)", buf.len());
}

pub fn load_checkpoint(path: &str) -> Checkpoint {
    let raw = fs::read(path).unwrap_or_else(|e| {
        eprintln!("error: failed to read '{path}': {e}");
        std::process::exit(1);
    });
    if raw.len() < 4 || &raw[..4] != CHECKPOINT_MAGIC {
        eprintln!("error: '{path}' is not a valid checkpoint (bad magic)");
        std::process::exit(1);
    }
    let data: CheckpointData = rmp_serde::from_slice(&raw[4..]).unwrap_or_else(|e| {
        eprintln!("error: failed to deserialize checkpoint '{path}': {e}");
        std::process::exit(1);
    });

    let tokenizer = data.tokenizer.into_tokenizer();
    // Recompute derived fields
    let mut config = data.config;
    config.head_dim = config.n_embd / config.n_head;
    config.vocab_size = tokenizer.vocab_size();

    let params = Params {
        data: data.params_data,
        m: data.params_m,
        v: data.params_v,
    };

    let tok_label = match tokenizer {
        Tokenizer::Char(_) => "char",
        Tokenizer::Bpe(_) => "bpe",
    };
    println!(
        "loaded checkpoint from {path}: layers={} embd={} heads={} params={} tokenizer={} step={}",
        config.n_layer,
        config.n_embd,
        config.n_head,
        params.data.len(),
        tok_label,
        data.completed_step
    );

    Checkpoint {
        config,
        tokenizer,
        params,
        completed_step: data.completed_step,
        adam: data.adam,
        batch_size: data.batch_size,
    }
}

pub fn compute_val_loss(
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
            if !max_val.is_finite() {
                continue;
            }
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
