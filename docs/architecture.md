# Architecture

lua-llama is a pure Lua (5.3+) implementation of LLaMA-style transformer inference. It is intentionally low-level and dependency-free — no Torch, no C extensions — making every step of the inference pipeline readable in plain Lua. The codebase targets the [llama2.c](https://github.com/karpathy/llama2.c) binary format and is designed as an educational reference, not a production runtime.

---

## Module Overview

```
lua-llama/
├── main.lua          # CLI entry point; parses args, wires modules together
├── model.lua         # Binary checkpoint loading and weight storage
├── tokenizer.lua     # BPE tokenizer: loads vocab, encodes, decodes
├── generate.lua      # Forward pass, KV cache, RoPE, SwiGLU, autoregressive loop
├── transformer.lua   # Alternative Torch-based sketch (not used by main.lua)
├── sampler.lua       # Sampling strategies (temperature, top-p, greedy)
├── config.lua        # Default hyperparameter configuration
├── configurator.lua  # Config override/merge helper
└── utils.lua         # Binary I/O helpers, matmul, rmsnorm, softmax, argmax
```

The primary inference path runs through `main.lua → model.lua → tokenizer.lua → generate.lua`, with `utils.lua` providing the math primitives used throughout.

`transformer.lua` is an earlier Torch-based sketch that uses `nn.LookupTable`, `nn.LayerNorm`, and `nn.Sequential`. It is not invoked by `main.lua` and lives in the repo as a reference implementation.

---

## Data Flow

```
CLI args
   │
   ▼
main.lua
   ├─── model.lua ──────► loads .bin checkpoint → config + weights tables
   ├─── tokenizer.lua ──► loads tokenizer.bin → vocab[] + merge_lookup{}
   └─── generate.lua
           ├── tokenizer:encode(prompt)  → token id array
           ├── prefill loop              → forward() for each prompt token
           └── autoregressive loop
                 ├── generate.forward()  → logits[vocab_size]
                 ├── sample / argmax     → next token id
                 └── tokenizer:decode()  → string fragment → stdout
```

---

## Model Loading (`model.lua`)

The `.bin` checkpoint follows the llama2.c format: a 7-field int32 header followed by raw float32 weight arrays laid out in a fixed order.

**Header fields** (read in sequence):

| Field | Meaning |
|---|---|
| `dim` | Embedding / hidden dimension |
| `hidden_dim` | FFN intermediate dimension |
| `n_layers` | Number of transformer blocks |
| `n_heads` | Number of query heads |
| `n_kv_heads` | Number of key/value heads (< n_heads enables GQA/MQA) |
| `vocab_size` | Vocabulary size (**positive** signals a shared classifier; negative means separate `wcls` weights follow) |
| `seq_len` | Maximum sequence length |

**Derived values** computed after reading the header:
- `head_size = dim // n_heads`
- `kv_dim = n_kv_heads * head_size`

**Weight arrays** are stored *tensor-major*: each tensor holds **all layers contiguously**
before the next tensor begins. They are not interleaved per layer. The order is:

1. `token_embedding_table` — shape `(vocab_size, dim)`
2. `rms_att_weight` — all layers, shape `(n_layers, dim)`
3. `wq` — all layers, shape `(n_layers, dim, dim)`
4. `wk` — all layers, shape `(n_layers, kv_dim, dim)`
5. `wv` — all layers, shape `(n_layers, kv_dim, dim)`
6. `wo` — all layers, shape `(n_layers, dim, dim)`
7. `rms_ffn_weight` — all layers, shape `(n_layers, dim)`
8. `w1` — all layers, shape `(n_layers, hidden_dim, dim)` (SwiGLU gate)
9. `w2` — all layers, shape `(n_layers, dim, hidden_dim)` (down-projection)
10. `w3` — all layers, shape `(n_layers, hidden_dim, dim)` (up-projection)
11. `rms_final_weight` — final RMSNorm scale
12. RoPE frequency tables (skipped; frequencies are computed on-the-fly)
13. `wcls` — classifier head weights, present only when `vocab_size` was negative

Reading these interleaved per layer consumes exactly the same number of bytes and so
fails silently, while every tensor after the embedding table lands on the wrong slice
of the file.

`Model:create_run_state()` allocates the activation buffers (`x`, `xb`, `xb2`, `hb`, `hb2`, `q`, `k`, `v`, `logits`) that `generate.lua` reuses across every forward pass.

---

## Tokenizer (`tokenizer.lua`)

Implements Byte-Pair Encoding over the `tokenizer.bin` format.

**Loading:** reads `max_token_length`, then for each of `vocab_size` entries reads a float32 score, an int32 length, and the token bytes. Ids `0`, `1`, `2` are `<unk>`, `<s>`, `</s>`; the 256 single-byte fallback tokens `<0x00>`–`<0xFF>` occupy ids **3–258**. A byte with value `b` therefore maps to token id `b + 3`, not `b`.

**Encoding (`tokenizer:encode`):**
1. Prepends the BOS token (id `1`) and a dummy prefix space, matching llama2.c.
2. Walks the input one UTF-8 codepoint at a time and looks up that codepoint's *literal
   text* in the vocabulary. If it is present, its id is used directly; if not, the
   codepoint falls back to one token per byte at id `byte + 3`.
3. Greedily merges adjacent pairs using a `merge_lookup` hash table
   (`literal_text → token_id`) built over the full vocabulary, for O(1) pair lookups
   instead of scanning it.
4. At each step, picks the merge with the highest score. Repeats until no pair merges.

The lookup is keyed on the token's literal text, so byte-fallback tokens are keyed on the
raw byte they represent rather than on their `<0xNN>` display form. Keying on the display
form means no pair ever matches and the encoder silently degrades to one token per
character.

**Decoding (`tokenizer:decode`):**
- Handles `<0xNN>` hex byte escape tokens.
- Replaces the `▁` SentencePiece space character with a regular space.

---

## Forward Pass (`generate.lua`)

Each call to `forward(model, token, pos, state, key_cache, value_cache, att_buf)` runs one full transformer forward pass for a single token at position `pos`.

### Step-by-step

**1. Token embedding lookup**  
Copy the row `token_embedding_table[token * dim]` into `state.x`.

**2. Per-layer transformer block** (repeated `n_layers` times):

- **Attention pre-norm:** `utils.rmsnorm(xb, x, rms_att_weight[l], dim)`

- **QKV projections:**  
  `matmul(q, xb, wq[l])` — shape `(dim,)`  
  `matmul(k, xb, wk[l])` — shape `(kv_dim,)`  
  `matmul(v, xb, wv[l])` — shape `(kv_dim,)`

- **RoPE (Rotary Position Embedding):**  
  Applied in-place to `q` and `k`. For each head and each pair of dimensions `(2i, 2i+1)`:
  ```
  freq = 1 / 10000^(2i / head_size)
  angle = pos * freq
  [q_2i, q_2i+1] = [q_2i·cos - q_2i+1·sin, q_2i·sin + q_2i+1·cos]
  ```
  The exponent is normalised by `head_size`, not the full embedding `dim` — the rotation
  is defined within each head. With `dim = 288` and `head_size = 48` the two differ by 6×.

- **KV cache write:**  
  Stores `k` and `v` into flat arrays indexed as `layer * seq_len * kv_dim + pos * kv_dim`.

- **Grouped Query Attention (GQA):**  
  Supports `n_kv_heads < n_heads`. Each query head `h` maps to KV head `floor(h * n_kv_heads / n_heads)`. For each query head:
  1. Dot-product scores against all cached keys up to `pos`, scaled by `1/√head_size`.
  2. Softmax over the time dimension.
  3. Weighted sum of cached values → `xb2`.

- **Output projection + residual:**  
  `matmul(xb, xb2, wo[l])`, then `accum(x, xb)` (elementwise add).

- **FFN pre-norm:** `utils.rmsnorm(xb, x, rms_ffn_weight[l], dim)`

- **SwiGLU FFN:**  
  ```
  gate = matmul(xb, w1[l])          -- gate projection
  up   = matmul(xb, w3[l])          -- up-projection
  hb   = SiLU(gate) * up            -- elementwise gated activation
  out  = matmul(hb, w2[l])          -- down-projection
  ```
  SiLU: `x / (1 + exp(-x))`

- **FFN residual:** `accum(x, out)`

**3. Final RMSNorm:** `utils.rmsnorm(x, x, rms_final_weight, dim)`

**4. Classifier projection:**  
Dot product of `x` against `wcls` (or `token_embedding_table` if shared) to produce `logits[vocab_size]`.

---

## Generation Loop (`generate.generate`)

```
1. Encode prompt → token array
2. Truncate to seq_len - 1 if necessary
3. Allocate KV cache: n_layers × seq_len × kv_dim (flat Lua table, zeroed)
4. Prefill: run forward() for each prompt token *except the last*, advancing pos
   (the last prompt token seeds the decode loop; prefilling it too would forward it
    twice, at consecutive positions)
5. Autoregressive decode:
   a. forward(next_token, pos) → logits
   b. if temperature < 0.01: greedy argmax
      else: divide logits by temperature, softmax, multinomial sample
   c. decode token → print to stdout (streaming)
   d. stop on EOS token (id 1 or 2) or reaching seq_len
```

The KV cache is a single flat Lua table. The indexing formula `layer * seq_len * kv_dim + pos * kv_dim + head_offset` avoids nested tables for cache locality (as much as Lua allows).

---

## Math Primitives (`utils.lua`)

All numerical operations called from `generate.lua` live here:

| Function | Description |
|---|---|
| `matmul(out, x, w, rows, cols)` | Dense matrix-vector multiply: `out = w · x` |
| `rmsnorm(out, x, weight, dim)` | RMS layer norm: `out = x / rms(x) * weight` |
| `softmax(arr, size)` | In-place numerically stable softmax |
| `accum(a, b, dim)` | Elementwise add: `a += b` |
| `argmax(arr)` | Returns index of maximum value |
| `sample(probs)` | Multinomial sample from probability array |
| `read_int32(f)` | Reads a little-endian int32 from a file handle |
| `read_float32(f)` | Reads a little-endian float32 |
| `read_float32_array(f, n)` | Reads n floats into a Lua table |

All math is done in pure Lua with no external libraries.

---

## Configuration (`config.lua`, `configurator.lua`)

`config.lua` holds default generation hyperparameters (temperature, max tokens, etc.). `configurator.lua` provides a simple merge utility so command-line arguments or external callers can override defaults without touching the defaults table directly.

---

## Key Design Decisions

**No tensor library.** All weight matrices and activation buffers are plain Lua tables of floats. `matmul` and other operations loop in Lua. This is intentionally slow but maximally transparent.

**Flat KV cache.** Rather than nested tables (`cache[layer][pos][dim]`), the KV cache is a single flat table with manual index arithmetic. This mirrors how C implementations lay out memory and is easier to reason about when reading the code.

**On-the-fly RoPE.** The checkpoint file includes precomputed frequency tables, but they are skipped on load. Frequencies are recomputed each forward pass (`1 / 10000^(2i/dim)`), trading a tiny amount of compute for simplicity.

**GQA/MQA via head mapping.** Grouped Query Attention is supported naturally: the query-to-KV-head mapping `floor(h * n_kv_heads / n_heads)` collapses to standard MHA when `n_kv_heads == n_heads` and to MQA when `n_kv_heads == 1`.

**Shared classifier.** When `vocab_size` is negative in the checkpoint header, the embedding table doubles as the output classifier (`wcls = nil`, use `token_embedding_table`). This halves memory for the output projection with no code branching at inference time.

---

## Relation to llama2.c

This project is a direct Lua port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c). The binary checkpoint format, weight layout, RoPE implementation, and KV cache indexing are all compatible. The `stories15M.bin` and `tokenizer.bin` files included in the repo are the same files used by llama2.c.