# Lua LlaMA

<p align="center">
  <img width="350px" src="./assets/lua-llama.png" alt="Lua Llama">
  </p>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


A **pure Lua (5.3+) implementation of LLaMA-style transformer inference**.  
Runs CPU-only inference directly from binary model and tokenizer files.

> This is a low-level implementation intended for learning, experimentation, and clarity. It is not a production inference runtime.
---

## Requirements

- Lua **5.3+** (tested with Lua 5.4)
- A model checkpoint and tokenizer in [llama2.c](https://github.com/karpathy/llama2.c) binary format

Both are already checked into this repo — `stories15M.bin` (the TinyStories 15M
checkpoint) and `tokenizer.bin` — so it runs with no downloads.

---

## Usage

```bash
lua54 main.lua <model.bin> <tokenizer.bin> "<prompt>" <max_tokens> <temperature>
```

## Arguments

<model.bin> – Path to a compatible model checkpoint

<tokenizer.bin> – Path to a compatible tokenizer

"<prompt>" – Input prompt

<max_tokens> – Number of tokens to generate

<temperature> – Sampling temperature

0.0 = greedy / deterministic

0.7–1.0 = more random


## Example
```bash
lua54 main.lua stories15M.bin tokenizer.bin "Once upon a time" 100 0.8
```

## Example Output

At temperature `0.0` (greedy, deterministic):

```
$ lua54 main.lua stories15M.bin tokenizer.bin "Once upon a time" 90 0.0

Config: dim=288, layers=6, heads=6, kv_heads=6, vocab=32000, seq_len=256
Prompt tokens: 5

Once upon a time, there was a little girl named Lily. She loved to play outside
in the sunshine. One day, she saw a big, red ball in the sky. It was the sun!
She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried
to jump and reach it, but she couldn't. Then, she had an idea. She would

Generated in 24.54 seconds
```

Greedy decoding is deterministic, so this output is reproducible — and it matches
what `llama2.c` produces from the same checkpoint and prompt, which is how the
implementation is verified. Throughput is roughly **3–4 tokens/sec** on CPU in pure
Lua; the goal is a readable reference implementation, not speed.
## Speculative Decoding

A small draft model proposes several tokens; the large target model verifies them all
in one round and accepts the longest prefix it agrees with. The emitted text is
identical to running the target alone — only the number of target invocations changes.

```bash
lua54 main_speculative.lua <target.bin> <draft.bin> tokenizer.bin "<prompt>" <max_tokens> <temperature> [lookahead]
```

A 15M draft proposing for a 42M target accepts 82% of its proposals, so each target
round yields 4.29 tokens instead of one:

```
Draft proposals:     46 accepted / 56 proposed (82.1%)
Target rounds:       14 (vs 60 for one-token-at-a-time)
Tokens per round:    4.29
```

Passing the same checkpoint as both target and draft is the correctness check —
acceptance is then exactly 100%. Note that wall-clock time does **not** improve in
pure Lua: the speedup requires evaluating a round's positions as one batched matmul,
which needs BLAS. `target rounds` is the metric that transfers to a real runtime.
See [`examples/`](examples/) for full transcripts and the reasoning.

## Project Structure

```
lua-llama/
├── main.lua        # CLI entry point
├── model.lua       # Model loading and weight parsing
├── tokenizer.lua   # Tokenizer loading and encode/decode
├── generate.lua    # Forward pass, KV cache, sampling
├── speculative.lua # Speculative decoding: draft proposes, target verifies
├── utils.lua       # Binary IO and math utilities
```

## License

MIT
