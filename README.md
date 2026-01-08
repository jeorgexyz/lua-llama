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
- A compatible model checkpoint (`.bin`)
- A compatible tokenizer (`.bin`)

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
## Project Structure

```
lua-llama/
├── main.lua        # CLI entry point
├── model.lua       # Model loading and weight parsing
├── tokenizer.lua   # Tokenizer loading and encode/decode
├── generate.lua    # Forward pass, KV cache, sampling
├── utils.lua       # Binary IO and math utilities
```

## License

MIT
