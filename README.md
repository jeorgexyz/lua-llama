# Lua LlaMA
<p align="center">
  <img width="350px" src="./assets/lua-llama.png" alt="Lua Llama">
</p>
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status (GitHub Actions)](https://img.shields.io/github/workflow/status/user/repo/workflow-name)](https://github.com/user/repo/actions)


A Lua port of Andrej Karpathy's llama2.c 


While Lua wouldn't be my first choice in language to do such project (I only did it because I thought the name Lua LLaMA sounded cool ), it still was a good learning experience.

Lua LLaMA is a Lua-based implementation of the LLaMA language model, a large-scale transformer-based model developed by Meta AI.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Generation](#generation)
  - [Chat Mode](#chat-mode)
- [Model and Training](#model-and-training)
- [Custom Tokenizers](#custom-tokenizers)
- [Performance Optimization](#performance-optimization)
- [Testing](#testing)
- [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)

## Introduction
The Lua LLaMA project aims to provide a Lua-based implementation of the LLaMA model, enabling users to generate text and engage in interactive conversations using this powerful language model. This project is intended to be a self-contained and portable solution, leveraging the Lua programming language and the Torch framework.

## Requirements
To run the Lua LLaMA project, you'll need the following dependencies:
- Lua (version 5.1 or higher)
- Torch (the Lua scientific computing framework)
- LuaRocks (the Lua package manager)

## Installation
1. Clone the Lua LLaMA repository:
   ```
   git clone https://github.com/your-username/lua-llama.git
   cd lua-llama
   ```
2. Install the required Lua packages using LuaRocks:
   ```
   luarocks install --deps-mode=none
   ```
3. Download the LLaMA model checkpoint and tokenizer files. You can obtain these files by following the instructions provided by the LLaMA project.
4. Update the `config.lua` file with the paths to your LLaMA model checkpoint and tokenizer files.

## Usage
The Lua LLaMA project provides two main modes of operation: **Generation** and **Chat Mode**.

### Generation
To generate text using the Lua LLaMA model, run the following command:
```
lua main.lua path/to/model.bin -n 256 -i "Once upon a time"
```
This will generate 256 tokens starting from the provided prompt "Once upon a time".

### Chat Mode
To engage in an interactive chat session with the Lua LLaMA model, run the following command:
```
lua main.lua path/to/model.bin -m chat -i "Hello" -y "Hi there, how can I assist you today?"
```
This will start a chat session, where you can provide user prompts, and the model will generate responses.

## Model and Training
The Lua LLaMA implementation uses the same transformer-based architecture as the original LLaMA model you can find on huggingface. While this project does not include the training code, you can use the provided model checkpoint files to run the inference and generation tasks.

In the future, I plan to add support for fine-tuning the LLaMA model on custom datasets, enabling users to adapt the model to their specific use cases.


## Training

## Custom Tokenizers
The Lua LLaMA project includes a tokenizer implementation that can handle the default LLaMA tokenizer. However, you can also integrate custom tokenizers by modifying the `Tokenizer` class and the corresponding functions in the codebase.

This allows you to experiment with different tokenization strategies, such as using Byte Pair Encoding (BPE) or WordPiece, to potentially improve the model's performance on your specific use case.

## Performance Optimization
The Lua LLaMA implementation has been designed with performance in mind, leveraging the efficiency of the Lua and Torch ecosystems. However, there are always opportunities for further optimization, such as:
- Exploring the use of Lua JIT (Just-In-Time) compilation to improve runtime performance.
- Investigating the use of GPU acceleration for the model's forward pass.
- Optimizing memory usage and caching strategies.


## Configuration
The behavior of the Lua LLaMA model can be configured through the `config.lua` file. This file contains various hyperparameters, such as the model dimensions, number of layers, number of heads, and more. Adjust these values as needed to suit your requirements.

## Testing
To ensure the correctness and reliability of the Lua LLaMA implementation, I have integrated a comprehensive test suite. The tests cover various aspects of the model, tokenizer, and utility functions.

You can run the tests using the following command:
```
luaunit test/
```
