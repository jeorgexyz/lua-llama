# Lua Llama
<p align="center">
  <img width="350px" src="./assets/lua-llama.png" alt="Lua Llama">
</p>


A Lua port of Andrej Karpathy's llama2.c 


While Lua wouldn't be my first choice in language to do such project (I only did it because I thought the name Lua Llama sounded cool ), it still was a good learning experience.

## Installation
1. Clone this repository: git clone 


2. Install the required Lua dependencies using Luarocks:

3. Run: main.lua

To run the script, use the following command: main.lua



## Models

Just like the original, we can load any huggingface models that use the LLama 2 architecture.

## Training

## Custom Tokenizers

## Performance

| system                  | model           | llama2.c      | llama.cpp          | llama2.go[^simple] | llama2.go[^fast] |
| ------------------------| --------------- | ------------: | -----------------: | -----------------: | ---------------: |
| Apple M1 Max 10CPU 64GB | stories110M     |  101.84 tok/s |                    |        10.47 tok/s |      39.28 tok/s |  
| Apple M1 Max 10CPU 64GB | llama2_7b       |    1.83 tok/s |        20.36 tok/s |                    |       0.87 tok/s | 
| Apple M1 Max 10CPU 64GB | llama2_13b      |    (segfault) |        11.71 tok/s |                    |       0.38 tok/s |

## Platforms

## Tests
