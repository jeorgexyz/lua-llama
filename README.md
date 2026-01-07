# Lua LlaMA

[![Lua Llama](https://github.com/jeorgexyz/lua-llama/raw/main/assets/lua-llama.png)](https://github.com/jeorgexyz/lua-llama/blob/main/assets/lua-llama.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Lua port of Andrej Karpathy's llama2.c

Lua LLaMA is a lightweight implementation of the LLaMA language model architecture, enabling inference directly in Lua.

## Requirements

* Lua 5.1 or LuaJIT
* LuaRocks
* LuaUnit (for testing)
* *Optional:* Torch (only if using legacy tensor operations)

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/jeorgexyz/lua-llama.git](https://github.com/jeorgexyz/lua-llama.git)
   cd lua-llama