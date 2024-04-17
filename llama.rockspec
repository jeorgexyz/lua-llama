package = "llama"
version = "1.0-1"
source = {
   url = "https://example.com/llama.tar.gz",
   branch = "master"
}
description = {
   summary = "Lua LLaMA Transformer Model",
   detailed = [[
     This is a Lua implementation of the LLaMA Transformer model.
   ]],
   homepage = "",
   license = "MIT"
}
dependencies = {
   "lua >= 5.1, < 5.5",
   "torch >= 1.7.0",
   "nn >= 1.0.0",
   "paths >= 0.3.0"
}
build = {
   type = "builtin",
   modules = {
      ["llama.inference"] = "inference.lua",
      ["llama.model"] = "model.lua",
      ["llama.tokenizer"] = "tokenizer.lua",
      ["llama.utils"] = "utils.lua"
   }
}