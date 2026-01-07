-- main.lua - Pure Lua LLaMA (no Torch dependencies)

print("=== Lua LLaMA - Pure Lua Implementation ===\n")

-- Check Lua version
if not string.unpack then
    print("ERROR: Requires Lua 5.3+ for string.unpack")
    print("Please install Lua 5.3 or higher, or use LuaJIT")
    os.exit(1)
end

-- Parse command line arguments
local checkpoint_path = arg[1] or "stories15M.bin"
local tokenizer_path = arg[2] or "tokenizer.bin"
local prompt = arg[3] or "Once upon a time"
local max_tokens = tonumber(arg[4]) or 100
local temperature = tonumber(arg[5]) or 0.9

print("Configuration:")
print("  Checkpoint: " .. checkpoint_path)
print("  Tokenizer:  " .. tokenizer_path)
print("  Prompt:     " .. prompt)
print("  Max tokens: " .. max_tokens)
print("  Temperature:" .. temperature)
print("")

-- Initialize random seed
math.randomseed(os.time())

-- Load modules
local Model = require('model')
local Tokenizer = require('tokenizer')
local gen = require('generate')

-- Load model
print("Loading model...")
local success, model = pcall(Model.new, checkpoint_path)
if not success then
    print("ERROR loading model: " .. tostring(model))
    print("\nTo download the model (60MB):")
    print("  wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin")
    os.exit(1)
end

-- Load tokenizer
print("Loading tokenizer...")
local success, tokenizer = pcall(Tokenizer.new, tokenizer_path, model.config.vocab_size)
if not success then
    print("ERROR loading tokenizer: " .. tostring(tokenizer))
    print("\nTo download the tokenizer (500KB):")
    print("  wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin")
    os.exit(1)
end

print("\n" .. string.rep("=", 60))
print("Ready! Generating text...")
print(string.rep("=", 60) .. "\n")

-- Generate
local start_time = os.clock()
gen.generate(model, tokenizer, prompt, max_tokens, temperature)
local end_time = os.clock()

print(string.rep("=", 60))
print(string.format("Generated in %.2f seconds", end_time - start_time))
print(string.rep("=", 60))