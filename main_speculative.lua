-- main_speculative.lua - CLI entry point for speculative decoding
--
-- Usage:
--   lua54 main_speculative.lua <target.bin> <draft.bin> <tokenizer.bin> "<prompt>" <max_tokens> <temperature> [lookahead]
--
-- Passing the same checkpoint as both target and draft is the correctness check:
-- acceptance should be 100% and the output identical to main.lua.

print("=== Lua LLaMA - Speculative Decoding ===\n")

if not string.unpack then
    print("ERROR: Requires Lua 5.3+ for string.unpack")
    os.exit(1)
end

local target_path = arg[1] or "stories15M.bin"
local draft_path = arg[2] or target_path
local tokenizer_path = arg[3] or "tokenizer.bin"
local prompt = arg[4] or "Once upon a time"
local max_tokens = tonumber(arg[5]) or 100
local temperature = tonumber(arg[6]) or 0.0
local lookahead = tonumber(arg[7]) or 4

print("Configuration:")
print("  Target:      " .. target_path)
print("  Draft:       " .. draft_path)
print("  Tokenizer:   " .. tokenizer_path)
print("  Prompt:      " .. prompt)
print("  Max tokens:  " .. max_tokens)
print("  Temperature: " .. temperature)
print("  Lookahead:   " .. lookahead)
print("")

math.randomseed(os.time())

local Model = require('model')
local Tokenizer = require('tokenizer')
local speculative = require('speculative')

local function load_model(path, label)
    print("Loading " .. label .. " model...")
    local ok, model = pcall(Model.new, path)
    if not ok then
        print("ERROR loading " .. label .. " model: " .. tostring(model))
        os.exit(1)
    end
    return model
end

local target = load_model(target_path, "target")
local draft = (draft_path == target_path) and target or load_model(draft_path, "draft")

if draft_path == target_path then
    print("Draft and target are the same checkpoint - running the correctness check.")
end

print("Loading tokenizer...")
local ok, tokenizer = pcall(Tokenizer.new, tokenizer_path, target.config.vocab_size)
if not ok then
    print("ERROR loading tokenizer: " .. tostring(tokenizer))
    os.exit(1)
end

if draft.config.vocab_size ~= target.config.vocab_size then
    print("ERROR: draft and target must share a vocabulary")
    print(string.format("  target vocab=%d, draft vocab=%d",
        target.config.vocab_size, draft.config.vocab_size))
    os.exit(1)
end

print("\n" .. string.rep("=", 60))
print("Ready! Generating with speculative decoding...")
print(string.rep("=", 60) .. "\n")

local start_time = os.clock()
local stats = speculative.generate(
    target, draft, tokenizer, prompt, max_tokens, temperature, lookahead)
local elapsed = os.clock() - start_time

local acceptance = stats.proposed > 0 and (stats.accepted / stats.proposed * 100) or 0
local per_round = stats.target_rounds > 0 and (stats.emitted / stats.target_rounds) or 0

print(string.rep("=", 60))
print(string.format("Emitted %d tokens in %.2f seconds", stats.emitted, elapsed))
print(string.format("Draft proposals:     %d accepted / %d proposed (%.1f%%)",
    stats.accepted, stats.proposed, acceptance))
print(string.format("Target rounds:       %d (vs %d for one-token-at-a-time)",
    stats.target_rounds, stats.emitted))
print(string.format("Tokens per round:    %.2f", per_round))
print(string.format("Forward passes:      %d target, %d draft",
    stats.target_forwards, stats.draft_forwards))
print(string.rep("=", 60))
print("")
print("Target rounds is the metric a batched runtime would convert into speed: each")
print("round evaluates block+1 positions in one pass over the target weights. This")
print("implementation runs them sequentially in pure Lua, so wall-clock time does not")
print("improve here - the arithmetic per position is unchanged.")
