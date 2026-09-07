-- tokenizer.lua - Pure Lua BPE tokenizer (no Torch)

local utils = require('utils')

local Tokenizer = {}
Tokenizer.__index = Tokenizer

function Tokenizer.new(tokenizer_path, vocab_size)
    local self = setmetatable({}, Tokenizer)
    
    self.vocab_size = vocab_size
    self.vocab = {}
    self.vocab_scores = {}
    self.text = {}
    
    local file = io.open(tokenizer_path, "rb")
    if not file then
        error("Cannot open tokenizer file: " .. tokenizer_path)
    end
    
    print("Loading tokenizer from: " .. tokenizer_path)
    
    -- Read max token length
    self.max_token_length = utils.read_int32(file)
    
    -- Read all tokens
    for i = 0, vocab_size - 1 do
        local score = utils.read_float32(file)
        local len = utils.read_int32(file)
        local token = file:read(len)
        
        if not token or #token ~= len then
            error(string.format("Failed to read token %d", i))
        end
        
        self.vocab[i] = token
        self.vocab_scores[i] = score
        local hex = token:match("^<0x(%x%x)>$")
        self.text[i] = hex and string.char(tonumber(hex, 16)) or token
    end
    
    file:close()
    
    -- Assert byte tokens exist for 0-255
    for i = 0, 255 do
        if not self.vocab[i] then
            error(string.format("Missing byte token %d in tokenizer", i))
        end
    end
    
    -- Lookup over the FULL vocab keyed on literal token text (lowest id wins)
    self.merge_lookup = {}
    for id = 0, vocab_size - 1 do
        local t = self.text[id]
        if t and self.merge_lookup[t] == nil then
            self.merge_lookup[t] = id
        end
    end

    print(string.format("Loaded %d tokens (%d merges)", vocab_size, vocab_size - 256))
    return self
end

-- Helper: convert string to byte array
function str_to_bytes(s)
    local bytes = {}
    for i = 1, #s do
        bytes[i] = string.byte(s, i)
    end
    return bytes
end

-- BPE encoding: llama2.c-compatible (BOS + dummy prefix + byte fallback + greedy merge)
function Tokenizer:encode(text, add_bos)
    if add_bos == nil then add_bos = true end
    local tokens = {}
    if add_bos then tokens[#tokens + 1] = 1 end
    if text ~= "" then text = " " .. text end

    local i = 1
    while i <= #text do
        local b = text:byte(i)
        local n = (b < 0x80 and 1) or (b < 0xE0 and 2) or (b < 0xF0 and 3) or 4
        local cp = text:sub(i, i + n - 1)
        local id = self.merge_lookup[cp]
        if id then
            tokens[#tokens + 1] = id
        else
            for k = 1, #cp do tokens[#tokens + 1] = cp:byte(k) + 3 end
        end
        i = i + n
    end

    while true do
        local best_score, best_id, best_idx = -1e10, -1, -1
        for j = 1, #tokens - 1 do
            local merged = self.merge_lookup[self.text[tokens[j]] .. self.text[tokens[j + 1]]]
            if merged and self.vocab_scores[merged] > best_score then
                best_score, best_id, best_idx = self.vocab_scores[merged], merged, j
            end
        end
        if best_idx == -1 then break end
        tokens[best_idx] = best_id
        table.remove(tokens, best_idx + 1)
    end

    return tokens
end

function Tokenizer:decode(token_id)
    if token_id < 0 or token_id >= self.vocab_size then
        return ""
    end
    
    local token = self.vocab[token_id]
    if not token then
        return ""
    end
    
    -- Handle special hex byte tokens like <0x0A>
    if token:match("^<0x%x%x>$") then
        local hex = token:match("<0x(%x%x)>")
        return string.char(tonumber(hex, 16))
    end
    
    -- Replace special space character
    token = token:gsub("▁", " ")
    
    return token
end

function Tokenizer:decode_tokens(tokens)
    local pieces = {}
    for _, token_id in ipairs(tokens) do
        local piece = self:decode(token_id)
        if piece and piece ~= "" then
            table.insert(pieces, piece)
        end
    end
    return table.concat(pieces, "")
end

return Tokenizer