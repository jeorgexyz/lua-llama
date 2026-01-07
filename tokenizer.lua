-- tokenizer.lua - Pure Lua BPE tokenizer (no Torch)

local utils = require('utils')

local Tokenizer = {}
Tokenizer.__index = Tokenizer

function Tokenizer.new(tokenizer_path, vocab_size)
    local self = setmetatable({}, Tokenizer)
    
    self.vocab_size = vocab_size
    self.vocab = {}
    self.vocab_scores = {}
    
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
    end
    
    file:close()
    
    -- Assert byte tokens exist for 0-255
    for i = 0, 255 do
        if not self.vocab[i] then
            error(string.format("Missing byte token %d in tokenizer", i))
        end
    end
    
    -- Build merge lookup map for O(1) lookup (instead of O(vocab_size))
    self.merge_lookup = {}
    for id = 256, vocab_size - 1 do
        local token_str = self.vocab[id]
        if token_str then
            self.merge_lookup[token_str] = id
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

-- BPE encoding: greedily merge byte pairs
function Tokenizer:encode(text)
    -- Start with raw bytes
    local tokens = {}
    
    -- Convert text to UTF-8 bytes (tokens 0-255)
    for i = 1, #text do
        tokens[i] = string.byte(text, i)
    end
    
    -- Greedily merge tokens based on vocab scores using O(1) lookup
    while #tokens > 1 do
        local best_score = -1e10
        local best_id = -1
        local best_idx = -1
        
        -- Find best merge using lookup map
        for i = 1, #tokens - 1 do
            -- Concatenate this pair
            local pair_str = self.vocab[tokens[i]] .. self.vocab[tokens[i+1]]
            
            -- O(1) lookup instead of O(vocab_size) scan
            local merge_id = self.merge_lookup[pair_str]
            if merge_id then
                local score = self.vocab_scores[merge_id]
                if score > best_score then
                    best_score = score
                    best_id = merge_id
                    best_idx = i
                end
            end
        end
        
        -- No more merges found
        if best_idx == -1 then
            break
        end
        
        -- Merge the best pair efficiently (avoid table.remove)
        tokens[best_idx] = best_id
        -- Shift remaining tokens left
        for i = best_idx + 1, #tokens - 1 do
            tokens[i] = tokens[i + 1]
        end
        tokens[#tokens] = nil
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