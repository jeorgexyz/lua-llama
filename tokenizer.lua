local Tokenizer = {}
Tokenizer.__index = Tokenizer

function Tokenizer.new(path, vocab_size)
    local self = setmetatable({}, Tokenizer)
    self.vocab_size = vocab_size
    self:load(path)
    return self
end

function Tokenizer:load(path)
    -- Load tokenizer data from file
    print("Loading tokenizer from " .. path)
    -- Example loading function, update with actual implementation
    self.vocab = torch.load(path)
end

function Tokenizer:encode(text, bos, eos)
    -- Encode text to token IDs
    local tokens = {}
    if bos then table.insert(tokens, 1) end
    for word in text:gmatch("%S+") do
        table.insert(tokens, tonumber(word) or self.vocab[word] or #self.vocab)
    end
    if eos then table.insert(tokens, 2) end
    return tokens
end

function Tokenizer:decode(prev_token, token)
    -- Decode token IDs to text
    return self.vocab[token] .. " "
end

return Tokenizer
