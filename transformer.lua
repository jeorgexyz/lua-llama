-- Lua implementation of the Transformer model

local torch = require 'torch'
local nn = require 'nn'

local Transformer = {}
Transformer.__index = Transformer

function Transformer.new(config)
    local self = setmetatable({}, Transformer)
    self.config = config
    self.token_embedding = nn.LookupTable(config.vocab_size, config.dim)
    self.position_embedding = nn.LookupTable(config.max_seq_len, config.dim)
    
    self.layers = {}
    for _ = 1, config.n_layers do
        table.insert(self.layers, self:create_transformer_block())
    end
    
    self.ln_f = nn.LayerNorm(config.dim)
    self.lm_head = nn.Linear(config.dim, config.vocab_size)
    
    return self
end

function Transformer:create_transformer_block()
    local block = nn.Sequential()
    block:add(self:create_attention_block())
    block:add(self:create_mlp_block())
    return block
end

function Transformer:create_attention_block()
    local block = nn.Sequential()
    block:add(nn.LayerNorm(self.config.dim))
    block:add(self:create_multi_head_attention())
    block:add(nn.CAddTable())  -- Residual connection
    return block
end

function Transformer:create_multi_head_attention()
    local attention = nn.ConcatTable()
    for _ = 1, self.config.n_heads do
        attention:add(self:create_attention_head())
    end
    return nn.Sequential()
        :add(attention)
        :add(nn.JoinTable(3))
        :add(nn.Linear(self.config.dim, self.config.dim))
end

function Transformer:create_attention_head()
    local head = nn.Sequential()
    head:add(nn.Linear(self.config.dim, self.config.dim // self.config.n_heads))
    head:add(nn.MM())  -- Matrix multiplication for attention
    head:add(nn.SoftMax())
    head:add(nn.MM())  -- Matrix multiplication with values
    return head
end

function Transformer:create_mlp_block()
    local block = nn.Sequential()
    block:add(nn.LayerNorm(self.config.dim))
    block:add(nn.Linear(self.config.dim, 4 * self.config.dim))
    block:add(nn.ReLU())
    block:add(nn.Linear(4 * self.config.dim, self.config.dim))
    block:add(nn.CAddTable())  -- Residual connection
    return block
end

function Transformer:forward(idx)
    local B, T = idx:size(1), idx:size(2)
    local x = self.token_embedding(idx) + self.position_embedding(torch.range(1, T):long())
    
    for _, layer in ipairs(self.layers) do
        x = layer:forward(x)
    end
    
    x = self.ln_f(x)
    local logits = self.lm_head(x)
    return logits
end

function Transformer:generate(idx, max_new_tokens)
    for _ = 1, max_new_tokens do
        local logits = self:forward(idx:narrow(2, -self.config.max_seq_len, -1))
        local next_token = torch.multinomial(nn.SoftMax():forward(logits[{-1, -1}]), 1)
        idx = torch.cat(idx, next_token:view(1, 1), 2)
    end
    return idx
end

return Transformer