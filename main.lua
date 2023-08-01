-- Jeorge D. Anderson II
-- inference for Llama-2 Transformer model in Lua
-- Lua does not support multi-threading or direct memory management like C does, so I'll have to use C with Lua 


require 'torch'


--Transformer model

Config = {
    dim = nil, -- tranformer dimension
    hidden_dim = nil, -- for ffn layers
    n_layers = nil, -- number of query heads
    n_kv_heads = nil, -- number of key/value heads (can be < query heads because of multiquery)
    vocab_size = nil, -- vocab size, (256)
    seq_len = nil -- max sequence length
}

TransformerWeights = {
    -- token embeddings table
    token_embeddings_table = nil, --(vocab_size, dim)
    -- weights for rmsnorms
    rms_att_weight = nil, -- (layer, dim) rmsnorm weights
    rms_ffn_weight = nil, -- (layer, dim)
    -- weights for matmuls. note dim == n_heads * head_size
    wq = nil, -- (layer, dim, n_heads * head_size)
    wk = nil, -- (layer, dim, n_kv_heads * head_size)
    wv = nil, -- (layer, dim, n_kv_heads * head_size)
    wo = nil, -- (layer, n_heads * head_size, dim)
    -- weights for ffn
    w1 = nil, -- (layer, hidden_dim, dim)
    w2 = nil, -- (layer, dim, hidden_dim)
    w3 = nil, -- (layer, hidden_dim, dim)
    -- final rmsnorm
    rms_final_weight = nil, -- (dim,)
    -- (optional) classifier weights for the logits, on the last layer
    wcls = nil
}

RunState = {
    -- current wave of activations
    x = nil, -- activation at current time stamp (dim,)
    xb = nil, -- same, but inside a residual branch (dim,)
    xb2 = nil, -- an additional buffer just for convenience (dim,)
    hb = nil, -- buffer for hidden dimension in the ffn (hidden_dim,)
    hb2 = nil, -- buffer for hidden dimension in the ffn (hidden_dim,)
    q = nil, -- query (dim,)
    k = nil, -- key (dim,)
    v = nil, -- value (dim,)
    att = nil, -- buffer for scores/attention values (n_heads, seq_len)
    logits = nil, -- output logits
    -- kv cache
    key_cache = nil,   -- (layer, seq_len, dim)
    value_cache = nil -- (layer, seq_len, dim)
}

Transformer = {
    config = Config, -- the hyperparameters of the architecture (the blueprint)
    weights = TransformerWeights, -- the weights of the model
    state = RunState -- buffers for the "wave" of activations in the forward pass
    -- some more state needed to properly clean up the memory mapping (sigh)
    fd = nil, -- file descriptor for memory mapping
    data = nil, -- memory mapped data pointer
    file_size = nil -- size of the checkpoint file in bytes
}

function malloc_run_state(s, p)
    -- we calloc instead of malloc to keep valgrind happy
    local kv_dim = (p.dim * p.n_kv_heads) / p.n_heads
    s.x = {} -- activation at current time stamp (dim,)
    s.xb = {} -- same, but inside a residual branch (dim,)
    s.xb2 = {} -- an additional buffer just for convenience (dim,)
    s.hb = {} -- buffer for hidden dimension in the ffn (hidden_dim,)
    s.hb2 = {} -- buffer for hidden dimension in the ffn (hidden_dim,)
    s.q = {} -- query (dim,)
    s.key_cache = {} -- (layer, seq_len, dim)
    s.value_cache = {} -- (layer, seq_len, dim)
    s.att = {} -- buffer for scores/attention values (n_heads, seq_len)
    s.logits = {} -- output logits
end

function free_run_state(s)
    s.x = nil
    s.xb = nil
    s.xb2 = nil
    s.hb = nil
    s.hb2 = nil
    s.q = nil
    s.att = nil
    s.logits = nil
    s.key_cache = nil
end

-- Inference for Llama-2 Transformer model in Lua

-- This is a helper function to allocate the necessary weight matrices for a transformer model and return pointers to them in a struct. The modular design allows customizing different components like number of layers, heads, etc.

function memory_map_weights(w, p, ptr, shared_weights)
    local head_size = p.dim /p.n_heads

    local n_layers = p.n_layers
    w.token_embedding_table = ptr
    ptr = ptr + p.vocab_size * p.dim
    w.rms_att_weight = ptr
    ptr. = ptr + n_layers * p.dim
    w.wq = ptr
    ptr = ptr + n_layers * p.dim * (p.n_heads * head_size)
    w.wk = ptr
    ptr = ptr + n_layers * p.dim (p.n_kv_heads * head_size)
    w.wo = ptr
    ptr = ptr + n_layers * (p.n_heads * head_size) * p.dim
    w.rms_ffn_weight = ptr
    ptr = ptr + n_layers * p.dim
    w.w1 = ptr
    ptr = ptr + n_layers * p.dim * p.hidden_dim
    w.w2 = ptr
    ptr = ptr + n_layers * p.hidden_dim * p.dim
    w.w3 = ptr
    ptr = ptr + n_layers * p.dim * p.hidden_dim
    w.rms_final_weight = ptr
    ptr = ptr + p.dim
    ptr = ptr + p.seq_len * head_size / 2 -- skip what used to be freq_cis_real (for RoPE)
    ptr = ptr + p.seq_len * head_size / 2 -- skip what used to be freq_cis_imag (for RoPE)
    w.wcls = shared_weights and w.token_embedding_table or ptr
end

function read_checkpoint(checkpoint, config, weights)
    local file = io.open(checkpoint, "rb")
    if not file then
        print("Couldn't open file " .. checkpoint)
        os.exit()
    end
    -- read in the config header
    config = file:read("*all")
    -- figure out the file size
    file:seek("end") -- move file pointer to end of file
    local file_size = file:seek() -- get the file size, in bytes
    file:close()
    -- memory map the Transformer weights into the data pointer
    local data = mmap(checkpoint, file_size) -- mmap is not a built-in function in Lua. You would need to use a library or write your own function.
    local weights_ptr = data + #config
    memory_map_weights(weights, config, weights_ptr, shared_weights)
end

function build_transformer(t, checkpoint_path)
    -- read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, t.config, t.weights)
    -- allocate the RunState buffers
    malloc_run_state(t.state, t.config)
end

function free_transformer(t)
    -- close the memory mapping
    if t.data then munmap(t.data, t.file_size) end -- munmap is not a built-in function in Lua. You would need to use a library or write your own function.
    -- free the RunState buffers
    free_run_state(t.state)
end

-- neutral net blocks

function rmsnorm(o, x, weight, size)
    -- calculates sum of squares
    local ss = 0.0
    for j = 1, size do
        ss = ss + x[j] * x[j]
    end 
    ss = ss / size
    ss = ss + 1e-5
    ss = 1.0 / math.sqrts(ss)
    -- normalize and scale
    for j = 1, size do
        o[j] = weight[j] * (ss * x[j])
    end
end

function softmax(x, size)
    -- find max value (for numerical stability)
    local max_val = x[1]
    for i = 2, size do
        if x[i] > max_val then
            max_val = x[i]
        end
    end
    -- exp and sum
    local sum = 0.0
    for i = 1, size do
        x[i] = math.exp(x[i] - max_val)
        sum = sum + x[i]
    end
    -- normalize
    for i = 1, size do
        x[i] = x[i] / sum
    end
end

function matmul(xout, x, w, n, d)
    -- W (d,n) @ x (n,) -> xout (d,)
    -- by far the most amount of time is spent inside this little function
    for i = 1, d do
        local val = 0.0
        for j = 1, n do
            val = val + w[(i - 1) * n + j] * x[j]
        end
        xout[i] = val
    end
end

function forward(transformer, token, pos)
    -- a few convenience variables
    local p = transformer.config
    local w = transformer.weights
    local s = transformer.state
    local x = s.x
    local dim = p.dim
    local kv_dim = (p.dim * p.n_kv_heads) / p.n_heads
    local kv_mul = p.n_heads / p.n_kv_heads -- integer multiplier of the kv sharing in multiquery
    local hidden_dim =  p.hidden_dim
    local head_size = dim / p.n_heads

    -- copy the token embedding into x
    local content_row = w.token_embedding_table[token]
    for i = 1, dim do
        x[i] = content_row[i]
    end

    -- forward all the layers
    for l = 1, p.n_layers do
        -- attention rmsnorm
        rmsnorm(s.xb, x, w.rms_att_weight[l], dim)

        -- key and value point to the kv cache
        local loff = l * p.seq_len * kv_dim -- kv cache layer offset for convenience
        s.k = s.key_cache[loff + pos * kv_dim]
        s.v = s.value_cache[loff + pos * kv_dim]

        -- qkv matmuls for this position
        matmul(s.q, s.xb, w.wq[l], dim, dim)
        matmul(s.k, s.xb, w.wk[l], dim, kv_dim)
        matmul(s.v, s.xb, w.wv[l], dim, kv_dim)

        -- RoPE relative positional encoding: complex-valued rotate q and k in each head
        for i = 1, dim, 2 do
            local head_dim = i % head_size
            local freq = 1.0 / math.pow(10000.0, head_dim / head_size)
            local val = pos * freq
            local fcr = math.cos(val)
            local fci = math.sin(val)
            local rotn = i < kv_dim and 2 or 1 -- how many vectors? 2 = q & k, 1 = q only
            for v = 1, rotn do
                local vec = v == 1 and s.q or s.k -- the vector to rotate (query or key)
                local v0 = vec[i]
                local v1 = vec[i + 1]
                vec[i] = v0 * fcr - v1 * fci
                vec[i + 1] = v0 * fci + v1 * fcr
            end
        end
    end
end

-- multihead attention

for h = 1, p.n_heads do
    -- get the query vector for this head
    local q = s.q[h * head_size]

    -- attention scores for this head
    local att = s.att[h * p.seq_len]
    -- iterate over all timesteps (including current one)
    for t=1, pos do
        local k = s.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size]
        -- calculate the attention score as the dot product of q and k
        local score = 0.0
        for i = 1, head_size do
            score = score + q[i] * k[i]
        end
        score = score / math.sqrt(head_size)
        -- save the score to the attention buffer
        att[t] = score
    end

    -- softmax the scores to get attention weights

    softmax(att, pos + 1)

    -- weighted sum of the values, store back into xb
    local xb = s.xb[h * head_size]
    for i = 1, head_size do
        xb[i] = 0
    end 
    for t =1, pos do
        -- get the value vector for this head and at this timestep
        local v = s.value_cache[loff + t * kv_dim + (h / kv_mul) * head_size]
        -- get the attention weight for this timestep
        local a = att[t]
        -- accumulate the weighted value into xb
        for i = 1, head_size do
            xb[i] = xb[i] + a * v[i]
        end
    end
end

-- final matmul to get the output of the attention

matmul(s.xb2, s.xb, w.wo[1], dim, dim)

-- residual connection back into x
for i = 1, dim do
    x[i] = x[i] + s.xb2[i]
end

-- ffn rmsnorm
rmsnorm(s.xb, x w.rms_ffn_weight[1], dim)

-- for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
-- calculate self.w1(x) and self.w3(x)
matmul(s.hb, s.xb, w.1[l], dim, hidden_dim)
matmul(s.hb2, s.xb, w.w3[l], dim, hidden_dim)
-- SwiGLU non-linearity
for i = 1, hidden_dim do
    local val = s.hb[i]
    -- silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val = val * (1.0 / (1.0 + math.exp(-val)))
    -- elementwise multiply with w3(x)
    val = val * s.hb2[i]
    s.hb[i] = val
end
-- final matmul to get the output of the ffn
matmul(s.xb, s.hb, w.w2[l], hidden_dim, dim)

-- residual connection
for i = 1, dim do
    x[i] = x[i] + s.xb[i]
end
-- final rmsnorm
rmsnorm(x, x, w.rms_final_weight, dim)

-- classifier into logits
matmul(s.logits, x, w.wcls, p.dim, p.vocab_size)
return s.logits

-- The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

TokenIndex = {
    str = nil,
    id = nil
}

Tokenizer = {
    vocab = nil,
    vocab_scores = nil,
    sorted_vocab = nil,
    vocab_size = nil,
    max_token_length = nil,
    byte_pieces = nil -- stores all single-byte strings

}

function compare_tokens(a ,b)
    return a.str < b.str
end

function build_tokenizer(t, tokenizer_path, vocab_size)
    t.vocab_size = vocab_size-- malloc space to hold the scores and the strings
    t.vocab = {} -- vocab_size * sizeof(float)
    t.sorted_vocab = nil
    for i = 0, 255 do
        t.byte_pieces[i * 2] = string.char(i)
        t.byte_pieces[i * 2 + 1] = '\0'
    end

    -- read in the file
    local file = io.open(tokenizer-path, "rb")
    if not file then
        print("could not load" .. tokenizer_path)
        os.exit()
    end
    -- read in the config header
    t.max_token_length = file:read("*n")
    for i = 1, vocab_size do
        t.vocab_scores[i] = file:read("*n")
        local len = file:read("*n")
        t.vocab[i] = file:read(len)
    end
    file:close()
end

function free_tokenizer(t)
    t.vocab = nil
    t.vocab_scores = nil
    t.sorted_vocab = nil
end

function decode(t, prev_token, token)
    local piece = t.vocab[token]
    -- following BOS (1) token, sesntencepiece decoder strips any leading whitespace 
    if prev_token == 1 and piece:sub(1,1) == '' then piece = piece:sub(2)
end

    local byte_val = piece:match("<0x(%02hhX)>")
    if byte_val then
        piece = t.byte_pieces[tonumber(byte_val, 16) * 2]
    end 
    return piece
end
