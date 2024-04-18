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


local ffi = require("ffi")

-- C declaration for munmap:
ffi.cdef[[
  int munmap(void *addr, size_t length);
]]

-- Assume an OS where libc is available:
ffi.C.munmap(t.data, t.file_size) 
function build_transformer(t, checkpoint_path)
    read_checkpoint(checkpoint_path, t.config, t.weights)
    malloc_run_state(t.state, t.config)
end

function free_transformer(t)
    if t.data then 
        ffi.C.munmap(t.data, t.file_size) 
        t.data = nil  
    end 
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

function safe_printf(piece)
    if not piece or piece == '' then return end
    if #piece == 1 then
        local byte_val = string.byte(piece)
        if not (string.isprint(byte_val) or string.isspace(byte_val)) then
            return -- bad byte, don't print it
        end
    end
    print(piece)
end 

function str_lookup(str, sorted_vocab, vocab_size)
    -- efficiently find the perfect match for str in vocab, return its index or -1 if not found
    local tok = {str = str } -- acts as the key to search for
    local res = table.binary_search(sorted_vocab, tok, compare_tokens)
    return res and res.id or -1
end

function encode(t, text, bos, eos, tokens, n_tokens)
    -- encode the string text (input) into an upper-bound preallocated tokens [] array
    -- bos != 0 means prepend the BOS token (=1), eos !=0 means append the EOS token (=2)
    if not text then
        print("cannot encode NULL text")
        os.exit()
    end

    if not t.sorted_vocab then
        -- lazily malloc and sort the vocabulary
        t.sorted_vocab = {} -- vocab_size * sizeofTokenIndex)
        for i  = 1, t.vocab[i] = { str = t.vocab[i], id = i }
        end
        table.sort(t.sorted_vocab, compare_tokens)
    end
    
    -- create a temporary buffer that will store merger candidates of alwats two consecutive tokens 
    -- *2 for concat, +1 for null terminator +2 for UTSF8 (in case max_token_length is 1)
    local str_buffer = {} -- (t.max_token_length*2 +1 +2) * sizeof(char)
    local str_len = 0

    --start at 0 tokens
    n_tokens = 0

    -- add optional BOS (=1) token, if desired 
    if bos then tokens[n_tokens + 1] = 1 end

    -- add_dummy_prefix is true by default
    -- so prepend a dummy prefix token to the input string, but only if text != ""
    if text ~= '' then
        local dummy_prefix = str_lookup("", t.sorted_vocab, t.vocab_size)
        tokens[n_tokens + 1] = dummy_prefix
    end

    -- Code point ↔ UTF-8 conversion
    -- First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    -- U+0000	U+007F	    0xxxxxxx
    -- U+0080	U+07FF	    110xxxxx	10xxxxxx
    -- U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    -- U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    -- process the raw (UTF-8) byte sequence of the input string
    for c in text:gmatch(".") do
           -- reset buffer if the current byte is ASCII or a leading byte
        -- 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        -- 0x80 is 10000000
        -- in UTF-8, all continuation bytes start with "10" in first two bits
        -- so in English this is: "if this byte is not a continuation byte"
        if (string.byte(c) & 0xC0) ~= 0x80 then
            -- this byte must be either a leading byte (11...) or an ASCII char (0x...)
            -- => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0
        end

           -- append the current byte to the buffer
           str_buffer[str_len + 1] = c -- ++ is post-increment, incremented after this line
           str_buffer[str_len + 2] = '\0'
   
           -- while the next character is a continuation byte, continue appending
           -- but if there are too many of them, just stop to avoid overruning str_buffer size.
           if (string.byte(text, str_len + 2) & 0xC0) == 0x80 and str_len < 4 then
               str_len = str_len + 1
               continue
           end
           -- ok c+1 is not a continuation byte, so we've read in a full codepoint
           local id = str_lookup(table.concat(str_buffer), t.sorted_vocab, t.vocab_size)

           if id ~= -1 then
               -- we found this codepoint in vocab, add it as a token
               tokens[n_tokens + 1] = id
           else
               -- byte_fallback encoding: just encode each byte as a token
               -- +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
               -- so the individual bytes only start at index 3
               for i = 1, str_len do
                   tokens[n_tokens + 1] = string.byte(str_buffer[i]) + 3
               end
           end
           str_len = 0 -- protect against a sequence of stray UTF8 continuation bytes
       end
   end
   -- merge the best consecutive pair each iteration, according to the scores in vocab_scores
   while true do
    local best_score = -1e10
    local best_id = -1
    local best_idx = -1

    for i = 1, n_tokens - 1 do
        -- check if we can merge the pair (tokens[i], tokens[i+1])
        local str_buffer = t.vocab[tokens[i]] .. t.vocab[tokens[i + 1]]
        local id = str_lookup(str_buffer, t.sorted_vocab, t.vocab_size)
        if id ~= -1 and t.vocab_scores[id] > best_scores then
            -- this merge pair exists in vocab! record its score and position
            best_score = t.vocab_scores[id]
            best_id = id
            best_idx = i
        end
    end

    if best_idx == -1 then
        break -- we couldn't find any more pairs to merge, so we're done
    end
    -- merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id
    -- delete token at position best_idx+1, shift the entire sequence back 1
    for i = best_idx + 1, n_tokens - 1 do
        tokens[i] = tokens[i + 1]
    end
    n_tokens = n_tokens - 1 -- token length decreased
end
-- add optional EOS (=2) token, if desired
if eos then tokens[n_tokens + 1] = 2 end
-- The Sampler, which takes logits and returns a sampled token
-- sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
ProbIndex = {
    prob = nil,
    index = nil
}

Sampler = {
    vocab_size = nil,
    probindex = nil, -- buffer used in top-p sampling
    temperature = nil,
    topp = nil,
    rng_state = nil
}

function sample_argmax(probabilities, n)
    -- return the index that has the highest probability
    local max_i = 1
    local max_p = probabilities[1]
    for i = 2, n do
        if probabilities[i] > max_p then
            max_i = I
            max_p = probabilities[i]
        end
    end
    return max_i
end

function sample_mult(probabilities, n, coin)
    -- sample index from probabilities (they must sum to 1!)
    -- coin is a random number in [0, 1], usually from random_f32()
    local cdf = 0.0
    for i = 1, n do
        cdf = cdf + probabilities[i]
        if coin < cdf then
            return i 
        end
    end
    return n -- in case of rounding errors
end

function compare(a, b)
    if a.prob > b.prob then return -1 end 
    if a.prob < b.prob then return 1 end
    return 0
end

function sample_topp(probabilities, n, topp, probindex, coin)
    -- topp sampling (or "nucleus sampling") samples from the smallest set of
    -- tokens that exceed probability topp. This way we never sample tokens that
    -- have very low probabilities and are less likely to go "off the rails".
    -- coin is a random number in [0, 1), usually from random_f32()

    local n0 = 0
    -- quicksort indices in descending order of probabilities
    -- values smaller than (1 - topp) / (n - 1) cannot be part of the result
    -- so for efficiency we crop these out as candidates before sorting
    local cutoff = (1.0 - topp) / (n - 1)
    for i = 1, n do
        if probabilities[i] >= cutoff then
            probindex[n0 + 1] = { index = i, prob = probabilities[i] }
            n0 = n0 + 1
        end
    end
    table.sort(probindex, compare)
        -- truncate the list where cumulative probability exceeds topp
        local cumulative_prob = 0.0
        local last_idx = n0 -- in case of rounding errors consider all elements
        for i = 1, n0 do
            cumulative_prob = cumulative_prob + probindex[i].prob
            if cumulative_prob > topp then
                last_idx = i
                break -- we've exceeded topp by including last_idx
            end
        end
            -- sample from the truncated list
    local r = coin * cumulative_prob
    local cdf = 0.0
    for i = 1, last_idx do
        cdf = cdf + probindex[i].prob
        if r < cdf then
            return probindex[i].index
        end
    end
    return probindex[last_idx].index -- in case of rounding errors
end
function build_sampler(sampler, vocab_size, temperature, topp, rng_seed)
    sampler.vocab_size = vocab_size
    sampler.temperature = temperature
    sampler.topp = topp
    sampler.rng_state = rng_seed
    -- buffer only used with nucleus sampling; may not need but it's ~small
    sampler.probindex = {} -- sampler.vocab_size * sizeof(ProbIndex)
end

function free_sampler(sampler)
    sampler.probindex = nil
end

function random_u32(state)
    -- xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    state = state ~ (state >> 12)
    state = state ~ (state << 25)
    state = state ~ (state >> 27)
    return ((state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF
end

function random_f32(state) -- random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0
end
function sample(sampler, logits)
    -- sample the token given the logits and some hyperparameters
    local next
    if sampler.temperature == 0.0 then
        -- greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler.vocab_size)
    else
          -- apply the temperature to the logits
          for q = 1, sampler.vocab_size do logits[q] = logits[q] / sampler.temperature end
          -- apply softmax to the logits to get the probabilities for next token
          softmax(logits, sampler.vocab_size)
          -- flip a (float) coin (this is our source of entropy for sampling)
          local coin = random_f32(sampler.rng_state)
          -- we sample from this distribution to get the next token
          if sampler.topp <= 0 or sampler.topp >= 1 then
              -- simply sample from the predicted probability distribution
              next = sample_mult(logits, sampler.vocab_size, coin)
          else
              -- top-p (nucleus) sampling, clamping the least likely tokens to zero
              next = sample_topp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin)
          end
      end
      return next
  end
  -- utilities: time      

function time_in_ms()
    -- return time in milliseconds, for benchmarking the model speed
    return os.time() * 1000
end

-- generation loop

function generate(transformer, tokenizer, sampler, prompt, steps)
    local empty_prompt = ""
    if not prompt then prompt = empty_prompt end

    -- encode the (string) prompt into tokens sequence
    local num_prompt_tokens = 0
    local prompt_tokens = {} -- (strlen(prompt)+3) * sizeof(int) -- +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, num_prompt_tokens)
    if num_prompt_tokens < 1 then
        print("something is wrong, expected at least 1 prompt token")
        os.exit()
    end
-- start the main loop
local start = 0  -- used to time our code, only initialized after first iteration
local next        -- will store the next token in the sequence
local token = prompt_tokens[1] -- kick off with the first token in the prompt
local pos = 0     -- position in the sequence
while pos < steps do

    -- forward the transformer to get logits for the next token
    local logits = forward(transformer, token, pos)

    -- advance the state machine
    if pos < num_prompt_tokens - 1 then
        -- if we are still processing the input prompt, force the next prompt token
        next = prompt_tokens[pos + 2]
    else
        -- otherwise sample the next token from the logits
        next = sample(sampler, logits)
    end
    pos = pos + 1

    -- data-dependent terminating condition: the BOS (=1) token delimits sequences
    if next == 1 then break end

    -- print the token as string, decode it with the Tokenizer object
    local piece = decode(tokenizer, token, next)
    safe_printf(piece) -- same as printf("%s", piece), but skips "unsafe" bytes
    io.flush()
    token = next

    -- init the timer here because the first iteration can be slower
    if start == 0 then start = time_in_ms() end
end
print("\n")

-- report achieved tok/s (pos-1 because the timer starts after first iteration)
if pos > 1 then
    local end_time = time_in_ms()
    print(string.format("achieved tok/s: %f\n", (pos-1) / (end_time-start)*1000))
end

prompt_tokens = nil
end

function read_stdin(guide, buffer, bufsize)
-- read a line from stdin, up to but not including \n
print(guide)
buffer = io.read()
end
-- chat loop
-- I manually inspected the tokens for a few chat conversations compared to
-- python reference and that seemed ok, but this was not thoroughly tested and
-- is not safely implemented, it's more a proof of concept atm.

function chat(transformer, tokenizer, sampler, cli_user_prompt, cli_system_prompt, steps)

-- buffers for reading the system prompt and user prompt from stdin
-- you'll notice they are soomewhat haphazardly and unsafely set atm
local system_prompt = ""
local user_prompt = ""
local rendered_prompt = ""
local num_prompt_tokens = 0
local prompt_tokens = {} -- 1152 * sizeof(int)
local user_idx

-- start the main loop
local user_turn = true -- user starts
local next        -- will store the next token in the sequence
local token       -- stores the current token to feed into the transformer
local prev_token
local pos = 0     -- position in the sequence
while pos < steps do

    -- when it is the user's turn to contribute tokens to the dialog...
    if user_turn then
        -- get the (optional) system prompt at position 0
        if pos == 0 then
            -- at position 0, the user can also contribute a system prompt
            if not cli_system_prompt then
                -- system prompt was not passed in, attempt to get it from stdin
                print("Enter system prompt (optional): ")
                system_prompt = io.read()
            else
                -- system prompt was passed in, use it
                system_prompt = cli_system_prompt
            end
        end
        -- get the user prompt
        if pos == 0 and cli_user_prompt then
            -- user prompt for position 0 was passed in, use it
            user_prompt = cli_user_prompt
        else
            -- otherwise get user prompt from stdin
            print("User: ")
            user_prompt = io.read()
        end
        -- render user/system prompts into the Llama 2 Chat schema
        if pos == 0 and system_prompt ~= "" then
            local system_template = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]"
            rendered_prompt = string.format(system_template, system_prompt, user_prompt)
        else
            local user_template = "[INST] %s [/INST]"
            rendered_prompt = string.format(user_template, user_prompt)
        end
        -- encode the rendered prompt into tokens
        encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, num_prompt_tokens)
        user_idx = 1 -- reset the user index
        user_turn = false
        print("Assistant: ")
    end

    -- determine the token to pass into the transformer next
    if user_idx <= num_prompt_tokens then
        -- if we are still processing the input prompt, force the next prompt token
        token = prompt_tokens[user_idx]
        user_idx = user_idx + 1
    else
        -- otherwise use the next token sampled from previous turn
        token = next
    end
    -- EOS (=2) token ends the Assistant turn
    if token == 2 then user_turn = true end

    -- forward the transformer to get logits for the next token
    local logits = forward(transformer, token, pos)
    next = sample(sampler, logits)
    pos = pos + 1

    if user_idx > num_prompt_tokens and next ~= 2 then
        -- the Assistant is responding, so print its output
        local piece = decode(tokenizer, token, next)
        safe_printf(piece) -- same as printf("%s", piece), but skips "unsafe" bytes
        io.flush()
    end
    if next == 2 then print("\n") end
end
print("\n")
prompt_tokens = nil
end

function read_stdin(guide, buffer, bufsize)
-- read a line from stdin, up to but not including \n
print(guide)
buffer = io.read()
end
-- CLI, include only if not testing
-- TESTING = false

function error_usage()
print("Usage:   run <checkpoint> [options]")
print("Example: run model.bin -n 256 -i \"Once upon a time\"")
print("Options:")
print("  -t <float>  temperature in [0,inf], default 1.0")
print("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9")
print("  -s <int>    random seed, default time(NULL)")
print("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len")
print("  -i <string> input prompt")
print("  -z <string> optional path to custom tokenizer")
print("  -m <string> mode: generate|chat, default: generate")
print("  -y <string> (optional) system prompt in chat mode")
os.exit()
end

function main()
-- default parameters
local checkpoint_path = nil  -- e.g. out/model.bin
local tokenizer_path = "tokenizer.bin"
local temperature = 1.0   -- 0.0 = greedy deterministic. 1.0 = original. don't set higher
local topp = 0.9          -- top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
local steps = 256            -- number of steps to run for
local prompt = nil        -- prompt string
local rng_seed = 0 -- seed rng with time by default
local mode = "generate"    -- generate|chat
local system_prompt = nil -- the (optional) system prompt to use in chat mode
-- poor man's C argparse so we can override the defaults above from the command line
if #arg >= 2 then checkpoint_path = arg[1] else error_usage() end
for i = 2, #arg, 2 do
    -- do some basic validation
    if i + 1 >= #arg then error_usage() end -- must have arg after flag
    if arg[i]:sub(1, 1) ~= '-' then error_usage() end -- must start with dash
    if #arg[i] ~= 2 then error_usage() end -- must be -x (one dash, one letter)
    -- read in the args
    if arg[i]:sub(2, 2) == 't' then temperature = tonumber(arg[i + 1])
    elseif arg[i]:sub(2, 2) == 'p' then topp = tonumber(arg[i + 1])
    elseif arg[i]:sub(2, 2) == 's' then rng_seed = tonumber(arg[i + 1])
    elseif arg[i]:sub(2, 2) == 'n' then steps = tonumber(arg[i + 1])
    elseif arg[i]:sub(2, 2) == 'i' then prompt = arg[i + 1]
    elseif arg[i]:sub(2, 2) == 'z' then tokenizer_path = arg[i + 1]
    elseif arg[i]:sub(2, 2) == 'm' then mode = arg[i + 1]
    elseif arg[i]:sub(2, 2) == 'y' then system_prompt = arg[i + 1]
    else error_usage() end
end
-- parameter validation/overrides
if rng_seed <= 0 then rng_seed = os.time() end
if temperature < 0.0 then temperature = 0.0 end
if topp < 0.0 or topp > 1.0 then topp = 0.9 end
if steps < 0 then steps = 0 end
-- build the Transformer via the model .bin file
local transformer = Transformer:new()
transformer:build(checkpoint_path)
if steps == 0 or steps > transformer.config.seq_len then steps = transformer.config.seq_len end -- ovrerride to ~max length
-- build the Tokenizer via the tokenizer .bin file
local tokenizer = Tokenizer:new()
tokenizer:build(tokenizer_path, transformer.config.vocab_size)
-- build the Sampler
local sampler = Sampler:new()
sampler:build(transformer.config.vocab_size, temperature, topp, rng_seed)
-- run!
if mode == "generate" then
    generate(transformer, tokenizer, sampler, prompt, steps)
elseif mode == "chat" then
    chat(transformer, tokenizer, sampler, prompt, system_prompt, steps)
else
    print("unknown mode: " .. mode)
    error_usage()
end

-- memory and file handles cleanup
sampler:free()
tokenizer:free()
transformer:free()
return 0
end

if not TESTING then main() end



    