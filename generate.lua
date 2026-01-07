-- generate.lua - Pure Lua text generation (no Torch)

local utils = require('utils')

-- RoPE: Rotary Position Embedding (fixed: use dim in frequency base)
local function apply_rope(q, k, pos, n_heads, head_size, dim)
    local theta_base = 10000.0
    for h = 0, n_heads - 1 do
        local head_offset = h * head_size

        for i = 1, head_size, 2 do
            local idx = head_offset + i
            local pair_idx = (i - 1) // 2

            -- Correct frequency: base uses full embedding dim
            local freq = 1.0 / math.pow(theta_base, (2.0 * pair_idx) / dim)
            local val = pos * freq
            local cos_val = math.cos(val)
            local sin_val = math.sin(val)

            -- Rotate q
            local q0 = q[idx]
            local q1 = q[idx + 1]
            q[idx]     = q0 * cos_val - q1 * sin_val
            q[idx + 1] = q0 * sin_val + q1 * cos_val

            -- Rotate k
            local k0 = k[idx]
            local k1 = k[idx + 1]
            k[idx]     = k0 * cos_val - k1 * sin_val
            k[idx + 1] = k0 * sin_val + k1 * cos_val
        end
    end
end

-- Multi-head attention (supports GQA/MQA via n_kv_heads)
local function multihead_attention(output, q, key_cache, value_cache, pos, c, layer, att_scores_buf)
    local head_size = c.head_size
    local kv_dim = c.kv_dim  -- dim per KV group (dim * n_kv_heads / n_heads)

    for h = 0, c.n_heads - 1 do
        local q_offset = h * head_size + 1

        -- Map query head → KV head (repeating for GQA)
        local kv_h = math.floor(h * c.n_kv_heads / c.n_heads)
        local kv_offset_base = kv_h * head_size

        -- Compute attention scores against all previous positions
        for t = 0, pos do
            local score = 0.0
            local k_base = layer * c.seq_len * kv_dim + t * kv_dim + kv_offset_base

            for i = 0, head_size - 1 do
                score = score + q[q_offset + i] * key_cache[k_base + i + 1]
            end

            att_scores_buf[t + 1] = score / math.sqrt(head_size)
        end

        -- Softmax over time dimension
        utils.softmax(att_scores_buf, pos + 1)

        -- Weighted sum of values
        for i = 0, head_size - 1 do
            local sum = 0.0
            for t = 0, pos do
                local v_base = layer * c.seq_len * kv_dim + t * kv_dim + kv_offset_base
                sum = sum + att_scores_buf[t + 1] * value_cache[v_base + i + 1]
            end
            output[q_offset + i] = sum
        end
    end
end

-- Full transformer forward pass
local function forward(model, token, pos, state, key_cache, value_cache, att_scores_buf)
    local c = model.config
    local w = model.weights
    local s = state

    -- Token embedding → x
    local embed_offset = token * c.dim + 1
    for i = 0, c.dim - 1 do
        s.x[i + 1] = w.token_embedding_table[embed_offset + i]
    end

    -- Transformer blocks
    for l = 1, c.n_layers do
        local layer_idx = l - 1  -- 0-based for cache

        -- Attention RMSNorm
        utils.rmsnorm(s.xb, s.x, w.rms_att_weight[l], c.dim)

        -- QKV projections
        utils.matmul(s.q, s.xb, w.wq[l], c.dim, c.dim)
        utils.matmul(s.k, s.xb, w.wk[l], c.dim, c.dim)
        utils.matmul(s.v, s.xb, w.wv[l], c.dim, c.dim)

        -- Apply Rotary Position Embedding
        apply_rope(s.q, s.k, pos, c.n_heads, c.head_size, c.dim)

        -- Cache K and V for this position
        local kv_base = layer_idx * c.seq_len * c.kv_dim + pos * c.kv_dim + 1
        for i = 0, c.kv_dim - 1 do
            key_cache[kv_base + i] = s.k[i + 1]
            value_cache[kv_base + i] = s.v[i + 1]
        end

        -- Multi-head attention
        multihead_attention(s.xb2, s.q, key_cache, value_cache, pos, c, layer_idx, att_scores_buf)

        -- Attention output projection + residual
        utils.matmul(s.xb, s.xb2, w.wo[l], c.dim, c.dim)
        utils.accum(s.x, s.xb, c.dim)

        -- FFN RMSNorm
        utils.rmsnorm(s.xb, s.x, w.rms_ffn_weight[l], c.dim)

        -- FFN (SwiGLU)
        utils.matmul(s.hb,  s.xb, w.w1[l], c.hidden_dim, c.dim)
        utils.matmul(s.hb2, s.xb, w.w3[l], c.hidden_dim, c.dim)

        for i = 1, c.hidden_dim do
            local x = s.hb[i]
            local silu = x / (1.0 + math.exp(-x))
            s.hb[i] = silu * s.hb2[i]
        end

        utils.matmul(s.xb, s.hb, w.w2[l], c.dim, c.hidden_dim)
        utils.accum(s.x, s.xb, c.dim)
    end

    -- Final RMSNorm
    utils.rmsnorm(s.x, s.x, w.rms_final_weight, c.dim)

    -- Final classifier (shared or separate)
    local cls = w.wcls or w.token_embedding_table
    for i = 1, c.vocab_size do
        local sum = 0.0
        local offset = (i - 1) * c.dim + 1
        for j = 0, c.dim - 1 do
            sum = sum + s.x[j + 1] * cls[offset + j]
        end
        s.logits[i] = sum
    end

    return s.logits
end

-- Text generation loop
local function generate(model, tokenizer, prompt, max_tokens, temperature)
    temperature = temperature or 0.9
    max_tokens = max_tokens or 100

    local c = model.config
    local tokens = tokenizer:encode(prompt)

    print("Prompt tokens: " .. #tokens)

    -- Truncate prompt if too long
    if #tokens >= c.seq_len then
        print(string.format("WARNING: Prompt truncated from %d to %d tokens", #tokens, c.seq_len - 1))
        local truncated = {}
        for i = 1, c.seq_len - 1 do truncated[i] = tokens[i] end
        tokens = truncated
    end

    local state = model:create_run_state()

    -- KV cache: layers × seq_len × kv_dim
    local cache_size = c.n_layers * c.seq_len * c.kv_dim
    local key_cache = {}
    local value_cache = {}
    for i = 1, cache_size do
        key_cache[i] = 0.0
        value_cache[i] = 0.0
    end

    -- Attention scores buffer
    local att_scores_buf = {}
    for i = 1, c.seq_len do att_scores_buf[i] = 0.0 end

    print("\nGenerating...\n")
    print(prompt)

    local pos = 0

    -- Process prompt (prefill)
    for _, token in ipairs(tokens) do
        if pos >= c.seq_len then break end
        forward(model, token, pos, state, key_cache, value_cache, att_scores_buf)
        pos = pos + 1
    end

    -- Autoregressive generation
    local next_token = tokens[#tokens] or 0
    for _ = 1, max_tokens do
        if pos >= c.seq_len then
            print("\n[Reached maximum context length]")
            break
        end

        local logits = forward(model, next_token, pos, state, key_cache, value_cache, att_scores_buf)

        local sampled
        if temperature < 0.01 then
            sampled = utils.argmax(logits) - 1  -- greedy
        else
            for j = 1, c.vocab_size do
                logits[j] = logits[j] / temperature
            end
            utils.softmax(logits, c.vocab_size)
            sampled = utils.sample(logits) - 1
        end

        next_token = sampled
        local token_str = tokenizer:decode(next_token)
        io.write(token_str)
        io.flush()

        if next_token == 1 or next_token == 2 then  -- common EOS tokens
            break
        end

        pos = pos + 1
    end

    print("\n")
end

return {
    forward = forward,
    generate = generate
}