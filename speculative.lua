-- speculative.lua - Speculative decoding: a draft model proposes, the target verifies
--
-- A small draft model runs k steps ahead. The large target model then evaluates all
-- k proposals plus one extra position, and every proposal the target agrees with is
-- accepted without costing an extra target step. The target is invoked once per
-- block rather than once per token.
--
-- The emitted distribution is exactly the target model's. Under greedy decoding the
-- text is identical to running the target alone; under sampling, the acceptance rule
-- plus a residual resample preserves the target distribution.
--
-- Reference: Leviathan et al. 2023, "Fast Inference from Transformers via
-- Speculative Decoding"; Chen et al. 2023, "Accelerating Large Language Model
-- Decoding with Speculative Sampling".

local utils = require('utils')
local gen = require('generate')

local speculative = {}

-- A decode context is a model plus its own activations and KV cache, so the draft
-- and the target advance independently over the same token sequence.
local function new_context(model)
    local c = model.config
    local ctx = {
        model = model,
        state = model:create_run_state(),
        key_cache = {},
        value_cache = {},
        att_buf = {},
    }

    local cache_size = c.n_layers * c.seq_len * c.kv_dim
    for i = 1, cache_size do
        ctx.key_cache[i] = 0.0
        ctx.value_cache[i] = 0.0
    end
    for i = 1, c.seq_len do
        ctx.att_buf[i] = 0.0
    end

    return ctx
end

local function step(ctx, token, pos)
    return gen.forward(ctx.model, token, pos, ctx.state, ctx.key_cache, ctx.value_cache, ctx.att_buf)
end

-- Snapshot logits into a probability table. forward() reuses one logits buffer, so a
-- distribution must be copied out before the next call overwrites it. Returns the
-- distribution and, for greedy decoding, the argmax index.
local function distribution(logits, vocab_size, temperature)
    local p = {}

    if temperature < 0.01 then
        local best = utils.argmax(logits, vocab_size)
        for i = 1, vocab_size do
            p[i] = 0.0
        end
        p[best] = 1.0
        return p, best
    end

    for i = 1, vocab_size do
        p[i] = logits[i] / temperature
    end
    utils.softmax(p, vocab_size)
    return p, nil
end

-- Sample from (target - draft)+ renormalised. This is what keeps the emitted
-- distribution equal to the target's after a proposal is rejected.
local function sample_residual(target_p, draft_p, vocab_size)
    local resid = {}
    local total = 0.0

    for i = 1, vocab_size do
        local d = target_p[i] - draft_p[i]
        if d < 0 then d = 0 end
        resid[i] = d
        total = total + d
    end

    if total <= 0 then
        return utils.sample(target_p)
    end

    for i = 1, vocab_size do
        resid[i] = resid[i] / total
    end
    return utils.sample(resid)
end

-- target, draft: Model instances. Passing the same model as both is the correctness
--                check: every proposal should be accepted and the output should match
--                plain generation exactly.
-- lookahead:     k, the number of tokens drafted per round.
-- Generated text streams to stdout as it is committed; a stats table is returned.
function speculative.generate(target, draft, tokenizer, prompt, max_tokens, temperature, lookahead)
    temperature = temperature or 0.0
    max_tokens = max_tokens or 100
    local k = lookahead or 4

    local c = target.config
    local vocab = c.vocab_size
    local greedy = temperature < 0.01

    -- Separate contexts even when the two models are the same object: each needs its
    -- own KV cache, because the draft speculates past the committed sequence.
    local target_ctx = new_context(target)
    local draft_ctx = new_context(draft)

    local tokens = tokenizer:encode(prompt)
    io.write(prompt)
    io.flush()

    -- Prefill both contexts with every prompt token except the last, which seeds the
    -- first round (same convention as generate.lua).
    local pos = 0
    for i = 1, #tokens - 1 do
        if pos >= c.seq_len - 1 then break end
        step(target_ctx, tokens[i], pos)
        step(draft_ctx, tokens[i], pos)
        pos = pos + 1
    end

    local current = tokens[#tokens]
    local stats = {
        emitted = 0,
        proposed = 0,
        accepted = 0,
        target_rounds = 0,    -- batched target invocations: the metric that matters
        target_forwards = 0,  -- individual target forward passes
        draft_forwards = 0,
        prompt_tokens = #tokens,
        lookahead = k,
    }

    while stats.emitted < max_tokens do
        -- Shrink the block near the context limit so no position overruns seq_len.
        local block = math.min(k, max_tokens - stats.emitted, c.seq_len - pos - 2)
        if block < 1 then break end

        -- 1. Draft proposes `block` tokens autoregressively.
        local draft_idx, draft_probs = {}, {}
        local token = current
        for i = 1, block do
            local logits = step(draft_ctx, token, pos + i - 1)
            stats.draft_forwards = stats.draft_forwards + 1

            local p, best = distribution(logits, vocab, temperature)
            local idx = greedy and best or utils.sample(p)
            draft_idx[i] = idx
            draft_probs[i] = p
            token = idx - 1
        end
        stats.proposed = stats.proposed + block

        -- 2. Target evaluates the current token and every proposal. In a batched
        --    runtime these block+1 positions are one matmul over a single weight
        --    load, which is where the speedup comes from; here they run
        --    sequentially, so they count as one round.
        local target_probs, target_best = {}, {}
        for i = 1, block + 1 do
            local input = (i == 1) and current or (draft_idx[i - 1] - 1)
            local logits = step(target_ctx, input, pos + i - 1)
            stats.target_forwards = stats.target_forwards + 1
            target_probs[i], target_best[i] = distribution(logits, vocab, temperature)
        end
        stats.target_rounds = stats.target_rounds + 1

        -- 3. Accept the longest prefix the target agrees with.
        local emitted = {}
        local n_accepted = 0
        for i = 1, block do
            local idx = draft_idx[i]
            local accept
            if greedy then
                accept = (idx == target_best[i])
            else
                local q = draft_probs[i][idx]
                accept = (q <= 0) or (math.random() < math.min(1.0, target_probs[i][idx] / q))
            end

            if not accept then break end
            n_accepted = n_accepted + 1
            emitted[#emitted + 1] = idx
        end
        stats.accepted = stats.accepted + n_accepted

        -- When every proposal is accepted, the target commits a token the draft
        -- produced but never consumed, leaving a hole in the draft KV cache at that
        -- position. Fill it here, or the draft silently diverges from the target on
        -- the next round and starts proposing tokens that cannot be accepted.
        if n_accepted == block then
            step(draft_ctx, draft_idx[block] - 1, pos + block)
            stats.draft_forwards = stats.draft_forwards + 1
        end

        -- 4. One correction token from the target: a free bonus when every proposal
        --    was accepted, otherwise a corrected token at the first disagreement.
        if n_accepted == block then
            if greedy then
                emitted[#emitted + 1] = target_best[block + 1]
            else
                emitted[#emitted + 1] = utils.sample(target_probs[block + 1])
            end
        else
            local i = n_accepted + 1
            if greedy then
                emitted[#emitted + 1] = target_best[i]
            else
                emitted[#emitted + 1] = sample_residual(target_probs[i], draft_probs[i], vocab)
            end
        end

        -- 5. Commit. KV entries written past this point belong to rejected proposals
        --    and are overwritten next round, so no explicit cache rewind is needed.
        local stop = false
        for _, idx in ipairs(emitted) do
            local id = idx - 1
            if id == 1 or id == 2 then
                stop = true
                break
            end

            io.write(tokenizer:decode(id))
            io.flush()

            current = id
            pos = pos + 1
            stats.emitted = stats.emitted + 1
            if stats.emitted >= max_tokens then break end
        end

        if stop or pos >= c.seq_len - 1 then break end
    end

    print("")
    return stats
end

return speculative
