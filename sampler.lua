local Sampler = {}
Sampler.__index = Sampler

function Sampler.new(vocab_size, temperature, top_p)
    local self = setmetatable({}, Sampler)
    self.vocab_size = vocab_size
    self.temperature = temperature
    self.top_p = top_p
    return self
end

function Sampler:sample(logits)
    -- Apply temperature scaling
    logits = logits / self.temperature

    -- Convert logits to probabilities
    local probs = self:softmax(logits)
    local sorted_probs, sorted_indices = self:sort_descending(probs)
    
    local cumulative_probs = 0
    local cutoff_index = self.vocab_size
    for i = 1, self.vocab_size do
        cumulative_probs = cumulative_probs + sorted_probs[i]
        if cumulative_probs > self.top_p then
            cutoff_index = i
            break
        end
    end

    local rand = math.random()
    local cdf = 0
    for i = 1, cutoff_index do
        cdf = cdf + sorted_probs[i]
        if rand < cdf then
            return sorted_indices[i]
        end
    end

    return sorted_indices[cutoff_index]
end

function Sampler:softmax(logits)
    local max_logit = torch.max(logits)
    local exp_logits = torch.exp(logits - max_logit)
    local sum_exp = torch.sum(exp_logits)
    return exp_logits / sum_exp
end

function Sampler:sort_descending(t)
    local sorted, indices = torch.sort(t, true)
    return sorted, indices
end

return Sampler
