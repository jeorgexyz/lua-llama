-- model.lua - Pure Lua model loading (no Torch)

local utils = require('utils')

local Model = {}
Model.__index = Model

function Model.new(checkpoint_path)
    local self = setmetatable({}, Model)
    
    local file = io.open(checkpoint_path, "rb")
    if not file then
        error("Cannot open checkpoint file: " .. checkpoint_path)
    end
    
    print("Loading model from: " .. checkpoint_path)
    
    -- Read config (7 int32s)
    local dim = utils.read_int32(file)
    local hidden_dim = utils.read_int32(file)
    local n_layers = utils.read_int32(file)
    local n_heads = utils.read_int32(file)
    local n_kv_heads = utils.read_int32(file)
    local vocab_size = utils.read_int32(file)
    local seq_len = utils.read_int32(file)
    
    -- Check if classifier weights are shared (llama2.c convention: negative vocab_size means shared)
    local shared_classifier = vocab_size > 0
    if vocab_size < 0 then
        vocab_size = -vocab_size
    end
    
    self.config = {
        dim = dim,
        hidden_dim = hidden_dim,
        n_layers = n_layers,
        n_heads = n_heads,
        n_kv_heads = n_kv_heads,
        vocab_size = vocab_size,
        seq_len = seq_len,
        shared_classifier = shared_classifier
    }
    
    local c = self.config
    print(string.format("Config: dim=%d, layers=%d, heads=%d, kv_heads=%d, vocab=%d, seq_len=%d, shared_classifier=%s", 
        c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len, tostring(c.shared_classifier)))
    
    -- Calculate derived values (required for KV cache and attention)
    c.head_size = c.dim // c.n_heads
    c.kv_dim = c.n_kv_heads * c.head_size   -- total KV dimension (important for GQA/MQA)
    
    -- Allocate weights
    self.weights = {}
    local w = self.weights
    
    -- Token embedding table
    w.token_embedding_table = utils.read_float32_array(file, c.vocab_size * c.dim)
    
    -- llama2.c stores each tensor with ALL layers contiguous, in this exact order:
    -- rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3
    w.rms_att_weight = {}
    for l = 1, c.n_layers do w.rms_att_weight[l] = utils.read_float32_array(file, c.dim) end

    w.wq = {}
    for l = 1, c.n_layers do w.wq[l] = utils.read_float32_array(file, c.dim * c.dim) end
    w.wk = {}
    for l = 1, c.n_layers do w.wk[l] = utils.read_float32_array(file, c.kv_dim * c.dim) end
    w.wv = {}
    for l = 1, c.n_layers do w.wv[l] = utils.read_float32_array(file, c.kv_dim * c.dim) end
    w.wo = {}
    for l = 1, c.n_layers do w.wo[l] = utils.read_float32_array(file, c.dim * c.dim) end

    w.rms_ffn_weight = {}
    for l = 1, c.n_layers do w.rms_ffn_weight[l] = utils.read_float32_array(file, c.dim) end

    w.w1 = {}
    for l = 1, c.n_layers do w.w1[l] = utils.read_float32_array(file, c.hidden_dim * c.dim) end
    w.w2 = {}
    for l = 1, c.n_layers do w.w2[l] = utils.read_float32_array(file, c.dim * c.hidden_dim) end
    w.w3 = {}
    for l = 1, c.n_layers do w.w3[l] = utils.read_float32_array(file, c.hidden_dim * c.dim) end

    -- Final RMS norm
    w.rms_final_weight = utils.read_float32_array(file, c.dim)
    
    -- Skip precomputed RoPE frequencies (we compute them on-the-fly)
    local freq_size = c.seq_len * (c.head_size // 2)
    utils.read_float32_array(file, freq_size)  -- freq_cis_real
    utils.read_float32_array(file, freq_size)  -- freq_cis_imag
    
    -- Read classifier weights (wcls) if not shared with embeddings
    if not c.shared_classifier then
        print("Attempting to read separate classifier weights (wcls)...")
        
        local current_pos = file:seek()
        local file_end = file:seek("end")
        file:seek("set", current_pos)
        
        local remaining = file_end - current_pos
        local wcls_size = c.vocab_size * c.dim * 4  -- bytes
        
        if remaining >= wcls_size then
            w.wcls = utils.read_float32_array(file, c.vocab_size * c.dim)
            print("✓ Read separate classifier weights")
        else
            print(string.format("⚠ Not enough data for wcls (need %d bytes, have %d)", wcls_size, remaining))
            print("  Falling back to shared classifier (using token_embedding_table)")
            w.wcls = nil
            c.shared_classifier = true
        end
    else
        print("Using shared classifier (token_embedding_table)")
        w.wcls = nil
    end
    
    file:close()
    
    print("Model loaded successfully!")
    return self
end

-- Allocate run state (activations)
function Model:create_run_state()
    local c = self.config
    local s = {}
    
    -- Activations
    s.x = {}      -- current activation (dim,)
    s.xb = {}     -- residual branch (dim,)
    s.xb2 = {}    -- attention output buffer (dim,)
    s.hb = {}     -- FFN hidden (hidden_dim,)
    s.hb2 = {}    -- FFN gate (hidden_dim,)
    s.q = {}      -- query (dim,)
    s.k = {}      -- key (dim,)
    s.v = {}      -- value (dim,)
    s.logits = {} -- output logits (vocab_size,)
    
    -- Initialize arrays to zero
    for i = 1, c.dim do
        s.x[i] = 0
        s.xb[i] = 0
        s.xb2[i] = 0
        s.q[i] = 0
        s.k[i] = 0
        s.v[i] = 0
    end
    
    for i = 1, c.hidden_dim do
        s.hb[i] = 0
        s.hb2[i] = 0
    end
    
    for i = 1, c.vocab_size do
        s.logits[i] = 0
    end
    
    return s
end

return Model