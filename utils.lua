-- utils.lua - Pure Lua utility functions (no Torch)

local utils = {}

-- Binary file reading
function utils.read_int32(file)
    local bytes = file:read(4)
    if not bytes or #bytes ~= 4 then
        error("Failed to read int32")
    end
    return string.unpack("<i4", bytes)
end

function utils.read_float32(file)
    local bytes = file:read(4)
    if not bytes or #bytes ~= 4 then
        error("Failed to read float32")
    end
    return string.unpack("<f", bytes)
end

function utils.read_float32_array(file, n)
    local arr = {}
    for i = 1, n do
        arr[i] = utils.read_float32(file)
    end
    return arr
end

-- Math operations
function utils.softmax(x, size)
    size = size or #x
    
    -- Find max for numerical stability
    local max_val = x[1]
    for i = 2, size do
        if x[i] > max_val then
            max_val = x[i]
        end
    end
    
    -- Exp and sum
    local sum = 0
    for i = 1, size do
        x[i] = math.exp(x[i] - max_val)
        sum = sum + x[i]
    end
    
    -- Normalize
    for i = 1, size do
        x[i] = x[i] / sum
    end
    
    return x
end

function utils.rmsnorm(out, x, weight, size)
    -- RMS normalization
    local ss = 0
    for i = 1, size do
        ss = ss + x[i] * x[i]
    end
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)
    
    for i = 1, size do
        out[i] = weight[i] * (ss * x[i])
    end
end

function utils.matmul(out, x, w, n, d)
    -- Matrix-vector multiplication: out = w * x
    -- w is (n x d) matrix stored row-major
    -- x is (d,) vector
    -- out is (n,) vector
    for i = 1, n do
        local sum = 0
        for j = 1, d do
            sum = sum + w[(i-1)*d + j] * x[j]
        end
        out[i] = sum
    end
end

function utils.accum(a, b, size)
    -- a += b (element-wise)
    for i = 1, size do
        a[i] = a[i] + b[i]
    end
end

function utils.sample(probabilities)
    -- Sample from a probability distribution
    local r = math.random()
    local cdf = 0
    for i = 1, #probabilities do
        cdf = cdf + probabilities[i]
        if r < cdf then
            return i
        end
    end
    return #probabilities
end

function utils.argmax(x, size)
    size = size or #x
    local max_i = 1
    local max_val = x[1]
    for i = 2, size do
        if x[i] > max_val then
            max_val = x[i]
            max_i = i
        end
    end
    return max_i
end

return utils