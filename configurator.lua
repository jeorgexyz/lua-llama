-- Get command-line arguments
local args = {...}

-- Iterate over the arguments
for i, arg in ipairs(args) do
    -- Check if the argument is a key-value pair
    if arg:find('=') then
        -- Split the argument into a key and a value
        local key, val = arg:match('([^=]+)=(.+)')

        -- Check if the key exists in the global table
        if _G[key] then
            -- Convert the value to a number if possible
            local num = tonumber(val)
            if num then
                val = num
            end

            -- Set the global variable
            _G[key] = val
            print('Set ' .. key .. ' to ' .. tostring(val))
        else
            print('Unknown key: ' .. key)
        end
    else
        -- Assume the argument is a Lua file and execute it
        assert(loadfile(arg))()
        print('Executed file ' .. arg)
    end
end
