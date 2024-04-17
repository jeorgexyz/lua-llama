ModelConfig = {
    dim = 4096,
    n_layers = 32,
    n_heads = 32,
    n_kv_heads = 32,
    vocab_size = 32000,
    multiple_of = 256,
    ffn_dim_multiplier = 4,
    norm_eps = 1e-5,
    max_batch_size = 32,
    max_seq_len = 2048,
    device = "cuda"
}