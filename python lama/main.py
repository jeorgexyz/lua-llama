class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # No. of heads for the queries
    n_kv_heads: Options[int] = None number of heads for the K and V  
    vocab_size: int =  -1 
    multiple_of: int = 256
    ffn_dim_multiplier: Options[float] = None
    norm_eps: float = 1e-5

    #KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 1000.0) # Dimension of the embedding must be even
    
assert head_dim % 2 multiple_of == 0,

# build theta params
# theta_i = 10000 ^ (-2(i-1)/dim for i  = [1, 2, ... dim / 2]
# shape: (head_dim / 2)

theta_numerator = torch.arrange(0, head_dim, 2).float()
# shape: (head_dim / 2)
theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
# construct the positions (the "m" param)
# shape: (seq_len)
m = torch.arrange(seq_len, device=device)
# multiply each therta by each position using the outer product
# shape: (seq_len) outer_product * (head_dim / 2) - > (seq_len, head_dim / 2)
freqs = torch.polar(torch.ones_like(freqs), freqs)
return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seq_len, H, head_dim) - > (B, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    #(Seq_Len, Head_Dim / 2) -> (1, Seq_Len, 1, Head_Dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim / 2) - > (B, seq_len, H, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, head_dim / 2) -> (B, seq_len, H, head_dim)
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else 


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        return self.norm(x)
    
    # The gamma parameter
    self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensore):
        # (B, Seq_Len, Dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):# (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
     return self.weight * self._norm(x.float()).type_as(x)
    
    class SelfAttention(nn.Module):
     
     def __init__(self, args: ModelArgs):
         super().__init__()


         self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

         self.n_heads_q = args.n_heads

         self.n_heads_rep = self.n_heads_q // self.n_kv_heads

         self.head_dim = args.dim // args.n_heads


         self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
         self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
         self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
         self.w_out = nn.Linear(args.dim, args.dim, bias=False)

         self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
         self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))


         def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
             batch_size, seq_len, dim = x.shape # (B, 1, Dim)

        # Apply the Wq, Wk and Wv matrices to queries, keys and values 
        # (B, 1, Dim) -> (B, 1, H_Q * head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * head_Dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q * head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.reshape(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Does not change the shape of the tensors
        xq = apply_rotary_embeddings(xq, freqs_complex, self.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, self.device)

        # Replace the entry in the cache for this token
        self.cache_k[:batch-size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch-size, start_pos:start_pos+seq_len] = xv

        # Retrieve all the cached keys and values so far
        # (B, Seq_Len, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]
     
        # Repeat the head of the K and V tensors for each head of the Q tensor
    class FeedForward(nn.Module):
     
     def __init__(self, arg: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim /3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden = args.multiple_of if args.multiple_of is not None else hidden_dim

        # hidden_size = 7, multiple_of = 5

     class EncoderBlock(nn.Module):

        def __init__(self, args: ModelArgs):
            super().__init__()

            self.n_heads = args.n_heads
            self.dim = args.dim
            self.head_dim = args.dim // args.n_heads

            self.attn = SelfAttention(args)
            self.ff = FeedForward(args)

            # Normalization BEFORE the self attention
            seelf.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            # Normalization BEFORE the feed forward block
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

            def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
                # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
                h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
                out = h + self.ffn.forward(self.ffn_norm(h))
                return out




class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size > 0, "Vocab size must be specified"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args)) #Build out the encoder

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len = tokens.shape  # (B, 1, Dim)

          assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    
