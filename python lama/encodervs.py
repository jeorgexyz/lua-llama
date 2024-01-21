  class EncoderBlock(nn.Module):
        def __init__(self, args: ModelArgs) -> None:
            super().__init__()
            self.n_heads = args.n_heads
            self.dim = args.dim
            self.head_dim = args.dim // args.n_heads

            self.attn = MultiheadAttention(args)
            self.ffn = FeedForwardNetwork(args)

        def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
            x = self.attn(x, freqs_complex)
            x = self.ffn(x)
            return x