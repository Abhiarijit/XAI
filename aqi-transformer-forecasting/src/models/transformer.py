class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        return self.fc_out(x)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.ffn = FeedForwardNetwork(model_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.layer_norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout2(ffn_output))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.values = nn.Linear(model_dim, model_dim, bias=False)
        self.keys = nn.Linear(model_dim, model_dim, bias=False)
        self.queries = nn.Linear(model_dim, model_dim, bias=False)
        self.fc_out = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        N, seq_length, _ = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("nqhd,nkhd->nqhk", [queries, keys])
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nqhk,nvhd->nqhd", [attention, values]).reshape(N, seq_length, self.num_heads * self.head_dim)
        return self.fc_out(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, model_dim, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(model_dim, model_dim * 4)
        self.fc2 = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))