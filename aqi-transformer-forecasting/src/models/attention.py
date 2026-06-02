class ScaledDotProductAttention:
    def __init__(self, dropout_rate=0.1):
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class MultiHeadAttention:
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

        self.dense = torch.nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout_rate)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        output, attention_weights = self.attention.forward(query, key, value, mask)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        return self.dense(output), attention_weights