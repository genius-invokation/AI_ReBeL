import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_dim)
        x = x + self.pe[:x.size(0)]
        return x

class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, output_dim, dropout=0.1):
        super(CustomTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        # src shape: (seq_len, batch_size)
        embedded = self.embedding(src) * math.sqrt(embedded.size(-1))
        embedded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        # 取最后一个时间步的输出，或者根据需求进行池化
        output = self.fc_out(transformer_output[-1])
        return output


class CombinedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, trans_hidden_dim, trans_layers, dnn_hidden_dim, output_dim, dropout=0.1):
        super(CombinedModel, self).__init__()
        self.transformer = CustomTransformerModel(vocab_size, embed_dim, num_heads, trans_hidden_dim, trans_layers, output_dim, dropout)
        # 额外的 DNN 层
        self.dnn = nn.Sequential(
            nn.Linear(output_dim, dnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dnn_hidden_dim, output_dim)
        )

    def forward(self, src, src_key_padding_mask=None):
        transformer_output = self.transformer(src, src_key_padding_mask)
        output = self.dnn(transformer_output)
        return output