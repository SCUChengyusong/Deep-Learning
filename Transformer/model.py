import torch
import math
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        chunk = x.chunk(x.size(-1), dim=2)
        out = torch.Tensor([]).to(x.device)
        for i in range(len(chunk)):
            out = torch.cat((out, chunk[i] + self.pe[:chunk[i].size(0), ...]), dim=2)
        return out

def transformer_generate_tgt_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1
    mask = (
        mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
    )
    return mask

"""
    param n_encoder_inputs: dimension of the input dataset
    param n_decoder_inputs: dimension of the output dataset
    param d_model: dimension of embedding features
    param Sequence_length: length of time series
"""
class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, Sequence_length, embed_dim, num_heads, num_enlayers, num_delayers, dropout):
        super(Transformer, self).__init__()

        self.input_pos_embedding = nn.Embedding(5000, embedding_dim=embed_dim)
        self.target_pos_embedding = nn.Embedding(5000, embedding_dim=embed_dim)

        self.pos_encoding = PositionalEncoding(embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=4*embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=4*embed_dim)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enlayers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_delayers)

        self.input_projection = nn.Linear(in_dim, embed_dim)
        self.output_projection = nn.Linear(out_dim, embed_dim)

        self.linear = nn.Linear(embed_dim, out_dim)
        self.output_linear = nn.Linear(Sequence_length, out_dim)

    def encode_in(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (torch.arange(0, in_sequence_len, device=src.device).unsqueeze(0).repeat(batch_size, 1))
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def decode_out(self, tgt, memory):
        tgt_start = self.output_projection(tgt).permute(1, 0, 2)
        out_sequence_len, batch_size = tgt_start.size(0), tgt_start.size(1)
        pos_decoder = (torch.arange(0, out_sequence_len, device=tgt.device).unsqueeze(0).repeat(batch_size, 1))
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        tgt = tgt_start + pos_decoder
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask) + tgt_start
        # out = self.decoder(tgt=tgt, memory=memory) + tgt_start
        out = out.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        return out

    def forward(self, src, target_in):
        src = self.encode_in(src)
        out = self.decode_out(tgt=target_in, memory=src)
        # out = out.squeeze(2)
        out = self.linear(out)
        return out

