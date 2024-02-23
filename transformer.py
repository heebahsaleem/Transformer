import torch.utils.data as data
import torch
import math
import torch.nn as nn
#building blocks of transformer
#1. multihead

class MultiHead(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHead, self).__init__()
        assert d_model % num_heads ==0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model/num_heads

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        attn_scores = torch.matmul(Q, H.transpose(-2,-1))/math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        attn_prob = torch.softmax()
        output = torch.matmul(attn_prob,V)
        return output
    
    def split_heads(self,x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q,K,V, mask=None):
        Q = self.split_heads(self.w_q(Q))
        K = self.split_heads(self.w_k(K))
        V = self.split_heads(self.w_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.w_o(self.combine_heads(attn_output))

        return output
#2. positioning
class Positionfeedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(Positionfeedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
#3. encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__(PositionalEncoding, self)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_Term = torch.exp(torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_Term)
        pe[:, 1::2] = torch.cos(position * div_Term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

