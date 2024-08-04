import torch 
from torch import nn 
import torch.nn.functional as f 

from multihead_attention import MultiHeadAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out) # (batch_size, sentence_size, dim_inp)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor): # input_tensor: (batch_size, sentence_size, dim_inp == embedd_size)
        context = self.attention(input_tensor, attention_mask) # (batch_size, sentence_size, dim_inp)
        res = self.feed_forward(context) # (batch_size, sentence_size, dim_inp)
        return self.norm(res) # (batch_size, sentence_size, dim_inp)