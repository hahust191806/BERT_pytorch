import torch 
from torch import nn 
import torch.nn.functional as f 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionHead(nn.Module):
    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()
        
        self.dim_inp = dim_inp 
        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None): # input_tensor: (batch_size, num_token, embedd_size), attention_mask: (batch_size, num_token, num_token)
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor) # query, key, value: (batch_size, num_token, hidden_size)

        scale = query.size(1) ** 0.5 # 
        scores = torch.bnm(query, key.transpose(1, 2)) / scale # torch.bnm performs a batch matrix product: (batch_size, num_token, num_token)
        scores = scores.masked_fill_(attention_mask, -1e9) # (batch_size, num_token, num_token)
        attn = f.softmax(scores, dim=-1) # (batch_size, num_token, num_token)
        context = torch.bnm(attn, value) # (batch_size, num_token, hidden_size)
        return context # (batch_size, num_token, hidden_size)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.Modulelist([
            AttentionHead(dim_inp, dim_out) for _ in range(num_heads) # dim_inp == embedd_size: embedding space size of Embedding Object, dim_out: embedding space size of AttentionHead
        ])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp) # hidden_size = dim_out * num_heads
        self.norm = nn.LayerNorm(dim_inp) 

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor): # input_tensor: (batch_size, num_token, embedd_size), attention_mask: (batch_size, num_token, num_token)
        s = [head(input_tensor, attention_mask) for head in self.heads] # s: [(batch_size, num_token, hidden_size) * num_heads]
        scores = torch.cat(s, dim=-1) # (batch_size, num_token, hidden_size * num_heads)
        scores = self.linear(scores) # (batch_size, num_token, dim_inp == embedd_size)
        return self.norm(scores) # (batch_size, num_token, dim_inp == embedd_size)