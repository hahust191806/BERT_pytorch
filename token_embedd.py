import torch 
from torch import nn 
import torch.nn.functional as f 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(JointEmbedding, self).__init__()
        self.embed_size = embed_size # defined embedding space size 
        self.token_emb = nn.Embedding(vocab_size, embed_size) # defined Embedding Object for embed token
        self.segment_emb = nn.Embedding(vocab_size, embed_size) # defined Embedding Object for segment token
        self.norm = nn.LayerNorm(embed_size) # defined Layer Normalization to Normalize embedding vector 

    def forward(self, input_tensor): # input_tensor: (batch_size, indices_tensor)
        sentence_size = input_tensor.size(-1) # defined sentence_size == length of sentences, number of token in sentence 
        pos_tensor = self.attention_position(self.size, input_tensor) 

        segment_tensor = torch.zeros_like(input_tensor).to(device)
        segment_tensor[:, (sentence_size // 2 + 1):] = 1 
        
        output = self.token_emb(input_tensor) + self.segment_emb(segment_tensor) + pos_tensor
        return self.norm(output) # (batch_size, num_token, embedd_size)

    def atttention_position(self, dim, input_tensor): # input_tensor: (batch_size, indices_tensor)
        batch_size = input_tensor.size(0) # get batch_size 
        sentence_size = input_tensor.size(-1) # defined sentence_size == length of sentences, number of token in sentence 

        pos = torch.arange(sentence_size, dtype=torch.long).to(device)
        d = torch.arange(dim, dtype=torch.long).to(device)
        d = (2 * d / dim)

        pos = pos.unsqueeze(1)
        pos = pos / (1e4 ** d)

        pos[:, ::2] = torch.sin(pos[:, ::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        return pos.expand(batch_size, *pos.size())

    def numeric_position(self, dim, input_tensor):
        pos_tensor = torch.arange(dim, dtype=torch.long).to(device)
        return pos_tensor.expand_as(input_tensor)