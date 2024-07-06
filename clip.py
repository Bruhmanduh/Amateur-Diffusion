import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding
        
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self attention
        self.attention = SelfAttention(n_head, n_embd)
        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Feedforward layer
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # (Batch_Size, Seq_Len, Dim)
        residue = x
        
        ### SELF ATTENTION ###

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension. 

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output


'''
import torch
from torch import nn
from torch.nn import functional as F
from attention import Self_Attention

class CLIPEmbedding(nn.Module):
    def __init__(self, num_vocab: int, embed_length: int, num_tokens: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_vocab, embed_length)

        self.positional_embeddings = nn.Parameter(torch.zeros(num_tokens, embed_length))

    def forward(self, tokens):
        # (batch_size, num of tokens) --> (batch_size, num of tokens, embedding length)
        x = self.token_embeddings(tokens)

        x = x + self.positional_embeddings

        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, num_heads: int, embed_length: int):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(embed_length)

        self.attention = Self_Attention(num_heads, embed_length)

        self.layernorm2 = nn.LayerNorm(embed_length)

        self.linear1 = nn.Linear(embed_length, 4 * embed_length)

        self.linear2 = nn.Linear(4 * embed_length, embed_length)

    def forward(self, x: torch.Tensor):
        # (batch_size, num of tokens, embedding length)
        residue = x
        x = self.layernorm1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x

        x = self.layernorm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x) # quickgelu activation function
        x = self.linear2(x)

        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self, ):
        super().__init()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (batch_size, num of tokens) --> (batch_size, num of tokens, embedding length)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        # (batch_size, num of tokens, embedding length)
        output = self.layernorm(state)

        return output
'''