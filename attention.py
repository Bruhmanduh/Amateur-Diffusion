
import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: # (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape 
        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf) 
        
        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head) 

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) 

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2) 

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output

'''
class Self_Attention(nn.Module):
    def __init__(self, num_heads: int, embeddings: int, in_bias=True, out_bias=True):
        # heads for self attention and the channels behind each pixel will be the pixel's embedding (representative vector)
        super().__init__()
        self.in_projection = nn.Linear(embeddings, 3 * embeddings, bias = in_bias)
        self.out_projection = nn.Linear(embeddings, embeddings, bias = out_bias)
        self.heads = num_heads
        self.d_heads = embeddings // num_heads

        def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
            # x: (batch_size, lenght of sequence/ every pixel, channels/embeddings for each pixel)
            input_shape = x.shape
            batches, pixels, embeddings = input_shape

            interim_shape = (batches, pixels, self.heads, self.d_heads)

            # creates the three matrices for self attention
            q, k, v = self.in_projection(x).chunk(3, dim=-1)

            # (batches, pixels, embeddings) --> (batches, pixels, num of heads, embeddings per head) --> (batches, num of heads, pixels, embeddings per head)
            q = q.view(interim_shape).transpose(1, 2)
            k = k.view(interim_shape).transpose(1, 2)
            v = v.view(interim_shape).transpose(1, 2)

            # (batches, num of heads, pixels, embeds per head) @ (batches, num of heads, embeds per head, pixels) = (batches, num of heads, pixels, pixels)
            weight = q @ k.transpose(-1, -2)

            if causal_mask:
                mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
                weight.masked_fill(mask, -torch.inf)
            
            weight = weight // (math.sqrt(self.d_heads))

            weight = F.softmax(weight, dim=-1)

            # (batches, num of heads, pixels, pixels) @ (batches, num of heads, pixels, embeds per head) = (batches, num of heads, pixels, embeds per heads)
            output = weight @ v

            # (batches, num of heads, pixels, embeds per heads) --> (batches, pixels, num of heads, embeds per heads)
            output = output.transpose(1, 2)

            # (batches, pixels, num of heads, embeds per heads) --> (batches, pixels, embeddings)
            output = output.reshape(input_shape)

            output = self.out_projection(output)

            # (batches, pixels, embeddings)
            return output 


class Cross_Attention(nn.Module):
    def __init__(self, num_heads, image_embeds, prompt_embeds, in_bias=True, out_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(image_embeds, image_embeds, bias=in_bias)
        self.k_proj = nn.Linear(prompt_embeds, image_embeds, bias=in_bias)
        self.v_proj = nn.Linear(prompt_embeds, image_embeds, bias=in_bias)
        self.out_proj = nn.Linear(image_embeds, image_embeds, bias=out_bias)

        self.num_heads = num_heads
        self.dim_heads = image_embeds // num_heads

        def forward(self, x, y):
            # x = image: (batch_size, pixels, embeddings)
            # y = prompt: (batch_size, tokens, embeddings)
            input_shape = x.shape
            batch_size, pixels, embeddings = input_shape

            interim_shape = (batch_size, -1, self.num_heads, self.dim_heads)

            q = self.q_proj(x)
            k = self.k_proj(y)
            v = self.v_proj(y)

            # (batches, pixels/tokens, embeddings) --> (batches, pixels/tokens, num of heads, embeddings per head) --> (batches, num of heads, pixels/tokens, embeddings per head)
            q = q.view(interim_shape).transpose(1, 2)
            k = k.view(interim_shape).transpose(1, 2)
            v = v.view(interim_shape).transpose(1, 2)

            weight = q @ k.transpose(-1, -2)
            weight = weight // (math.sqrt(self.dim_heads))

            weight = F.softmax(weight, dim=-1)

            # (batches, num of heads, pixels, pixels) @ (batches, num of heads, pixels, embeds per head) = (batches, num of heads, pixels, embeds per heads)
            output = weight @ v

            # (batches, num of heads, pixels, embeds per heads) --> (batches, pixels, num of heads, embeds per heads)
            output = output.transpose(1, 2).contiguous()

            # (batches, pixels, num of heads, embeds per heads) --> (batches, pixels, embeddings)
            output = output.view(input_shape)

            output = self.out_proj(output)

            # (batches, pixels/tokens, embeddings)
            return output 
'''