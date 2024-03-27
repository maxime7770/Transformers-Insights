import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)      
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # Split the sequence embeddings in x across the attention heads
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.head_dim)
    
    def compute_attention(self, query, key, mask=None):
        # Compute dot-product attention scores
        # dimensions of query and key are (batch_size * num_heads, seq_length, head_dim)
        # after transpose, dimensions is (batch_size * num_heads, head_dim, seq_length)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        # now, dimensions of scores is (batch_size * num_heads, seq_length, seq_length)
        if mask is not None:
            scores = scores.view(-1, scores.shape[0] // self.num_heads, mask.shape[1], mask.shape[2]) # for compatibility
            scores = scores.masked_fill(mask == 0, float('-1e9'))
            scores = scores.view(-1, mask.shape[1], mask.shape[2])
        # Normalize attention scores into attention weights
        attention_weights = F.softmax(scores, dim=-1)

        return attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)

        attention_weights = self.compute_attention(query, key, mask)
            
        # Multiply attention weights by values, concatenate and linearly project outputs
        output = torch.matmul(attention_weights, value)
        output = output.view(batch_size, self.num_heads, -1, self.head_dim).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.output_linear(output)


# Subclass an appropriate PyTorch class 
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_length):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_length = max_length
        
        # Initialize the positional encoding matrix
        pe = torch.zeros(max_length, d_model)

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        # Calculate and assign position encodings to the matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    # Update the embeddings tensor adding the positional encodings
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class FeedForwardSubLayer(nn.Module):
    # Specify the two linear layers' input and output sizes
    def __init__(self, d_model, d_ff):
        super(FeedForwardSubLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    # Apply a forward pass
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# Complete the initialization of elements in the encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))