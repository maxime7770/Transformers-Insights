import torch.nn as nn
import torch.nn.functional as F
from src.model.layers import PositionalEncoder, EncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_sequence_length):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoder(d_model, max_sequence_length)
        # Define a stack of multiple encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    
    # Complete the forward pass method
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super(ClassifierHead, self).__init__()
        # Add linear layer for multiple-class classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        logits = self.fc(x[:, 0, :]) # first token corresponds to the classification token
        # Obtain log class probabilities upon raw outputs
        return F.softmax(logits, dim=-1)
