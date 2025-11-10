import torch.nn as nn
import torch 

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=48, embed_dim=32):
        super(SimpleAutoencoder, self).__init__()
        
        # Define layers: single hidden layer is the embedding layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        # Encoding step
        x = self.encoder(x)
        
        # Decoding step (reconstruction)
        reconstructed = self.decoder(x)
        return x, reconstructed  # Return both embedding and reconstruction
    
    
    
class FinetuneDecoder(nn.Module):
    def __init__(self, decoder, trg_mask, input_dim, hidden_dim, output_dim, projection_dim):
        super(FinetuneDecoder, self).__init__()
        self.decoder = decoder
        self.trg_mask = trg_mask
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.projection = nn.Linear(output_dim, projection_dim)
        self.tanh = nn.Tanh()
        
        
    def forward(self, x):
        x = self.decoder(x, self.trg_mask)
        
        # feature aggregation layer: Mean pooling
        x = x.permute(0, 2, 1)
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        
        # Projection linear layer
        x = self.tanh(x)
        x = self.projection(x)

        return x

    def l2_regularization(self, lambd):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return lambd * l2_reg