import torch
import torch.nn as nn
import torch.optim as optim
import math
from models import SimpleAutoencoder
from sklearn.preprocessing import MinMaxScaler

def get_token_embedding(batch_size, batch_measures, embedding_dim, num_embeddings=6):
    embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
    cls_token = torch.zeros(batch_size, 1, dtype=torch.long)
    token_embedding = embedding_layer(torch.cat((cls_token, torch.tensor(batch_measures).long()), dim=1))
    return token_embedding


def get_positional_embedding(batch_size, seq_len, embedding_dim):
    positional_embedding = torch.zeros(seq_len, embedding_dim)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
    positional_embedding[:, 0::2] = torch.sin(position * div_term)
    positional_embedding[:, 1::2] = torch.cos(position * div_term)
    return positional_embedding


def get_condition_embedding(batch_size, batch_conditions, embedding_dim, device):
    # Instantiate the model
    ae = SimpleAutoencoder(input_dim=batch_conditions.shape[1], hidden_dim=48, embed_dim=embedding_dim)
    ae = ae.to(device)
    optimizer = optim.Adam(ae.parameters(), lr=0.01)
    loss_function = nn.MSELoss()  # Reconstruction loss
    epochs = 500
    data_loader = torch.utils.data.DataLoader(
        torch.tensor(batch_conditions).float(), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True
        )

    for epoch in range(epochs):
        for inputs in data_loader:
            inputs = inputs.to(device)  
            optimizer.zero_grad()
            condition_embedding, reconstructed = ae(inputs)  
            loss = loss_function(reconstructed, inputs) 
            loss.backward()  
            optimizer.step()  
            # print('Epoch %d, Loss: %.4f' % (epoch, loss.item()))

    scaler = MinMaxScaler()
    condition_embedding_np = condition_embedding.cpu().detach().numpy()
    condition_embedding_norm = scaler.fit_transform(condition_embedding_np)
    condition_embedding_norm = torch.tensor(condition_embedding_norm)
    return condition_embedding_norm



def get_embedding(batch_size, seq_len, embedding_dim, batch_measures, batch_conditions):
    print("embedding...\t")
    token_embedding = get_token_embedding(batch_size, batch_measures, embedding_dim)
    positional_embedding = get_positional_embedding(batch_size, seq_len, embedding_dim)
    condition_embedding = get_condition_embedding(batch_size, batch_conditions, embedding_dim)

    # broadcasting summation
    condition_embedding_expanded = condition_embedding.unsqueeze(1).expand(-1, seq_len, -1)
    positional_embedding_expanded = positional_embedding.unsqueeze(0).expand(batch_size, -1, -1)
    all_embeddings = token_embedding + positional_embedding_expanded + condition_embedding_expanded
    print("embedding done.\t")
    return all_embeddings, token_embedding



## finetune embedding
def get_embedding_ft(batch_size, seq_len, embedding_dim, batch_conditions, device):
    positional_embedding = get_positional_embedding(batch_size, seq_len, embedding_dim)
    condition_embedding = get_condition_embedding(batch_size, batch_conditions, embedding_dim, device)
    condition_embedding_expanded = condition_embedding.unsqueeze(1).expand(-1, seq_len, -1)
    positional_embedding_expanded = positional_embedding.unsqueeze(0).expand(batch_size, -1, -1)
    all_embeddings = positional_embedding_expanded + condition_embedding_expanded
    
    return all_embeddings
