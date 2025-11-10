import os
current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
target_folder_path = os.path.join(parent_parent_dir, "dataset_generation")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import utils
import pandas as pd 
import numpy as np
from decoder import Decoder
import embedding 
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
import options
import pickle

samples_num = 100
# qubits_num = 8 # L
shots_num = 512 # K
device_id = [1, 2, 3, 4, 5, 6, 7]
train_samples = 100
n_layers = 8
def main(args, nx, ny, random_measurements=False):
    utils.fix_seed(2024)
    hamiltonian = args.h
    shots_num = 512#args.s1
    samples_num = 500#args.n1
    nx, ny = nx, ny # args.nx, args.ny
    qubit = nx * ny
    qubits_num = qubit
    batch_size = args.bs
    try:
        if hamiltonian == "heisenberg_1d":
            dataset_path = "/heisenberg_1d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q{q}.csv".format(samples_num=samples_num, shots=shots_num, q=qubits_num)
        elif hamiltonian == "heisenberg_2d":
            dataset_path = "/heisenberg_2d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q({nx}, {ny}).csv".format(samples_num=samples_num, shots=shots_num, nx=nx, ny=ny)
        elif hamiltonian == "tfim":
            dataset_path = "/tf_ising_1d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q{q}.csv".format(samples_num=samples_num, shots=shots_num, q=qubits_num)
        df = pd.read_csv(target_folder_path + dataset_path)
        print("Dataset found.\t")
    except:
        raise FileNotFoundError("Dataset not found.")
    
    if args.h == "heisenberg_2d":
        qubits_num = nx * ny
    
    conditions = np.array([utils.read_matrix_v2(x) for x in df['coupling_matrix'].values])
    
    if random_measurements: 
        meas_records = utils.generate_random_measurement_outcomes_vector(samples_num, qubits_num, shots_num)
    else:
        meas_records = np.array([utils.read_matrix_v2(x) for x in df['measurement_samples'].values])
    
    meas_records = meas_records.reshape(-1, shots_num, qubits_num)
    meas_records = meas_records.reshape(-1, qubits_num)

    new_conditions = []
    for i in range(samples_num):
        for _ in range(shots_num):
            new_conditions.append(conditions[i])
    new_conditions = np.array(new_conditions)
    print(new_conditions.shape, meas_records.shape)
    
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    embedding_dim = 512
    seq_len = qubits_num + 1 # +1 for the CLS token
    dec_voc_size = 512
    batch_conditions, batch_measures = [], []
    sample_idx = np.random.choice(range(samples_num*shots_num), train_samples, replace=False)
    batch_conditions = new_conditions[sample_idx]
    batch_measures = meas_records[sample_idx]
    
    
    all_embeddings, token_embedding = embedding.get_embedding(train_samples, seq_len, embedding_dim, batch_measures, batch_conditions)   
    embeddings = all_embeddings
    labels = F.softmax(token_embedding, dim=-1)
    
    
        
    dataset = TensorDataset(embeddings, labels)
    
    print("Dataset created. The size is ", len(dataset))
    
    dataloader = DataLoader(dataset, batch_size=train_samples, shuffle=True)
    
    trg_mask = torch.ones(8, all_embeddings.shape[1], all_embeddings.shape[1])
    trg_mask = torch.triu(trg_mask, diagonal=1)
    trg_mask = trg_mask.masked_fill(trg_mask == 1, float(0)).to(device)

    decoder = Decoder(dec_voc_size, seq_len, embedding_dim, ffn_hidden=128, n_head=8, n_layers=n_layers, drop_prob=0.1)  
    # decoder = nn.parallel(decoder, device_id)
    decoder = decoder.to(device)
    criterion = nn.KLDivLoss(reduction='sum')
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    accelerator = Accelerator()
    #dataloader, decoder, optimizer = accelerator.prepare(dataloader, decoder, optimizer)
    epochs = 1500
    decoder.train()
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
        
            outputs = decoder(inputs, trg_mask)
                
            loss = criterion(outputs.view(-1), targets.view(-1))
            gradients = torch.autograd.grad(loss, decoder.parameters(), retain_graph=True)
            for param, grad in zip(decoder.parameters(), gradients):
               param.grad = grad
            #accelerator.backward(loss, retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.8f}')

    # save the decoder model
    if args.h == "heisenberg_1d" or args.h == "tfim":
        pretrain_path = "save/n_layers_fixed_{}/{}_pretrain_q{}_s{}_bs{}_ep{}_rm{}.pt".format(n_layers, args.h, qubits_num, shots_num, batch_size, epochs, random_measurements)
    elif args.h == "heisenberg_2d":
        pretrain_path = "save/n_layers_fixed_{}/{}_pretrain_q({},{})_s{}_bs{}_ep{}_rm{}.pt".format(n_layers, args.h, nx, ny, shots_num, batch_size, epochs, random_measurements)
    
    torch.save(decoder.state_dict(), pretrain_path)
    print("Pretrained model saved at", pretrain_path)
    
if __name__ == "__main__":
    qubits_list = [8, 10, 12, 16, 25, 31, 48, 63, 100, 127]
    nxy_list = [[10, 10], [15, 15]]
    args = options.args_parser()
    random_measurements = [False, True]
    for nx, ny in nxy_list:
            for rm in random_measurements:
                main(args, nx, ny, rm)
    #for qubit in qubits_list:
    #        main(args, qubit, False)