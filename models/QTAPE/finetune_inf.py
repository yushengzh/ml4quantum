import os
current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
target_folder_path = os.path.join(parent_parent_dir, "dataset_generation")

from time import time
import sys
import pickle
import pandas as pd 
import numpy as np
from decoder import Decoder
import embedding 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
import loss
import models
import math
from tqdm import tqdm
import options
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import root_mean_squared_error
from sklearn.kernel_approximation import RBFSampler
test_size = 0.8
device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
n_layers = 8


def objective(trial, model, X_train, X_test, y_train, y_test):
    _threshold = trial.suggest_float("variance_threshold", 0.0, 1.0) 
    _alpha = trial.suggest_loguniform("ridge_alpha", 1, 1e5) 
    _gamma = trial.suggest_loguniform("rbf_gamma", 1e-2, 1e2)
    _n_components = trial.suggest_int("rbf_n_components", 10, 1000)
    global regressor
    selector = VarianceThreshold(threshold=_threshold)
    rbf_feature = RBFSampler(gamma=_gamma, n_components=_n_components, random_state=2024)
    
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)
    X_train_rff = rbf_feature.fit_transform(X_train_selected)
    X_test_rff = rbf_feature.transform(X_test_selected)
    X_train_combined = np.hstack([X_train,X_train_selected,X_train_rff])
    X_test_combined = np.hstack([X_test,X_test_selected,X_test_rff])
    if model == "Ridge":
        regressor = Ridge(alpha=_alpha)
    elif model == "Lasso":
        regressor = Lasso(alpha=_alpha)
    else:
        regressor = LinearRegression()
    regressor.fit(X_train_combined, y_train)
    y_pred = regressor.predict(X_test_combined)
    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse


def main(args, qubits_num, shots, train_samples, test_samples, task, pre_train, random_measurements, alpha):
    utils.fix_seed(args.seed)
    qubits_num = qubits_num
    shots_num = shots 
    samples_num = 100000
    try:
        if args.h == "heisenberg_1d":
            # qubits_num \in [8,10,12,16,25,31]
            # infinite shots and 20000 samples
            finetune_path = "/heisenberg_1d_more/n20000|X(coupling)_y(energy,entropy,corrs)_q{q}.csv".format(q=qubits_num)
            finetune_path = f"/mnt/urchin/kzou/yushengzh/workplace/ai4q/benchmark/dataset_generation/rebuttal/new_dataset/heisenberg_1d/n100000|X(coupling)_y(energy,entropy,corrs)_q{qubits_num}.csv"
            test_path = f"/mnt/urchin/kzou/yushengzh/workplace/ai4q/benchmark/dataset_generation/heisenberg_1d_more/n20000|X(coupling)_y(energy,entropy,corrs)_q8.csv"
            #test_ft_path = "/heisenberg_1d/n1700|X(coupling, meas{shots})_y(energy,entropy,corrs)_q{q}.csv".format(shots=shots_num, q=qubits_num)
        elif args.h == "heisenberg_2d":
            finetune_path = "/heisenberg_2d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q({nx}, {ny}).csv".format(samples_num=samples_num, shots=shots_num, nx=args.nx, ny=args.ny)    
        elif args.h == "tfim":
            finetune_path = "/tf_ising_1d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q{q}.csv".format(samples_num=samples_num, shots=shots_num, q=qubits_num)
        df = pd.read_csv(finetune_path)
        df_test = pd.read_csv(test_path)
        #df = pd.concat([df, df_test], ignore_index=True)
        #df_test = pd.read_csv(target_folder_path + test_ft_path)
        #df = pd.concat([df, df_test], ignore_index=True)
        
    except:
        raise FileNotFoundError("Dataset not found")
    if args.h == "heisenberg_2d":
        qubits_num = args.nx * args.ny
    embedding_dim = 512
    dec_voc_size = 512
    hidden_dim = 128
    seq_len = qubits_num + 1
    samples_num = 100000
    batch_size = 128
    NUM_WORKERS = 4

    conditions = np.array([utils.read_matrix_v2(x) for x in df['coupling_matrix'].values])
    if random_measurements == True:
        meas_records = utils.generate_random_measurement_outcomes_vector(samples_num, qubits_num, shots_num).reshape(-1, qubits_num, shots_num)
    elif random_measurements == False:
        meas_records = np.array([utils.read_matrix_v2(x) for x in df['measurement_samples'].values]).reshape(-1, qubits_num, shots_num)
    else:
        raise ValueError("Random measurements should be either True or False.")
  
    all_idx = np.arange(samples_num)
    batch_conditions = conditions[all_idx]
    batch_measures = meas_records[all_idx]
    cls_token = torch.zeros((samples_num, shots_num, 1), dtype=torch.long)
    batch_measures = torch.cat((cls_token, torch.tensor(batch_measures).permute(0, 2, 1).long()), dim=2).permute(0, 2, 1).float()
    project_layer = -1
    if task == "correlation":
        y = torch.tensor([utils.read_matrix_v2(x) for x in df['exact_correlation'].values])
        project_layer = qubits_num * qubits_num
    elif task == "entropy":
        y = torch.tensor([utils.read_matrix_v2(x) for x in df['exact_entropy'].values])
        
        project_layer = qubits_num - 1
    else:
        raise ValueError("Task not found.")
    
    # print basic information 
    print(f"Hamiltonian:{args.h}, qubits:{qubits_num}, shots:{shots_num}, samples:{samples_num}, task:{task}, random_measurements:{random_measurements}")
    embedding_path = "save/embeddings/rebuttal/{}_embeddings_q{}_s{}_rm{}_{}.pkl".format(args.h, qubits_num, shots_num, random_measurements, samples_num)

    try:    
        with open(embedding_path, "rb") as f:
            all_embedding = pickle.load(f)
        print("Embedding file loaded.")
    except:
        print(embedding_path)
        print("Embedding file not found. Generating new embeddings...")
        rnn = nn.LSTM(shots_num, embedding_dim, 1)
        token_embedding_ft, _ = rnn(batch_measures)
        all_embedding = token_embedding_ft + embedding.get_embedding_ft(samples_num, seq_len, embedding_dim, batch_conditions,device)
        with open(embedding_path, "wb") as f:
            pickle.dump(all_embedding, f)
        print("Embedding file saved.")

    train_sample_idx = np.random.choice(range(train_samples), train_samples, replace=False)
    test_sample_idx = np.arange(80000, 100000, 1)
    print(all_embedding.shape)
    X_train = all_embedding[train_sample_idx]
    y_train = y[train_sample_idx]
    X_test = all_embedding[test_sample_idx]
    y_test = y[test_sample_idx]
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test) 
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    decoder = Decoder(dec_voc_size, seq_len, embedding_dim, ffn_hidden=128, n_head=8, n_layers=n_layers, drop_prob=0.1)
    
    if pre_train:      
        try:
            if args.h == "heisenberg_1d" or args.h == "tfim":
                decoder.load_state_dict(torch.load("save/n_layers_fixed_8/{}_pretrain_q{}_s1024_bs100_ep1500_rmTrue.pt".format(args.h,qubits_num), weights_only=False,map_location='cpu'))
            elif args.h == "heisenberg_2d":
                decoder.load_state_dict(torch.load("save/n_layers_fixed_8/{}_pretrain_q({},{})_s1024_bs100_ep1500.pt".format(args.h, args.nx, args.ny), weights_only=True))
            print("Pretrained model loaded.")
        except:
            raise FileNotFoundError("Pretrained model not found.")
    else:
        # implement random initialization for the model
        pass
           

    finetune_model = models.FinetuneDecoder(decoder, None, embedding_dim, hidden_dim, embedding_dim, project_layer)
    #finetune_model = torch.nn.DataParallel(finetune_model)
    finetune_model = finetune_model.to(device)

    total_params = sum(p.numel() for p in finetune_model.parameters())
    print(f'Total parameters: {total_params}')
    optimizer = optim.Adam(finetune_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    #train_loader, test_loader, finetune_model, optimizer = accelerator.prepare(
    #    train_loader, test_loader, finetune_model, optimizer)
    start_time = time()
    epochs = 1000
    ''''''
    finetune_model.train()
    for i in tqdm(range(epochs)):
        total_loss = 0.0
        for idx, (input, target) in enumerate(train_loader):
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True) 
            output = finetune_model(input)
            
            train_loss = criterion(output, target) + finetune_model.l2_regularization(alpha)
            gradients = torch.autograd.grad(train_loss, finetune_model.parameters(), retain_graph=False, allow_unused=True)
            for param, grad in zip(finetune_model.parameters(), gradients):
                param.grad = grad
            #train_loss.backward()
            #accelerator.backward(train_loss, retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += math.sqrt(train_loss.item())
        if(i % 100 == 0):
            print("Epoch: {}, Loss: {}".format(i, total_loss / len(train_loader)))

    result = 0.0
    end_time = time()
    train_time = end_time - start_time
    # evaluation 
    finetune_model.eval()
    r2_model = 0.0
    with torch.no_grad():
        test_loss = 0.0
        for idx, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)
            output = finetune_model(input)
            test_loss += loss.rmse_loss(output, target)
        avg_rmse_loss = test_loss / len(test_loader)
        print("Test Loss: {}".format(avg_rmse_loss.item()))
        result = avg_rmse_loss.item()
    
    # save models
    '''
    if args.h == "heisenberg_1d" or args.h == "tfim":
        torch.save(finetune_model.state_dict(), "models/QTAPE/save/trained_models/{}_finetune_q{}_s{}_ntr{}_ep{}_rm{}_seed{}.pt".format(args.h, qubits_num, shots_num, train_samples, epochs, random_measurements, args.seed))
    elif args.h == "heisenberg_2d":
        torch.save(finetune_model.state_dict(), "models/QTAPE/save/trained_models/{}_finetune_q({},{})_s{}_ntr{}_ep{}_rm{}.pt".format(args.h, args.nx, args.ny, shots_num, train_samples, epochs, random_measurements))
    '''
    return result, train_time, total_params

if __name__ == "__main__":
    args = options.args_parser()
    qubits_list = [8] # 8, 10, 12, 16, 25, 31, 48, 63, 100, 
    train_samples_list = [100, 1000, 10000, 80000] # [20, 50, 90] 20,40,60,80, 63, 100, 
    test_samples = 10000
    task = args.t
    random_measurements = True # args.rm
    pre_train = False
    shots_list = [1]
    alpha = 0.001

    for qubits_num in qubits_list:
        for shots_num in shots_list:
            for train_samples in train_samples_list:
                tloss, train_time,_ = main(args, qubits_num, shots_num, train_samples, test_samples, task, pre_train, random_measurements, alpha)
                with open("results/n_layers_fixed_8/rebuttal/{hams}_{task}_rmse_pt{pt}_{test_samples}_rm{rm}_sd{seeds}.txt".format(hams=args.h, task=task, pt=pre_train, test_samples=test_samples, rm=True, seeds=args.seed), "a") as f:
                    f.write("qubits_num: {}, shots_num: {}, train_samples: {}, test loss: {}, train time: {}\n".format(qubits_num, shots_num, train_samples, tloss, train_time))
                f.close()
    

