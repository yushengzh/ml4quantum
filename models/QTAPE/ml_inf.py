import os
current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
target_folder_path = os.path.join(parent_parent_dir, "dataset_generation")
import optuna
import optuna.logging
from time import time
import sys
import pickle
import pandas as pd 
import numpy as np
from functools import partial
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
from sklearn.preprocessing import StandardScaler
test_size = 0.8
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
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
    samples_num = 20000
    try:
        if args.h == "heisenberg_1d":
            # qubits_num \in [8,10,12,16,25,31]
            # infinite shots and 20000 samples
            finetune_path = f"/mnt/urchin/kzou/yushengzh/workplace/ai4q/benchmark/dataset_generation/rebuttal/new_dataset/heisenberg_1d/n100000|X(coupling)_y(energy,entropy,corrs)_q{qubits_num}.csv"
            test_path = f"/mnt/urchin/kzou/yushengzh/workplace/ai4q/benchmark/dataset_generation/heisenberg_1d_more/n20000|X(coupling)_y(energy,entropy,corrs)_q8.csv"
        elif args.h == "heisenberg_2d":
            finetune_path = "/heisenberg_2d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q({nx}, {ny}).csv".format(samples_num=samples_num, shots=shots_num, nx=args.nx, ny=args.ny)    
        elif args.h == "tfim":
            finetune_path = "/tf_ising_1d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q{q}.csv".format(samples_num=samples_num, shots=shots_num, q=qubits_num)
        df = pd.read_csv(finetune_path)
        test_df = pd.read_csv(test_path)
        #df = pd.concat([df, test_df], ignore_index=True)
        
    except:
        raise FileNotFoundError("Dataset not found")
    if args.h == "heisenberg_2d":
        qubits_num = args.nx * args.ny
    embedding_dim = 512
    dec_voc_size = 512
    hidden_dim = 128
    seq_len = qubits_num + 1
    samples_num = 20000
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


    train_sample_idx = np.random.choice(range(train_samples), train_samples, replace=False)
    test_sample_idx = np.arange(80000, 100000, 1)
    X_train = conditions[train_sample_idx]
    y_train = y[train_sample_idx]
    X_test = conditions[test_sample_idx]
    y_test = y[test_sample_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
   
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    time_limit = 20
    start_time = time()
    study_lasso = optuna.create_study(direction="minimize")
    objective_with_data = partial(objective, model="Lasso", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    study_lasso.optimize(objective_with_data, n_trials=50,timeout=time_limit)
    rmse_lasso = study_lasso.best_value
    params_lasso = study_lasso.best_params
    print(f"RMSE Lasso: {rmse_lasso}")
    end_time = time()
    train_time_lasso = end_time - start_time

    start_time = time()
    study_ridge = optuna.create_study(direction="minimize")
    objective_with_data = partial(objective, model="Ridge", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    study_ridge.optimize(objective_with_data, n_trials=50,timeout=time_limit)
    rmse_ridge = study_ridge.best_value
    params_ridge = study_ridge.best_params
    print(f"RMSE Ridge: {rmse_ridge}")
    end_time = time()
    train_time_ridge = end_time - start_time
    
    return rmse_lasso, train_time_lasso, params_lasso, rmse_ridge, train_time_ridge, params_ridge

if __name__ == "__main__":
    args = options.args_parser()
    qubits_list = [8] # 8, 10, 12, 16, 25, 31, 48, 63, 100, 
    train_samples_list = [100, 1000, 10000, 80000] # [20, 50, 90] 20,40,60,80, 63, 100, 
    test_samples = 20000
    task = 'entropy'# args.t
    random_measurements = True # args.rm
    pre_train = False
    shots_list = [1,8,64,512]
    alpha = 0.001
    shots_num = 1

    for qubits_num in qubits_list:
        for train_samples in train_samples_list:
            rmse_lasso, train_time_lasso, params_lasso, rmse_ridge, train_time_ridge, params_ridge = main(args, qubits_num, shots_num, train_samples, test_samples, task, pre_train, random_measurements, alpha)
            with open("results/ML/rebuttal/{hams}_{task}_rmse_{test_samples}_sd{seeds}.txt".format(hams=args.h, task=task, test_samples=test_samples,seeds=args.seed), "a") as f:
                f.write("qubits_num: {}, train_samples: {}, rmse_lasso: {}, train_time_lasso: {}, params_lasso: {}, rmse_ridge: {}, train_time_ridge: {}, params_ridge: {}\n".format(qubits_num, train_samples, rmse_lasso, train_time_lasso, params_lasso, rmse_ridge, train_time_ridge, params_ridge))
            f.close()
    

