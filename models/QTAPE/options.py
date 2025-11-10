import os
import sys
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Pretraining of QTAPE (ICLR 2024)')
    

    parser.add_argument('--h', type=str, default='heisenberg_1d',
                        help='Hamiltonian type: heisenberg_1d, heisenberg_2d, tfim or xxz')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--n1', type=int, default=100, 
                        help='Number of pretrain samples')
    parser.add_argument('--n2', type=int, default=300, 
                        help='Number of finetune(Supervised Train) samples')
    
    parser.add_argument('--s1', type=int, default=1024,
                        help='Number of pretrain shots')
    parser.add_argument('--s2', type=int, default=64,
                        help='Number of finetune(Supervised Train) shots:[64,128,256,512]')
    ## 1d heisenberg model
    parser.add_argument('--q', type=int, default=63, 
                        help='Number of qubits')
    parser.add_argument('--t', type=str, default="entropy", 
                        help='tasks: correlation, entropy')
    parser.add_argument('--rm', type=lambda x: x.lower() == 'true', default=True,
                        help='Random measurements')
    
    ## 2d heisenberg model
    parser.add_argument('--nx', type=int, default=5,
                        help='Number of qubits in x direction of 2d Heisenberg model, \
                        or number of qubits in 1d Heisenberg model')
    parser.add_argument('--ny', type=int, default=5,
                        help='Number of qubits in y direction of 2d Heisenberg model')
    
    parser.add_argument('--bs', type=int, default=100,
                        help='Batch size')
 

    
    
    args = parser.parse_args()
    return args