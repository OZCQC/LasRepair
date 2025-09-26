"serves as the new main.py"

import argparse
from EMCL.multi import MultiModelIterativeGenerativeRepair
import torch


def EMCL():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beers')
    parser.add_argument('--model', type=str, default='t5-large')  # maybe switch model for test?
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--exclude_list', type=list, default=[0])
    parser.add_argument('--seed', type=int, default=114)

    parser.add_argument('--dirty_path', type=str, default='../dataset/beers/dirty.csv')
    parser.add_argument('--clean_path', type=str, default='../dataset/beers/clean.csv')
    parser.add_argument('--save_path', type=str, default='./result')
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    print(123)


if __name__ == '__main__':
    EMCL()
    
    