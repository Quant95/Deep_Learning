import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import sampler

import help_func
import new_ALL_Conv



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--net_name', type=str, default='BaseNet_C')
    
    args = parser.parse_args()
    
    train_batch_size = 4
    test_batch_size = 4

    NUM_TRAIN = 49000

    transform = transforms.Compose(
          [transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar10_train = torchvision.datasets.CIFAR10('./cs231n/datasets', train=True, download=True,
                             transform=transform)
    loader_train = torch.utils.data.DataLoader(cifar10_train, batch_size=train_batch_size, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = torchvision.datasets.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=transform)
    loader_val = torch.utils.data.DataLoader(cifar10_val, batch_size=train_batch_size, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = torchvision.datasets.CIFAR10('./cs231n/datasets', train=False, download=True, 
                            transform=transform)
    loader_test = torch.utils.data.DataLoader(cifar10_test, batch_size=test_batch_size)
    
    if(args.net_name == 'BaseNet_A'):
        net = new_ALL_Conv.BaseNet_A()
    elif(args.net_name == 'BaseNet_B'):
        net = new_ALL_Conv.BaseNet_B()
    elif(args.net_name == 'BaseNet_C'):
        net = new_ALL_Conv.BaseNet_C()
        
    run_num = args.run_num
    net_name = args.net_name
    lr = [0.025, 0.01, 0.005, 0.001]
    epoch = [200, 250, 300]
    
    
    help_func.running_model(run_num, net, net_name, lr, epoch, 
                        loader_train, loader_val, loader_test)
    
if __name__ == '__main__':
    main()