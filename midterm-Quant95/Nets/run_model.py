import argparse
from six.moves import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


import ALL_Conv



def run_model(net, net_name, run_num):
    
    train_batch_size = 4 # at first can be small, 64
    test_batch_size = 4 #64

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
    #trainset - abstract class representing the training Dataset
    trainset = torchvision.datasets.CIFAR10(root='./cs231n/datasets', train=True,
                                        download=True, transform=transform)
    #trainloader - combines train dataset and sampler, provides iterators
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True)
    #testset - abstract class representing the test Dataset
    testset = torchvision.datasets.CIFAR10(root='./cs231n/datasets', train=False,
                                       download=True, transform=transform)
    #testloader - combines test dataset and sampler, provides iterators
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False)
    USE_GPU = True
    dtype = torch.float32 # we will be using float throughout this tutorial
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Constant to control how frequently we print train loss
    print_every = 100

    print('using device:', device)
    
    
    net = net.to(device=device)
    criterion = nn.CrossEntropyLoss()        
        
    lr_1, lr_2, lr_3, lr_4 = 0.25, 0.1, 0.05, 0.01
    weight_decay = 0.001

    max_epoch = 350
    display_interval = 500

    train_size = 50000
    test_size = 10000

    num_train_batch = train_size/train_batch_size
    num_test_batch = test_size/test_batch_size

    train_loss = np.zeros((max_epoch,1))
    train_acc = np.zeros((max_epoch,1))
    test_loss = np.zeros((max_epoch,1))
    test_acc = np.zeros((max_epoch,1))

    for epoch in range(max_epoch):
        if(epoch<200):
            lr = lr_1
        elif(epoch<250):
            lr = lr_2
        elif(epoch<300):
            lr = lr_3
        else:
            lr = lr_4
    
        optimizer = optim.SGD( net.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)
        
        running_epoch_loss = 0.
        running_loss_print = 0.
        epoch_total_num = 0
        correct_num = 0
    
        for i, data in enumerate(trainloader):
            inputs_data, labels_data = data
            inputs, labels = Variable(inputs_data), Variable(labels_data)
            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.long)
    
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        
            loss.backward()
            optimizer.step()
        
            running_epoch_loss += loss.item()
            running_loss_print += loss.item()
            if i%500 == 499:
                print('%d epoch, %5d num, loss:%.3f' %(epoch+1, i+1, running_loss_print/500) )
                running_loss_print = 0.
            
            _, pred = torch.max(outputs, 1)
            epoch_total_num += labels.size(0)
            correct_num += (pred==labels).sum()
        
        
        train_loss[epoch] = running_epoch_loss/num_train_batch
        train_acc[epoch] = correct_num/epoch_total_num*100
    
    
        # test accuracy and loss
        ts_runningloss_epoch = 0.
        ts_correct = 0
        ts_epoch_num = 0
        for data in testloader:
            inputs_data, labels_data = data
            inputs, labels = Variable(inputs_data), Variable(labels_data)
            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.long)
        
            net.eval()
            ts_outputs = net(inputs)
            loss = criterion(outputs, labels)
        
            ts_runningloss_epoch += loss.item()
            _, ts_pred = torch.max(ts_outputs, 1)
        
            ts_epoch_num += label.size(0)
            ts_correct += (ts_pred==labels).sum()
        test_loss[epoch] = ts_runningloss_epoch/num_test_batch
        test_acc[epoch] = ts_correct/ts_epoch_num*100
    
        net.train()
        print(" num %d epoch " %epoch)
        print("####### Training Loss #######")
        print(train_loss[epoch])
        print("####### Training Accuracy #######")
        print(train_acc[epoch])
        print("####### Testing Loss #######")
        print(test_loss[epoch])
        print("####### Testing Accuracy #######")
        print(test_acc[epoch])
    
    print('finish training \n')


    #############################################################################################
    print('now begin saving datum for next step plotting')
    save_path = '../datum_for_plotting/' + str(run_num)+ net_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(save_path + 'train_loss.save', 'wb')
    cPickle.dump(train_loss, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    f = open(save_path + 'train_acc.save', 'wb')
    cPickle.dump(train_acc, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    f = open(save_path + 'test_loss.save', 'wb')
    cPickle.dump(test_loss, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    f = open(save_path + 'test_acc.save', 'wb')
    cPickle.dump(test_acc, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    torch.save(net, save_path+'BaseNet_A.pkl') # save whole net structure and params
    torch.save(net.state_dict, save_path+'BaseNet_A_params.pkl') # only save model params
 
    
    
    
    #############################################################################################    
    print("now plotting accuracies and losses")  
    itern_axis_train = np.array(np.linspace(1,max_epoch,num=max_epoch))
    itern_axis_test = np.array(np.linspace(1,max_epoch, num=max_epoch))

    #Train and Test Accuracy
    fig, ax=plt.subplots()
    ax.plot(itern_axis_train, train_acc,'-b.', label='Train')
    ax.plot(itern_axis_test, test_acc, '--r', label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy vs Epoch')
    legend = ax.legend(loc='upper center', shadow=True)
    plt.ylim((0,100))
    plt.savefig(save_path + 'Accuracy_epoch_' + str(max_epoch) + '.png')
    plt.show()

    #Train and Test Cost
    fig, ax=plt.subplots()
    ax.plot(itern_axis_train, train_loss,'-b.', label='Train')
    ax.plot(itern_axis_test, test_loss, '--r', label='Test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss vs Epoch')
    #ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    legend = ax.legend(loc='upper center', shadow=True)
    plt.ylim((0,5))
    plt.savefig(save_path + 'Loss_epoch_' + str(max_epoch) +'.png')
    plt.show()    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--net_name', type=str, default='BaseNet_C')
    
    args = parser.parse_args()
    
    if(args.net_name == 'BaseNet_A'):
        net = ALL_Conv.BaseNet_A()
    elif(args.net_name == 'BaseNet_B'):
        net = ALL_Conv.BaseNet_B()
    elif(args.net_name == 'BaseNet_C'):
        net = ALL_Conv.BaseNet_C()
        
    run_model(num, args.net_name, args.run_num)

if __name__ == '__main__':
    main()
       