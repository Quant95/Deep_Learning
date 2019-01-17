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
from torch.utils.data import sampler
import torch.utils.model_zoo as model_zoo


class BaseNet_A(nn.Module):
    def __init__(self):
        super (BaseNet_A, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        #self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv2.weight)
        #self.relu2 = nn.ReLU(inplace=True)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(192, 192, kernel_size=3, padding=3)
        nn.init.kaiming_normal_(self.conv3.weight)
        #self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.conv5 = nn.Conv2d(192, 10, kernel_size=1)
        nn.init.kaiming_normal_(self.conv5.weight)
        
        self.glb_avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        #out = self.relu1(self.conv1(x))
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        
        out = self.maxpool1(out)
        out = self.dropout2(out)
        
        out = self.conv2(out)
        #out = self.relu2(out)
        out = F.relu(out)
        
        out = self.maxpool2(out)
        out = self.conv3(out)
        #out = self.relu3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        
        out = self.glb_avg(out)
        out = out.view(-1, 10)
        return out

    
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc
    
train_batch_size = 4
test_batch_size = 4
NUM_TRAIN = 49000

'''
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
'''

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

USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)
    
net = BaseNet_A()
net = net.to(device=device)
criterion = nn.CrossEntropyLoss()        
        
#lr_1, lr_2, lr_3, lr_4 = 0.25, 0.1, 0.05, 0.01
lr_1, lr_2, lr_3, lr_4 = 0.01, 0.005, 0.001, 0.001*0.8
weight_decay = 0.001

max_epoch = 350
display_interval = 500

train_size = 50000
test_size = 10000

num_train_batch = train_size/train_batch_size
num_test_batch = test_size/test_batch_size

train_loss = np.zeros((max_epoch,1))
val_acc = np.zeros((max_epoch,1))
#train_acc = np.zeros((max_epoch,1))
#test_loss = np.zeros((max_epoch,1))
#test_acc = np.zeros((max_epoch,1))

epoch_acc = [] # max_epoch x num
print("begin training")
# epoch in paper [200, 250, 300]
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
    
    i_acc = []
    #for i, data in enumerate(trainloader):
    for i, data in enumerate(loader_train):
        net.train()
        
        inputs_data, labels_data = data
        inputs, labels = Variable(inputs_data), Variable(labels_data)
        inputs = inputs.to(device=device, dtype=dtype)
        labels = labels.to(device=device, dtype=torch.long)
        
    
        
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_epoch_loss += loss.item()
        #running_loss_print += loss.item()
        if i%500 == 499:
            print('%d epoch, %5d iteration, loss:%.3f' %(epoch+1, i+1, running_epoch_loss) )
            acc = check_accuracy(loader_val, net)
            i_acc.append(acc)
            #print('%d epoch, %5d iteration, loss:%.3f' %(epoch+1, i+1, running_loss_print/500) )
            #running_loss_print = 0.
            
        #_, pred = torch.max(outputs, 1)
        #epoch_total_num += labels.size(0)
        #correct_num += (pred==labels).sum()
        
        
    train_loss[epoch] = running_epoch_loss/num_train_batch
    epoch_acc.append(i_acc)
    val_acc[epoch] = np.sum(epoch_acc[epoch])
    #train_acc[epoch] = correct_num/epoch_total_num*100
    
    
    # test accuracy and loss
    '''
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
        
        ts_epoch_num += labels.size(0)
        ts_correct += (ts_pred==labels).sum()
    test_loss[epoch] = ts_runningloss_epoch/num_test_batch
    test_acc[epoch] = ts_correct/ts_epoch_num*100
    '''
    print(" num %d epoch " %epoch)
    print("####### Training Loss #######")
    print(train_loss[epoch])
    print("####### Validation Accuracy #######")
    print(val_acc[epoch])
    #print("####### Training Accuracy #######")
    #print(train_acc[epoch])
    #print("####### Testing Loss #######")
    #print(test_loss[epoch])
    #print("####### Testing Accuracy #######")
    #print(test_acc[epoch])
    
print('finish training \n')

test_acc = check_accuracy(loader_test, net)

'''
print("Now test on whole test dataset")
wh_ts_correct = 0
wh_ts_loss = 0
for data in testloader:
    images, lables = data
    outs = net(Variable(images))
    loss = criterion(outs, labels)
    wh_ts_loss += loss
    _, wh_pred = torch.max(outs, 1)
    wh_ts_correct += (wh_pred==labels).sum()
wh_ts_correct = wh_ts_correct/num_test_batch
'''



##################################################################################################
print('now begin saving datum for next step plotting')
run_num =1
save_path = '../datum_for_plotting/' + str(run_num)+'basenet_A/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
f = open(save_path + 'train_loss.save', 'wb')
cPickle.dump(train_loss, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f = open(save_path + 'val_acc.save', 'wb')
cPickle.dump(val_acc, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f = open(save_path + 'epoch_acc.save', 'wb')
cPickle.dump(epoch_acc, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

array_epoch_acc = np.array(epoch_acc)
f = open(save_path + 'array_epoch_acc.save', 'wb')
cPickle.dump(array_epoch_acc, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f = open(save_path + 'test_acc.save', 'wb')
cPickle.dump(test_acc, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

#f = open(save_path + 'train_acc.save', 'wb')
#cPickle.dump(train_acc, f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()
#f = open(save_path + 'test_loss.save', 'wb')
#cPickle.dump(test_loss, f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()
#f = open(save_path + 'test_acc.save', 'wb')
#cPickle.dump(test_acc, f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()

torch.save(net, save_path+'BaseNet_A.pkl') # save whole net structure and params
torch.save(net.state_dict, save_path+'BaseNet_A_params.pkl') # only save model params
 
    
    
    
##################################################################################################    
print("now plotting accuracies and losses")  
itern_axis_train = np.array(np.linspace(1,max_epoch,num=max_epoch))
itern_axis_test = np.array(np.linspace(1,max_epoch, num=max_epoch))

plt.plot(itern_axis_train, train_loss,'-b.', label='Train')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.savefig(save_path + 'train_loss' + str(max_epoch) + '.png')

plt.plot(itern_axis_test, test_loss, '--r', label='Test')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.savefig(save_path + 'validation_accuracy' + str(max_epoch) + '.png')

'''
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
'''