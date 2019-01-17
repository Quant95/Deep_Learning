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
import torch.utils.model_zoo as model_zoo


class BaseNet_A(nn.Module):
    def __init__(self):
        super (BaseNet_A, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(192, 192, kernel_size=3, padding=3)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
         
        self.conv4 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        self.conv5 = nn.Conv2d(192, 10, kernel_size=1)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0) 
        
        self.glb_avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        
        out = self.maxpool1(out)
        out = self.dropout2(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        
        out = self.maxpool2(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        
        out = self.glb_avg(out)
        out = out.view(-1, 10)
        return out
    
class BaseNet_B(nn.Module):
    def __init__(self):
        super (BaseNet_B, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        
        self.conv4 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.5)
        
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=3)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        
        self.conv6 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        
        self.conv7 = nn.Conv2d(192, 10, kernel_size=1)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)
        
        self.glb_avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        
        out = self.maxpool1(out)
        out = self.dropout2(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.maxpool2(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        
        out = self.conv6(out)
        out = F.relu(out)
        
        out = self.conv7(out)
        out = F.relu(out)
        
        out = self.glb_avg(out)
        out = out.view(-1, 10)
        return out
    
    
class BaseNet_C(nn.Module):
    def __init__(self):
        super (BaseNet_C, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.5)
        
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=3)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        
        self.conv6 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        
        self.conv7 = nn.Conv2d(192, 10, kernel_size=1)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)
        
        self.glb_avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        
        out = self.maxpool1(out)
        out = self.dropout2(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.maxpool2(out)
        
        out = self.dropout3(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        
        out = self.conv6(out)
        out = F.relu(out)
        
        out = self.conv7(out)
        out = F.relu(out)
        
        out = self.glb_avg(out)
        out = out.view(-1, 10)
        return out
    
class Stride_CNN_C(nn.Module):
    def __init__(self):
        super (Stride_CNN_C, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
       
        self.dropout3 = nn.Dropout(0.5)
        
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=3)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        
        self.conv6 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        
        self.conv7 = nn.Conv2d(192, 10, kernel_size=1)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)
        
        self.glb_avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        
        
        out = self.dropout2(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.dropout3(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        
        out = self.conv6(out)
        out = F.relu(out)
        
        out = self.conv7(out)
        out = F.relu(out)
        
        out = self.glb_avg(out)
        out = out.view(-1, 10)
        return out
    

class ConvPool_CNN_C(nn.Module):
    def __init__(self):
        super (ConvPool_CNN_C, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.5)
        
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=3)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)
        
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.constant_(self.conv8.bias, 0)
        
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1)
        nn.init.kaiming_normal_(self.conv9.weight)
        nn.init.constant_(self.conv9.bias, 0)
        
        self.glb_avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        
        out = self.maxpool1(out)
        out = self.dropout2(out)
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        
        out = self.conv6(out)
        out = F.relu(out)
        
        out = self.maxpool2(out)
        out = self.dropout3(out)
        
        out = self.conv7(out)
        out = F.relu(out)
        
        out = self.conv8(out)
        out = F.relu(out)
        
        out = self.conv9(out)
        out = F.relu(out)
        
        out = self.glb_avg(out)
        out = out.view(-1, 10)
        return out
    

class ALL_CNN_C(nn.Module):
    def __init__(self):
        super (ALL_CNN_C, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        
        self.dropout3 = nn.Dropout(0.5)
        
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=3)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)
        
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.constant_(self.conv8.bias, 0)
        
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1)
        nn.init.kaiming_normal_(self.conv9.weight)
        nn.init.constant_(self.conv9.bias, 0)
        
        self.glb_avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        
        out = self.dropout2(out)
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        
        out = self.conv6(out)
        out = F.relu(out)
        
        out = self.dropout3(out)
        
        out = self.conv7(out)
        out = F.relu(out)
        
        out = self.conv8(out)
        out = F.relu(out)
        
        out = self.conv9(out)
        out = F.relu(out)
        
        out = self.glb_avg(out)
        out = out.view(-1, 10)
        return out


class ALL_CNN_A(nn.Module):
    def __init__(self):
        super (ALL_CNN_A, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.dropout1 = nn.Dropout2d(0.2)
        
        
        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, stride=2)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        
        
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=2)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        self.dropout3 = nn.Dropout(0.5)
        
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=3)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
       
        
        self.conv6 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        
        self.conv7 = nn.Conv2d(192, 10, kernel_size=1)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)
        
        self.glb_avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.dropout3(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        
        out = self.conv6(out)
        out = F.relu(out)
        
        out = self.conv7(out)
        out = F.relu(out)
        
        out = self.glb_avg(out)
        out = out.view(-1, 10)
        return out    

    
class ALL_CNN_B(nn.Module):
    def __init__(self):
        super (ALL_CNN_B, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=2)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv4 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        self.conv5 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)
        
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.5)
        
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=3)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.constant_(self.conv7.bias, 0)
        
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.constant_(self.conv8.bias, 0)
        
        self.glb_avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        
        
        
        out = self.conv6(out)
        out = F.relu(out)
        
        out = self.conv7(out)
        out = F.relu(out)
        
        out = self.conv8(out)
        out = F.relu(out)
        
        out = self.conv9(out)
        out = F.relu(out)
        
        out = self.glb_avg(out)
        out = out.view(-1, 10)
        return out    
    
    
    
cnn_a = ALL_CNN_A()
cnn_b = ALL_CNN_B()
print('finish')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
netb = BaseNet_B()
netc = BaseNet_C()
print("right")