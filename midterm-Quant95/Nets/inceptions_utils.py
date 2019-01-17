import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.model_zoo as model_zoo



class BasicConv2d(nn.Module):
    
    def __init__(self, channel_in, channel_out, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, 
                              kernel_size=kernel_size,
                              stride = stride,
                              padding = padding, bias=False)
            
        self.bn = nn.BatchNorm2d(num_features = channel_out,
                                 eps = 1e-5, momentum=0.1,
                                 affine = True)
            
        self.relu = nn.ReLU(inplace=True)
            
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
        

class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        
        self.stem0 = nn.Sequential( 
                     BasicConv2d(3, 32, kernel_size=3, stride=2),
                     BasicConv2d(32, 32, kernel_size=3, stride=1),
                     BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1) )
        
        self.stem1_0 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.stem1_1 = BasicConv2d(64, 96, kernel_size=3, stride=2)
        
        self.stem2_0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1) )
        self.stem2_1 = nn.Sequential(
            BasicConv2d(64, 96, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 96, kernel_size=3, stride=1) )
        
        self.stem3_0 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.stem3_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x):
        x = self.stem0(x)
        
        x0 = self.stem1_0(x)
        x1 = self.stem1_1(x)
        x = torch.cat((x0, x1), 1)
        
        x0 = self.stem2_0(x)
        x1 = self.stem2_1(x)
        x = torch.cat((x0, x1), 1)
        
        x0 = self.stem3_0(x)
        x1 = self.stem3_1(x)
        out = torch.cat((x0, x1), 1)
        
        return out
    
class inceptionA(nn.Module):
    def __init__(self):
        super(inceptionA, self).__init__()
        
        self.branch0 = nn.Sequential(
                       nn.AvgPool2d(kernel_size=1, stride=1),
                       BasicConv2d(384, 96, kernel_size=1, stride=1))
        self.branch1 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
                       BasicConv2d(384, 64, kernel_size=1, stride=1),
                       BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(
                       BasicConv2d(384, 64, kernel_size=1, stride=1),
                       BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
                       BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
    
# the schema for 35*35 to 17*17 reduction module
# The k, l, m, n numbers represent filter bank sizes which can be used in variants of network in Net C.
class reductionA(nn.Module):
    def __init__(self, k=192, l=224, m=256, n=384):
        super(reductionA, self).__init__()
        
        self.branch0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0 )
        self.branch1 = BasicConv2d(384, n, kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(
                       BasicConv2d(384, k, kernel_size=1, stride=1),
                       BasicConv2d(k, l , kernel_size=3, stride=1, padding=1),
                       BasicConv2d(l, m, kernel_size=3, stride=2) )
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        
        
class inceptionB(nn.Module):
    def __init__(self):
        super(inceptionB, self).__init__()
        
        self.branch0 = nn.Sequential(
                 nn.AvgPool2d(kernel_size=1, stride = 1),
                 BasicConv2d(1024, 128, kernel_size=1, stride=1) )
        self.branch1 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3)) )
        self.branch3 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(192, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0)) )
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        
class reductionB(nn.Module):
    def __init__(self):
        super(reductionB, self).__init__()
        
        self.branch0 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2) )
        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2) )
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0 ,x1, x2), 1)
        return out
    
class inceptionC(nn.Module):
    def __init__(self):
        super(inceptionC, self).__init__()
        
        self.branch0 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1),
            BasicConv2d(1536, 256, kernel_size=1, stride=1) )
        
        self.branch1 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))     
        self.branch2_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.branch3_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch3_1 = BasicConv2d(384, 448, kernel_size=(1,3), stride=1)
        self.branch3_2 = BasicConv2d(448, 512, kernel_size=(3,1), stride=1)
        self.branch3_3a = BasicConv2d(512, 256, kernel_size=(3,1), stride=1)
        self.branch3_3b = BasicConv2d(512, 256, kernel_size=(1,3), stride=1)
        
    def forward(self, x):
        x0 = self.branch0(x)
        
        x1 = self.branch1(x)
        
        x2_0 = self.branch2_0(x)
        x2_1a = self.branch2_1a(x2_0)
        x2_1b = self.branch2_1b(x2_0)
        x2 = torch.cat((x2_1a, x2_1b), 1)
        
        x3_ = self.branch3_0(x)
        x3_ = self.branch3_1(x3_)
        x3_ = self.branch3_2(x3_)
        x3_a = self.branch3_3a(x3_)
        x3_b = self.branch3_3b(x3_)
        x3 = torch.cat((x3_a, x3_b), 1)
        
        out = torch.cat((x0, x1, x2, x3), 1)
        
        
class Inception_V4(nn.Module):
    def __init__(self, num_classes=1001):
        super(Inception_V4, self).__init__()
        
        self.input_space = None
        self.input_size = (299,299,3)
        self.mean = None
        self.std = None
        
        self.feature = nn.Sequential(
            Stem(),
            inceptionA(),
            inceptionA(),
            inceptionA(),
            inceptionA(),
            reductionA(),
            inceptionB(),
            inceptionB(),
            inceptionB(),
            inceptionB(),
            inceptionB(),
            inceptionB(),
            inceptionB(),
            reductionB(),
            inceptionC(),
            inceptionC(),
            inceptionC() )
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride = 1, count_include_pad = False)
        self.last_linear = nn.Linear(1536, num_classes)
        
    def logits(self, features):
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    
    def forward(self, inputs):
        x = self.features(inputs)
        x = self.logits(x)
        return x

    
    
    
Setting_InceptionV4 = {
    'inceptionv4': {
        'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
        'input_space': 'RGB',
        'input_size': [3, 299, 299],
        'input_range': [0, 1],
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'num_classes': 1000, }
     }




"""
class Model_inception_v4(object):
    
    def __init__(self, num_classes, transfer_learning=False, model=None):
        # when using transfer_learning, need to input model
        
        self.transfer_learning = transfer_learning
        
        if transfer_learning:
            pass
        else:
            setting = Setting_InceptionV4['inceptionv4']
            model = Inception_V4(num_classes=10)
            pretrained_model = Inception_V4(num_classes=1001)
            pretrained_model.load_state_dict(model_zoo.load_url(setting['url']))
        
            new_last_linear = nn.Linear(1536, 10)
            new_last_linear.weight.data = pretrained_model.last_linear.weight.data[1001-1-10:]
            new_last_linear.bias.data = pretrained_model.last_linear.bias.data[1001-1-10:]
            model.last_linear = new_last_linear
            
            params = pretrained_model.state_dict()
            for i, v in params.items():
                print("%s" %i)


def inception_v4(nn_classes, transfer_learning=False, model=None):
# when using transfer_learning, need to input model

    if transfer_learning:
        pass
    else:
        setting = Setting_InceptionV4['inceptionv4']
        model = Inception_V4(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(setting['url']))
        
        new_last_linear = nn.Linear(1536, 10)
        new_last_linear.weight.data = model.last_linear.weight.data[1001-1-10:]
        new_last_linear.bias.data = model.last_linear.bias.data[1001-1-10:]
        model.last_linear = new_last_linear
        
        

        
        
__all__ = ['InceptionV4', 'inceptionv4']

pretrained_settings = {
    'inceptionv4': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}        
        
"""        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        