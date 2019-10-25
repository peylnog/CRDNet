import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.utils import get_residue,rgb2grad
import numpy as np




def weights_init_kaiming(m):

    classname = m.__class__.__name__  # return classname and type is str
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data, a=0, mode='fan_in')  # kaiming init method
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight.data, a=0, mode='fan_in')


class MeanShift(nn.Conv2d):
    """
    Normalization operation
    """
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False


def conv_blocks_size_3(in_dim,out_dim, Use_pool = False ,Maxpool = True,Dropout = False):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 3 , padding= 1, stride=1))
    layer.add_module('relu', nn.ReLU(False))
    if Dropout:
        layer.add_module('dropout' , nn.Dropout(0.5))
    layer.add_module("bn", nn.BatchNorm2d(out_dim))

    layer.add_module("conv2", nn.Conv2d(out_dim ,out_dim, kernel_size= 3, padding= 1, stride= 1))
    layer.add_module('relu', nn.ReLU(False))
    if Dropout:
        layer.add_module('dropout' , nn.Dropout(0.5))
    layer.add_module("bn", nn.BatchNorm2d(out_dim))
    if Use_pool :
        if Maxpool:
            layer.add_module("Maxpool",nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layer.add_module("Avgpool", nn.AvgPool2d(kernel_size=2, stride=2))
    return layer


def conv_blocks_size_5(in_dim,out_dim, Use_pool = False ,Maxpool = True ,Dropout = False):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 5, padding= 2, stride=1))
    layer.add_module('relu', nn.ReLU(False))
    if Dropout:
        layer.add_module('dropout' , nn.Dropout(0.5))
    layer.add_module("bn", nn.BatchNorm2d(out_dim))


    layer.add_module("conv2", nn.Conv2d(out_dim ,out_dim, kernel_size= 5, padding= 2, stride= 1))
    layer.add_module('relu', nn.ReLU(False))
    if Dropout:
        layer.add_module('dropout' , nn.Dropout(0.5))
    layer.add_module("bn", nn.BatchNorm2d(out_dim))

    if Use_pool :
        if Maxpool:
            layer.add_module("Maxpool",nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layer.add_module("Avgpool", nn.AvgPool2d(kernel_size=2, stride=2))
    return layer




def conv_blocks_size_7(in_dim,out_dim, Use_pool = False ,Maxpool = True , Dropout = False):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 7 , padding= 3, stride=1))
    layer.add_module('relu', nn.ReLU(False))
    if Dropout:
        layer.add_module('dropout' , nn.Dropout(0.5))
    layer.add_module("bn", nn.BatchNorm2d(out_dim))

    layer.add_module("conv2", nn.Conv2d(out_dim ,out_dim, kernel_size= 7, padding= 3, stride= 1))
    layer.add_module('relu', nn.ReLU(False))
    if Dropout:
        layer.add_module('dropout', nn.Dropout(0.5))
    layer.add_module("bn", nn.BatchNorm2d(out_dim))


    if Use_pool :
        if Maxpool:
            layer.add_module("Maxpool",nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layer.add_module("Avgpool", nn.AvgPool2d(kernel_size=2, stride=2))
    return layer







def deconv_blocks_size_3(in_dim,out_dim,Use_pool=True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.ConvTranspose2d(in_dim , out_dim , 3 , 1, 1))
    layer.add_module('relu', nn.ReLU(False))
    layer.add_module('bn', nn.BatchNorm2d(out_dim))

    layer.add_module("conv2", nn.ConvTranspose2d(out_dim, out_dim ,3, 1, 1))
    layer.add_module('relu', nn.ReLU(False))
    layer.add_module('bn', nn.BatchNorm2d(out_dim))

    if Use_pool:
        layer.add_module("Upsamp",nn.UpsamplingNearest2d(scale_factor= 2))
    return layer


def deconv_block_size_3(in_dim,out_dim,Use_pool=True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.ConvTranspose2d(in_dim , out_dim , 3 , 1, 1))
    layer.add_module('relu', nn.ReLU(False))
    layer.add_module('bn', nn.BatchNorm2d(out_dim))
    if Use_pool:
        layer.add_module("Upsamp",nn.UpsamplingNearest2d(scale_factor= 2))
    return layer


def deconv_blocks_size_5(in_dim,out_dim,Use_pool=True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.ConvTranspose2d(in_dim , out_dim , 5 , 2, 1))
    layer.add_module('relu', nn.relu(True))
    layer.add_module('bn', nn.BatchNorm2d(out_dim))

    layer.add_module("conv2", nn.ConvTranspose2d(out_dim, out_dim ,5, 2, 1))
    layer.add_module('relu', nn.relu(True))
    layer.add_module('bn', nn.BatchNorm2d(out_dim))

    if Use_pool:
        layer.add_module("Upsamp",nn.UpsamplingNearest2d(scale_factor= 2))
    return layer


def deconv_block_size_6(in_dim,out_dim,Use_pool=True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.ConvTranspose2d(in_dim , out_dim , 6 , 2, 2))
    layer.add_module('relu', nn.relu(True))
    layer.add_module('bn', nn.BatchNorm2d(out_dim))
    if Use_pool:
        layer.add_module("Upsamp",nn.UpsamplingNearest2d(scale_factor=2))
    return layer


def deconv_blocks_size_7(in_dim,out_dim,Use_pool=True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.ConvTranspose2d(in_dim , out_dim , 7, 3, 1))
    layer.add_module('relu', nn.ReLU(True))
    layer.add_module('bn', nn.BatchNorm2d(out_dim))

    layer.add_module("conv2", nn.ConvTranspose2d(out_dim, out_dim ,7, 3, 1))
    layer.add_module('relu', nn.ReLU(True))
    layer.add_module('bn', nn.BatchNorm2d(out_dim))

    if Use_pool:
        layer.add_module("Upsamp",nn.UpsamplingNearest2d(scale_factor= 2))
    return layer


def deconv_block_size_8(in_dim,out_dim,Use_pool=True,ELU = True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.ConvTranspose2d(in_dim , out_dim , 8 , 2, 3))
    layer.add_module('relu', nn.relu(True))
    layer.add_module('bn', nn.BatchNorm2d(out_dim))
    if Use_pool:
        layer.add_module("Upsamp",nn.UpsamplingNearest2d(scale_factor= 2))
    return layer


def Nonlinear_layer(in_c , name,   bn=False ,  relu=True, LeakReLU = False , dropout=False ):
    layer = nn.Sequential()
    if relu:
        layer.add_module('%s_relu' % name, nn.ReLU(inplace=False))
    if LeakReLU:
        layer.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=False))
    if bn:
        layer.add_module('%s_bn' % name, nn.BatchNorm2d(in_c))

    if dropout:
        layer.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=False))
    return layer



class Low_BackGround(nn.Module):
    def __init__(self):
        super(Low_BackGround, self).__init__()
        self.encode1 = conv_blocks_size_3(in_dim=1 , out_dim=4 ,Use_pool=True) #256
        weights_init_kaiming(self.encode1)
        self.encode2 = conv_blocks_size_3(in_dim=4 ,out_dim=16,Use_pool=True)   #128
        weights_init_kaiming(self.encode2)
        self.encode3 = conv_blocks_size_3(in_dim=16 ,out_dim=64,Use_pool=True) #64
        weights_init_kaiming(self.encode3)
        self.encode4 = conv_blocks_size_3(in_dim=64,out_dim=128,Use_pool=True) #32
        weights_init_kaiming(self.encode4)
        self.encode5 = conv_blocks_size_3(in_dim=128,out_dim=256,Use_pool=True) # 16
        weights_init_kaiming(self.encode5)
        self.encode6 = conv_blocks_size_5(in_dim=256,out_dim=512,Use_pool=True)#8


        self.decode1 = deconv_blocks_size_3(in_dim=512,out_dim=256,Use_pool=True)
        weights_init_kaiming(self.decode1)
        self.decode2 =deconv_blocks_size_3(512,128,True)
        weights_init_kaiming(self.decode2)
        self.decode3 = deconv_blocks_size_3(256,64,True)
        weights_init_kaiming(self.decode3)
        self.decode4 = deconv_blocks_size_3(128,16,True)
        weights_init_kaiming(self.decode4)
        self.decode5 = deconv_blocks_size_3(32,8,True)
        weights_init_kaiming(self.decode5)
        self.decode6 = deconv_blocks_size_3(12,2,Use_pool=True)
        weights_init_kaiming(self.decode5)
    def forward(self, x):
        # 512 512 x
        x = 0.7 * get_residue(x) + 0.3 * rgb2grad(x)
        encode1 = self.encode1(x)      #4
        encode2 = self.encode2(encode1) #16
        encode3 = self.encode3(encode2) #64
        encode4 = self.encode4(encode3) #128
        encode5 = self.encode5(encode4) #256
        encode6 = self.encode6(encode5) #512
        #U-net


        decode = torch.cat([encode5 , self.decode1(encode6)] , dim = 1) #512
        decode = torch.cat([encode4 , self.decode2(decode)] , dim = 1) #
        decode = torch.cat([encode3,self.decode3(decode)] , dim = 1)
        decode = torch.cat([encode2,self.decode4(decode)] ,dim =1)
        decode = torch.cat([encode1,self.decode5(decode)] ,dim =1)
        decode =torch.cat([x , self.decode6(decode)],dim=1)
        return decode




class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        ##########put input in to big view
        self.refine_1 = deconv_blocks_size_3(6,9, Use_pool=True)
        weights_init_kaiming(self.refine_1)

        self.refine_2 = conv_blocks_size_3(9 , 3, Use_pool= True)
        weights_init_kaiming(self.refine_2)

        self.refine_3 = conv_blocks_size_3(3,3,Use_pool=False)
        weights_init_kaiming(self.refine_3)

        self.refine_4 = conv_blocks_size_3(3,3,Use_pool=False)
        weights_init_kaiming(self.refine_4)

        self.n_3= Nonlinear_layer(in_c=3, name='nonlinear',bn=True,LeakReLU=False,dropout=False)
        self.n_9= Nonlinear_layer(in_c =9 ,name ='nonlinear' ,bn=False,LeakReLU=False,dropout=False)


    def forward(self, x , rain_img ):
        x = torch.cat([rain_img , x ] ,dim = 1)
        x = self.refine_1(x)
        x = self.n_9(x)
        x = self.refine_2(x)
        x = self.n_3(x)
        # x = self.refine_3(x)
        # x = self.n_3(x)
        # x = self.refine_4(x)
        # x = self.n_3(x)
        # # x = self.n1(x)
        return x



class My_classfy(nn.Module):
    def __init__(self):
        super(My_classfy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 256 256 * 64


            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),    # 128 * 128 * 128


            nn.Conv2d(128 , 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 * 64 * 256

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),     # 32*32*512

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16*16*512


            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8*8*512
        )

        self.classfier = nn.Sequential(

            nn.Linear( 512*8*8 , 4096 ),
            nn.ReLU(True),

            nn.Dropout(),
            nn.Linear(4096 , 4096 ),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096 ,3),
            # nn.ReLU(True),
        )
        weights_init_kaiming(self.features)
        weights_init_kaiming(self.classfier)
    def forward(self,x):
        x = self.features(x)
        x = x.view( x.size(0) , -1)
        x = self.classfier(x)
        return  x

