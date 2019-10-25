import torch
import torch.nn as nn
from torch.nn import *



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)





def conv_blocks_size_3(in_dim,out_dim, Use_pool = False ,Maxpool = True,bn = True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 3 , padding= 1, stride=1))
    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))


    layer.add_module( "conv2",nn.Conv2d(out_dim ,out_dim , kernel_size= 3 , padding= 1, stride=1))
    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))


    if Use_pool :
        if Maxpool:
            layer.add_module("Maxpool",nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layer.add_module("Avgpool", nn.AvgPool2d(kernel_size=2, stride=2))
    return layer




def conv_block_size_3_without_bn(in_dim,out_dim, Use_pool = False ,Maxpool = True,bn = True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 3 , padding= 1, stride=1))
    layer.add_module('relu', nn.ReLU(False))

    return layer


def conv_blocks_size_5(in_dim,out_dim, Use_pool = False ,Maxpool = True ,bn = True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 5, padding= 2, stride=1))

    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))

    layer.add_module('relu', nn.ReLU(False))



    layer.add_module( "conv2",nn.Conv2d(out_dim ,out_dim , kernel_size= 5, padding= 2, stride=1))
    # layer.add_module('relu', nn.ReLU(False))

    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))

    if Use_pool :
        if Maxpool:
            layer.add_module("Maxpool",nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layer.add_module("Avgpool", nn.AvgPool2d(kernel_size=2, stride=2))
    return layer




def conv_blocks_size_7(in_dim,out_dim, Use_pool = False ,Maxpool = True , bn = True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 7 , padding= 3, stride=1))
    if bn:
        layer.add_module('bn', nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))


    layer.add_module( "conv2",nn.Conv2d(out_dim ,out_dim , kernel_size= 7 , padding= 3, stride=1))
    if bn:
        layer.add_module('bn', nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))

    if Use_pool :
        if Maxpool:
            layer.add_module("Maxpool",nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layer.add_module("Avgpool", nn.AvgPool2d(kernel_size=2, stride=2))
    return layer




def deconv_blocks_size_3(in_dim,out_dim,Use_pool=True,bn=True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.ConvTranspose2d(in_dim , out_dim , 3 , 1, 1))
    if bn:
        layer.add_module('bn', nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))


    layer.add_module("conv2", nn.ConvTranspose2d(out_dim, out_dim ,3, 1, 1))
    if bn:
        layer.add_module('bn', nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))

    if Use_pool:
        layer.add_module("Upsamp",nn.UpsamplingNearest2d(scale_factor= 2))
    return layer




def Nonlinear_layer(in_c=0  , name="nonlinear",   bn=False ,  relu=True, LeakReLU = False , dropout=False ):
    layer = nn.Sequential()
    if relu:
        layer.add_module('%s_relu' % name, nn.ReLU(inplace=False))
    if LeakReLU:
        layer.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if bn:
        layer.add_module('%s_bn' % name, nn.BatchNorm2d(in_c))

    if dropout:
        layer.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=False))
    return layer
