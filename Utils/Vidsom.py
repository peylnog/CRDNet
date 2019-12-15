import visdom
import torch
from skimage.measure import compare_psnr , compare_ssim
import numpy as np

from torchvision.transforms import ToPILImage , ToTensor


def updata_epoch_loss_display( train_loss, test_loss,v_epoch):
    '''updata train_loss . test_loss for every epoch
      X.dim and Y.dim in line must be >= 1 '''
    epoch_env = visdom.Visdom(env= "two_loss")

    train_loss = torch.tensor(train_loss)
    test_loss = torch.tensor(test_loss)

    Epoch = torch.tensor([i for i in range((v_epoch-1)*len(train_loss)+ 1 ,v_epoch*len(train_loss)+1)])
    if train_loss.dim() == 0 :
        train_loss = torch.tensor([train_loss])
    if test_loss.dim() == 0:
        test_loss = torch.tensor([test_loss])

    #line
    epoch_env.line(X=Epoch ,Y=train_loss , win="img_loss" ,update='append' ,name='img_loss')

    epoch_env.line(X=Epoch,Y=test_loss,win = "classfy_loss",update = 'append',name = 'classfy_loss',)



def test_train_loss_display(test_loss_list , train_loss_list , v_env  , epoch ):
    '''show train_loss , test loss in just a epoch  when testing
    Convert test_loss_list(python_list) to tensor
     test_train_loss_display([1,2,4,3],[3,4,3,2,3,1,31,31,2],1)'''

    test_loss_list = torch.tensor(test_loss_list)
    train_loss_list = torch.tensor(train_loss_list)

    vis = visdom.Visdom(env=v_env)
    train_x = torch.arange(len(train_loss_list))
    test_x = torch.arange(len(test_loss_list))



    vis.line(X=train_x,Y=train_loss_list ,win= "epoch%d_loss"%epoch ,update='append',\
             name = 'train_loss')
    vis.line(X = test_x , Y = test_loss_list , win="epoch%d_loss"%epoch,update='append',\
             name = 'test_loss')


def img_loss_withclassfy_vis_continue(v_epoch , train_loss_list , test_loss_list):
    train_x =torch.arange(v_epoch*len(train_loss_list),(v_epoch+1)*len(train_loss_list))
    test_x = torch.arange(v_epoch * len(test_loss_list), (v_epoch + 1) * len(test_loss_list))

    test_loss_list = torch.tensor(test_loss_list)
    train_loss_list = torch.tensor(train_loss_list)

    loss_vis = visdom.Visdom(env = 'loss')

    loss_vis.line(X=train_x,Y=train_loss_list ,win= "img_loss" ,update='append',\
             name = 'train_loss')
    loss_vis.line(X = test_x , Y = test_loss_list , win="classfy_loss",update='append',\
             name = 'test_loss')


def classfy_loss_vis_continue(v_epoch , train_loss_list , test_loss_list):
    train_x =torch.arange(v_epoch*len(train_loss_list),(v_epoch+1)*len(train_loss_list))
    test_x = torch.arange(v_epoch * len(test_loss_list), (v_epoch + 1) * len(test_loss_list))

    test_loss_list = torch.tensor(test_loss_list)
    train_loss_list = torch.tensor(train_loss_list)

    loss_vis = visdom.Visdom(env = 'classfy_loss')

    loss_vis.line(X=train_x,Y=train_loss_list ,win= "classfy_loss" ,update='append',\
             name = 'train_loss')
    loss_vis.line(X = test_x , Y = test_loss_list , win="classfy_loss",update='append',\
             name = 'test_loss')


def get_psnr_ssim(input_img , compared_img) :
    '''input and compared should be numpy
    batch_size 512 512 3'''
    if not type(input_img) == type(np.array([1])):
        input_img = np.squeeze(input_img.cpu().detach().numpy().transpose(0,2,3,1))

    if not type(compared_img) == type(np.array([1])):
        compared_img =np.squeeze(compared_img.cpu().detach().numpy().transpose(0,2,3,1))

    Ssim = compare_ssim(input_img , compared_img ,data_range = 1 ,  multichannel=True)
    Psnr = compare_psnr(input_img , compared_img ,data_range = 1)
    return Psnr,Ssim

def get_psnr_ssim2(input_img , compared_img):
    input_img = input_img.cpu().data[0]
    compared_img = compared_img.cpu().data[0]
    input_img =  ToPILImage()(input_img)
    compared_img = ToPILImage()(compared_img)

    input_img = np.array(input_img)
    compared_img = np.array(compared_img)

    Ssim = compare_ssim(input_img , compared_img , multichannel=True)
    Psnr = compare_psnr(input_img , compared_img )

    return Psnr, Ssim



def display_Psnr_Ssim( Psnr_list , Ssim_list ,v_epoch):

    Psnr_list = torch.tensor([Psnr_list])
    Ssim_list = torch.tensor([Ssim_list])

    x = torch.tensor([v_epoch])

    loss_vis = visdom.Visdom(env='ssim_psnr')

    loss_vis.line(X=x, Y=Psnr_list, win="Psnr", update='append', \
                  name='train_loss',opts={'title':"Derain_2019_test",'xlabel': 'epoch' , 'ylabel':'Psnr' })
    loss_vis.line(X=x ,Y=Ssim_list, win="Ssim", update='append', \
                  name='test_loss',opts={'title':"Derain_2019_test",'xlabel': 'epoch' , 'ylabel':'Ssim' })



