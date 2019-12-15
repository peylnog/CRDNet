from MyDataset.Datasets import  *
from Utils.utils import *
from model_w_net import *
from Utils.Vidsom import *
from torch.utils import data as Data
from config import DefaultConfig
from Utils.SSIM import SSIM

import torch.optim as optim
import torch.nn as nn
import argparse


def train():
    #parametre

    opt = DefaultConfig()

    batch_size = opt.batch_size
    num_works = opt.num_workers

    #freq_print

    test_freq = opt.test_freq
    train_epoch = opt.train_epoch
    train_print_freq = opt.train_print_freq

    # path
    train_data_root = opt.train_data_root
    test_data_root = opt.test_data_root
    save_root = opt.save_root
    load_root = opt.load_root


    stage1 = Rain_steaks_with_BG()
    weights = torch.load(load_root + '/stage1/'+'241.pth')
    stage1.load_state_dict(weights['state_dict'])

    classfy = Classfication()
    weights = torch.load(load_root + '/classfy/'+'241.pth')
    classfy.load_state_dict(weights['state_dict'])

    stage2 = Low_BackGround()
    weights = torch.load(load_root + '/stage2/' + '241.pth')
    stage2.load_state_dict(weights['state_dict'])

    derain = Derain()
    weights = torch.load(load_root + '/derain/' + '241.pth')
    derain.load_state_dict(weights['state_dict'])

    #  critersion and optimizer

    criter_loss_MSE= nn.MSELoss()
    criter_classfy = nn.MSELoss()
    criter_loss_ssim = SSIM()


    Adam_lr_refine = opt.Adam_lr_refine
    Adam_lr_classfy = opt.Adam_lr_classfy

    optimizer_stage1 = optim.Adam(stage1.parameters(), lr=Adam_lr_refine)
    optimizer_classfy = optim.Adam(classfy.parameters(),  lr=Adam_lr_classfy )
    optimizer_stage2 = optim.Adam(stage2.parameters(), lr=Adam_lr_refine)
    optimizer_derain =optim.Adam(derain.parameters(), lr=Adam_lr_refine)

    # dataloader


    train_datasets = derain_train_datasets(train_data_root)
    train_dataloader = Data.DataLoader(
        train_datasets,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_works,
    )



    # training
    for epoch in range(1,train_epoch):
        stage1.cuda()
        classfy.cuda()
        stage2.cuda()
        derain.cuda()

        classfy_loss =[]
        img_loss = []
        for step,(data,label,classfy_label) in enumerate(train_dataloader, 0):
            if opt.test == True:
                break
            stage1.train()
            classfy.train()
            stage2.train()
            derain.train()

            data = data.clone().detach().requires_grad_(True).cuda()
            label=label.cuda()
            classfy_label = classfy_label.cuda()


            Rain_High_data = stage1(data).cuda()
            classfy_data = classfy(Rain_High_data)
            img_low_backgroung = stage2(data).cuda()

            #model output
            out = derain(Rain_High_data,img_low_backgroung ,data).cuda()
            #loss
            train_img_loss = criter_loss_MSE(out , label)
            train_classfy_loss = criter_classfy(classfy_data , classfy_label)
            train_ssim_loss=  1- criter_loss_ssim (out , label)
            train_loss  =  train_img_loss + train_classfy_loss + 0.02 *train_ssim_loss

            optimizer_classfy.zero_grad()
            optimizer_stage1.zero_grad()
            optimizer_stage2.zero_grad()
            optimizer_derain.zero_grad()

            train_loss.backward()

            optimizer_classfy.step()
            optimizer_stage1.step()
            optimizer_stage2.step()
            optimizer_derain.step()

            if step % train_print_freq==0:
                print("epoch{} step {} Img_loss{} classfy_loss{} ssim_loss{}" .format(epoch, step , train_img_loss.item(),train_classfy_loss.item(),train_ssim_loss.item()))
            if step % 50 == 0 :
                img_loss.append(train_img_loss.item())
                classfy_loss.append(train_classfy_loss.item())

    else:
        print("all epoch is over ------ ")
        print("show epoch and epoch_loss in visdom")


if __name__ == "__main__" :
    train()
