from MyDataset.Datasets import *
from Utils.utils import *
from model_w_net import *
from Utils.Vidsom import *
from torch.utils import data as Data
from Utils.SSIM import SSIM

import torch.optim as optim
import torch.nn as nn
import argparse
import os

parser = argparse.ArgumentParser(description="Deraining")
parser.add_argument("--batchSize", type=int, default=16,
                    help="training batch size")
parser.add_argument("--train_epoch", type=int, default=300, help="number of epochs to train for")
parser.add_argument("--test_freq", type=int, default=1, help="test freq ")
parser.add_argument("--train_print_freq", type=int, default=100, help="train print freq ")


parser.add_argument("--Adam_lr_derain", type=float, default=1e-4, help="Adam derain Learning Rate")
parser.add_argument("--Adam_lr_classfy", type=float, default=1e-6, help="Adam classfy Learning Rate")

parser.add_argument("--cuda", type=str, default="Ture", help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers")

# checkpoints and dataset path
parser.add_argument("--save_root", default="../checkpoints", type=str, help="path to save networks")
parser.add_argument("--load_root", default="../checkpoints", type=str, help="path to load networks")

parser.add_argument("--train", default="/home/psdz/桌面/Derain_Data/train", type=str, help="train data root")
parser.add_argument("--test", default="/home/psdz/桌面/Derain_Data/test", type=str, help="test data root")



def main():
    #parametre

    opt = parser.parse_args()
    print(opt)

    batch_size = opt.batch_size
    num_works = opt.num_workers

    #freq_print

    test_freq = opt.test_freq
    train_epoch = opt.train_epoch
    train_print_freq = opt.train_print_freq

    # path
    train_data_root = opt.train
    test_data_root = opt.test
    save_root = opt.save_root
    load_root = opt.load_root


    stage1 = Rain_steaks_with_BG()
    weights = torch.load(load_root + '/stage1/'+'213.pth')
    stage1.load_state_dict(weights['state_dict'])

    classfy = Classfication()
    weights = torch.load(load_root + '/classfy/'+'213.pth')
    classfy.load_state_dict(weights['state_dict'])

    stage2 = Low_BackGround()
    weights = torch.load(load_root + '/stage2/' + '213.pth')
    stage2.load_state_dict(weights['state_dict'])

    derain = Derain()
    weights = torch.load(load_root + '/derain/' + '213.pth')
    derain.load_state_dict(weights['state_dict'])

    #  critersion and optimizer

    criter_loss_MSE= nn.MSELoss()
    criter_classfy = nn.MSELoss()
    criter_loss_ssim = SSIM()


    Adam_lr_derain = opt.Adam_lr_derain
    Adam_lr_classfy = opt.Adam_lr_classfy

    optimizer_stage1 = optim.Adam(stage1.parameters(), lr=Adam_lr_derain)
    optimizer_classfy = optim.Adam(classfy.parameters(),  lr=Adam_lr_classfy )
    optimizer_stage2 = optim.Adam(stage2.parameters(), lr=Adam_lr_derain)
    optimizer_derain =optim.Adam(derain.parameters(), lr=Adam_lr_derain)

    # dataloader


    train_datasets = derain_train_datasets(train_data_root)
    train_dataloader = Data.DataLoader(
        train_datasets,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_works,
    )

    test_datasets = derain_test_datasets(test_data_root)
    test_dataloader = Data.DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=True,
        num_workers=num_works
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
            train_loss  =  train_img_loss + train_classfy_loss + 0.1 *train_ssim_loss

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


        if epoch == 100 :
            Adam_lr_derain /= 10

        if epoch % test_freq == 0:
            print("------> testing")
            stage1.eval()
            stage2.eval()
            derain.eval()

            test_Psnr_sum = 0.0
            test_Ssim_sum = 0.0

            #showing list
            test_Psnr_loss = []
            test_Ssim_loss = []
            for test_step, (data, label,data_path ) in enumerate(test_dataloader,1):

                data = data.clone().detach().requires_grad_(True).cuda()
                label = label.cuda()
                Rain_High_data = stage1(data)
                img_low_backgroung = stage2(data)
                out = derain(Rain_High_data, img_low_backgroung, data).cuda()

                Psnr , Ssim = get_psnr_ssim(out, label)
                test_Psnr_sum +=Psnr
                test_Ssim_sum += Ssim
                loss = criter_loss_MSE ( out , label)
                # loss.append
                if test_step% 100 == 0:
                    print("epoch={}  Psnr={}  Ssim={} loss{}".format(epoch ,Psnr ,Ssim,loss.item()))
                    test_Psnr_loss.append(test_Psnr_sum / test_step)
                    test_Ssim_loss.append(test_Ssim_sum / test_step)

            print("epoch={}  avr_Psnr ={}  avr_Ssim={}".format(epoch ,test_Psnr_sum / test_step , test_Ssim_sum / test_step))

            # visdom showing
            print("---->testing over show in visdom")
            display_Psnr_Ssim(Psnr_list= test_Psnr_loss ,Ssim_list= test_Ssim_loss , v_epoch= epoch)
            updata_epoch_loss_display(img_loss , classfy_loss  , v_epoch = epoch)

        print("epoch {} train over-----> save model".format(epoch))
        print("saving checkpoint      save_root{}".format(save_root))
        save_checkpoint(root = save_root,model=stage1, epoch=epoch, model_stage="stage1")
        save_checkpoint(root = save_root,model=classfy, epoch=epoch, model_stage="classfy")
        save_checkpoint(root = save_root,model=stage2, epoch=epoch, model_stage="stage2")
        save_checkpoint(root = save_root,model = derain , epoch = epoch , model_stage="derain")
        print("finish save epoch{} checkporint".format({epoch}))

    else:
        print("all epoch is over ------ ")
        print("show epoch and epoch_loss in visdom")


if __name__ == "__main__" :
    os.system('clear')
    main()
