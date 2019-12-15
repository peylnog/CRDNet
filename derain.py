from MyDataset.Datasets import  *
from Utils.utils import *
from model_Dense_w_net_modelfy_dialate import *
from Utils.Vidsom import *
from torch.utils import data as Data
from config import DefaultConfig
from Utils.SSIM import SSIM

import torch.optim as optim
import torch.nn as nn
import argparse


def main():
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
    weights = torch.load(save_root + '/stage1/'+'339.pth')
    stage1.load_state_dict(weights['state_dict'])

    classfy = Classfication()
    weights = torch.load(save_root + '/classfy/'+'339.pth')
    classfy.load_state_dict(weights['state_dict'])

    stage2 = Low_BackGround()
    weights = torch.load(save_root + '/stage2/' + '339.pth')
    stage2.load_state_dict(weights['state_dict'])

    refine = Refine()
    weights = torch.load(save_root + '/refine/' + '339.pth')
    refine.load_state_dict(weights['state_dict'])

    #  critersion and optimizer

    criter_loss_MSE= nn.MSELoss()
    criter_classfy = nn.MSELoss()
    criter_loss_ssim = SSIM()


    Adam_lr_refine = opt.Adam_lr_refine
    Adam_lr_classfy = opt.Adam_lr_classfy

    optimizer_stage1 = optim.Adam(stage1.parameters(), lr=Adam_lr_refine)
    optimizer_classfy = optim.Adam(classfy.parameters(),  lr=Adam_lr_classfy )
    optimizer_stage2 = optim.Adam(stage2.parameters(), lr=Adam_lr_refine)
    optimizer_refine =optim.Adam(refine.parameters(), lr=Adam_lr_refine)

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
    for epoch in range(340,train_epoch):
        stage1.cuda()
        classfy.cuda()
        stage2.cuda()
        refine.cuda()

        classfy_loss =[]
        img_loss = []
        for step,(data,label,classfy_label) in enumerate(train_dataloader, 0):
            if opt.test == True:
                break
            stage1.train()
            classfy.train()
            stage2.train()
            refine.train()

            data = data.clone().detach().requires_grad_(True).cuda()
            label=label.cuda()
            classfy_label = classfy_label.cuda()


            Rain_High_data = stage1(data).cuda()
            classfy_data = classfy(Rain_High_data)
            img_low_backgroung = stage2(data).cuda()

            #model output
            out = refine(Rain_High_data,img_low_backgroung ,data).cuda()
            #loss
            train_img_loss = criter_loss_MSE(out , label)
            train_classfy_loss = criter_classfy(classfy_data , classfy_label)
            train_ssim_loss=  1- criter_loss_ssim (out , label)

            train_loss  =  train_img_loss + train_classfy_loss + 0.005 *train_ssim_loss

            optimizer_classfy.zero_grad()
            optimizer_stage1.zero_grad()
            optimizer_stage2.zero_grad()
            optimizer_refine.zero_grad()

            train_loss.backward()

            optimizer_classfy.step()
            optimizer_stage1.step()
            optimizer_stage2.step()
            optimizer_refine.step()

            if step % train_print_freq==0:
                print("epoch{} step {} Img_loss{} classfy_loss{} ssim_loss{}" .format(epoch, step , train_img_loss.item(),train_classfy_loss.item(),train_ssim_loss.item()))
            if step % 50 == 0 :
                img_loss.append(train_img_loss.item())
                classfy_loss.append(train_classfy_loss.item())

        if epoch == 100:
            Adam_lr_refine /= 3   #1e-4

        if epoch ==500 :
            Adam_lr_refine /= 5

        if epoch % test_freq == 0:
            print("------> testing")
            stage1.eval()
            stage2.eval()
            refine.eval()

            test_Psnr_sum = 0.0
            test_Ssim_sum = 0.0

            #showing list
            test_classfy_list = []
            test_Psnr_loss = []
            test_Ssim_loss = []
            dict_psnr_ssim = {}
            for test_step, (data, label,data_path ) in enumerate(test_dataloader,1):

                data = data.clone().detach().requires_grad_(True).cuda()
                label = label.cuda()
                Rain_High_data = stage1(data)
                img_low_backgroung = stage2(data)
                out = refine(Rain_High_data, img_low_backgroung, data).cuda()

                Psnr , Ssim = get_psnr_ssim(out, label)
                test_Psnr_sum +=Psnr
                test_Ssim_sum += Ssim
                loss = criter_loss_MSE ( out , label)

                if opt.save_image == True :
                    dict_psnr_ssim["Psnr%s_Ssim%s"%(Psnr , Ssim)] = data_path
                    out =  out.cpu().data[0]
                    out = ToPILImage()(out)
                    image_number =   re.findall(r'\d+', data_path[0])[0]
                    out.save("/home/psdz/桌面/excellent_result/psnr>35/dataset1_%s.jpg"%image_number)

                # loss.append
                if test_step% 100 == 0:
                    print("epoch={}  Psnr={}  Ssim={} loss{}".format(epoch ,Psnr ,Ssim,loss.item()))
                    #test_Psnr_loss.append(test_Psnr_sum / test_step)
                    #test_Ssim_loss.append(test_Ssim_sum / test_step)
    #
            print("epoch={}  avr_Psnr ={}  avr_Ssim={}".format(epoch ,test_Psnr_sum / test_step , test_Ssim_sum / test_step))
    #
            # visdom showing
            print("---->testing over show in visdom")
            display_Psnr_Ssim(Psnr_list= test_Psnr_sum / test_step ,Ssim_list= test_Ssim_sum / test_step , v_epoch= epoch)
            updata_epoch_loss_display(img_loss , classfy_loss  , v_epoch = epoch)

        print("epoch {} train over-----> save model".format(epoch))
        print("saving checkpoint      save_root{}".format(save_root))
        save_checkpoint(root = save_root,model=stage1, epoch=epoch, model_stage="stage1")
        save_checkpoint(root = save_root,model=classfy, epoch=epoch, model_stage="classfy")
        save_checkpoint(root = save_root,model=stage2, epoch=epoch, model_stage="stage2")
        save_checkpoint(root = save_root,model = refine , epoch = epoch , model_stage="refine")
        print("finish save epoch{} checkporint".format({epoch}))
    #
    else:
        print("all epoch is over ------ ")
        print("show epoch and epoch_loss in visdom")


if __name__ == "__main__" :
    main()
