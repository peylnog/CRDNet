from MyDataset.Datasets import  *
from model_w_net import *
from Utils.Vidsom import *
from torch.utils import data as Data
from config import DefaultConfig

import torch.nn as nn


def test():
    #parametre

    opt = DefaultConfig()
    num_works = opt.num_workers



    # path
    test_data_root = opt.test_data_root
    load_root = opt.load_root
    save_image_root = opt.save_image_root

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

    #Dataloader

    test_datasets = derain_test_datasets(test_data_root)
    test_dataloader = Data.DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=True,
        num_workers=num_works
    )
    criter_loss_MSE = nn.MSELoss()

    #######test#######

    print("------> testing")


    stage1.cuda()
    stage2.cuda()
    derain.cuda()
    stage1.eval()
    stage2.eval()
    derain.eval()

    test_Psnr_sum = 0.0
    test_Ssim_sum = 0.0

    #showing list
    test_Psnr_loss = []
    test_Ssim_loss = []
    dict_psnr_ssim = {}
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

        if opt.save_image == True :
            dict_psnr_ssim["Psnr%s_Ssim%s"%(Psnr , Ssim)] = data_path
            out =  out.cpu().data[0]
            out = ToPILImage()(out)
            image_number =   re.findall(r'\d+', data_path[0])[0]
            out.save(save_image_root + "/dataset1_%s.jpg"%image_number)

        # loss.append
        if test_step% 100 == 0:
            print("Psnr={}  Ssim={} loss{}".format(Psnr ,Ssim,loss.item()))
            test_Psnr_loss.append(test_Psnr_sum / test_step)
            test_Ssim_loss.append(test_Ssim_sum / test_step)
#
    print(" avr_Psnr ={}  avr_Ssim={}".format(test_Psnr_sum / test_step , test_Ssim_sum / test_step))


if __name__ == "__main__" :
    test()
