from MyDataset.Datasets import *
from Utils.utils import *
from model_w_net import *
from Utils.Vidsom import *
from torch.utils import data as Data
from Utils.SSIM import SSIM

import torch.nn as nn
import argparse
import os

parser = argparse.ArgumentParser(description="Deraining")
parser.add_argument("--cuda", type=bool, default=True, help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers")

# checkpoints and dataset path
parser.add_argument("--load", default="../checkpoints", type=str, help="path to load networks")
parser.add_argument("--test", default="/home/psdz/桌面/Derain_Data/test", type=str, help="test data root")
parser.add_argument("--save_image_root", default="../result", type=str, help="save image root")
parser.add_argument("--save_image", default=True, type=bool, help="save image or not")


def main():
    # parametre

    opt = parser.parse_args()
    print(opt)


    num_works = opt.num_workers
    test_data_root = opt.test
    load_root = opt.load
    save_image = opt.save_image
    save_image_root = opt.save_image_root
    Use_cuda = opt.cuda

    stage1 = Rain_steaks_with_BG()
    weights = torch.load(load_root + '/stage1/' + '213.pth')
    stage1.load_state_dict(weights['state_dict'])

    classfy = Classfication()
    weights = torch.load(load_root + '/classfy/' + '213.pth')
    classfy.load_state_dict(weights['state_dict'])

    stage2 = Low_BackGround()
    weights = torch.load(load_root + '/stage2/' + '213.pth')
    stage2.load_state_dict(weights['state_dict'])

    derain = Derain()
    weights = torch.load(load_root + '/derain/' + '213.pth')
    derain.load_state_dict(weights['state_dict'])

    #  critersion and optimizer

    criter_loss_MSE = nn.MSELoss()
    criter_loss_ssim = SSIM()



    test_datasets = derain_test_datasets(test_data_root)
    test_dataloader = Data.DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=True,
        num_workers=num_works
    )


    print("------> testing")
    if Use_cuda:
        stage1.cuda()
        stage2.cuda()
        derain.cuda()

    stage1.eval()
    stage2.eval()
    derain.eval()

    test_Psnr_sum = 0.0
    test_Ssim_sum = 0.0
    # showing list
    test_Psnr_loss = []
    test_Ssim_loss = []
    for test_step, (data, label, data_path) in enumerate(test_dataloader, 1):

        data = data.clone().detach().requires_grad_(True).cuda()
        label = label.cuda()
        Rain_High_data = stage1(data)
        img_low_backgroung = stage2(data)
        out = derain(Rain_High_data, img_low_backgroung, data).cuda()

        Psnr, Ssim = get_psnr_ssim(out, label)
        test_Psnr_sum += Psnr
        test_Ssim_sum += Ssim
        loss = criter_loss_MSE(out, label)
        if save_image :
            out = out.cpu().data[0]
            out = ToPILImage()(out)
            image_number = re.findall(r'\d+', data_path[0])[0]
            out.save( save_image_root + "/dataset1_%s.jpg" % image_number)
        # loss.append
        if test_step % 100 == 0:
            print("Psnr={}  Ssim={} loss{}".format(Psnr, Ssim, loss.item()))
            test_Psnr_loss.append(test_Psnr_sum / test_step)
            test_Ssim_loss.append(test_Ssim_sum / test_step)

    print("avr_Psnr ={}  avr_Ssim={}".format(test_Psnr_sum / test_step, test_Ssim_sum / test_step))
    print("all images have been tested")

if __name__ == "__main__":
    os.system('clear')
    main()
