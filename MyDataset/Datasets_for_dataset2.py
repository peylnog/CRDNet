
from os import listdir
from os.path import join
from PIL import Image
from os.path import basename
import torch.utils.data as Data
from torchvision import transforms as TF
import torch
import numpy as np
from Utils.utils import get_mean_and_std
import re




def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class derain_train_datasets(Data.Dataset):
    """dataroot /home/psdz/桌面/Derain_Dataset3/train"""
    def __init__(self, data_root ):
        super(derain_train_datasets, self).__init__()
        self.data_filenames = [join(data_root + "/rain/", x) for x in listdir(data_root+"/rain/") if is_image_file(x) ]
        self.transform = TF.Compose(
            [

                TF.CenterCrop(320),
                # TF.RandomHorizontalFlip(0.1),
                TF.ToTensor(),  # tensor range for 0 to 1
                # TF.Normalize(mean = [0.4975, 0.4887, 0.4604] , std= [0.2246, 0.2232, 0.2283]) ,
            ]
        )


    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        info_lis = re.findall(r'\d+', data_path)
        image_number  = int(info_lis [2])
        image_density = int(info_lis[3])
        if image_density ==1:
            classfy_label = [0,0,1] #light
        if image_density ==2 :
            classfy_label = [0,1,1] #Middle
        if image_density ==3 :
            classfy_label = [1,1,1] #Middle

        label_data_path = "/home/psdz/桌面/Derain_Dataset2/train/clear/%d.jpg"%image_number
        data = Image.open(data_path)
        label = Image.open(label_data_path)

        data = self.transform(data)
        label = self.transform(label)
        classfy_label = torch.tensor(classfy_label).float()



        return data, label , classfy_label

    def __len__(self):
        return len(self.data_filenames)


class derain_test_datasets(Data.Dataset):
    """dataroot /home/psdz/桌面/Derain_Dataset2/train"""

    def __init__(self, data_root):
        super(derain_test_datasets, self).__init__()
        self.data_filenames = [join(data_root + "/rain/", x) for x in listdir(data_root + "/rain/") if is_image_file(x)]
        self.transform = TF.Compose(
            [

                TF.ToTensor(),  # tensor range for 0 to 1
            ]
        )

    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        info_lis = re.findall(r'\d+', data_path)
        image_number = int(info_lis[2])


        label_data_path = "/home/psdz/桌面/Derain_Dataset2/test/clear/%d.jpg" % image_number
        data = Image.open(data_path)
        label = Image.open(label_data_path)

        data = self.transform(data)
        label = self.transform(label)

        return data, label , data_path

    def __len__(self):
        return len(self.data_filenames)
