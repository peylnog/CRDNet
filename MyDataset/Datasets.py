
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
    '''return rain_img ,clear , classfy_label'''
    def __init__(self, data_root):
        super(derain_train_datasets, self).__init__()
        self.data_filenames = [join(data_root, x) for x in listdir(data_root) if is_image_file(x) and '._' not in x]

        self.transform = TF.Compose(
                [
                    # TF.RandomVerticalFlip(),
                    # TF.RandomHorizontalFlip(),
                    # TF.RandomCrop(224),
                    TF.ToTensor(),  # tensor range for 0 to 1
                    # TF.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5], )
                ]
        )
    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        data = Image.open(data_path)
        data = self.transform(data)

        label = data[:, :, 512:1024]
        data = data[:, :, :512]

        if data_path[-5] == '1':  # Light dense
            classfy_label = [0, 0, 1]
        elif data_path[-5] == '2':  # Middle dense
            classfy_label = [0, 1, 1]
        elif data_path[-5] == '3':  # Heavy dense
            classfy_label = [1, 1, 1]

        classfy_label = torch.tensor(classfy_label).float()

        return data, label , classfy_label

    def __len__(self):
        return len(self.data_filenames)


class derain_test_datasets(Data.Dataset):
    '''return rain_img . classfy_label'''

    def __init__(self, data_root):
        super(derain_test_datasets, self).__init__()
        self.data_filenames = [join(data_root, x) for x in listdir(data_root) if is_image_file(x) and '._' not in x]
        self.transform = TF.Compose(
                [
                    # TF.RandomVerticalFlip(),
                    # TF.RandomHorizontalFlip(),
                    # TF.RandomCrop(224),
                    TF.ToTensor(),  # tensor range for 0 to 1
                    # TF.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5], )
                ]
            )


    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        data = Image.open(data_path)
        data = self.transform(data)

        label = data[:, :, 512:1024]
        data = data[:, :, :512]

        image_number = re.findall(r'\d+', data_path)[0]

        return data, label , data_path

    def __len__(self):
        return len(self.data_filenames)