import torch.nn  as  nn
import torch.optim as optim
import torch


class DefaultConfig(object):


    #path
    train_data_root = "" 
    test_data_root =  ""
    env = "Dearin 2019" 
    save_root = ""         #save checkpoints root
    load_root = ""         #load checkpoints root 
    save_image_root = ""

    #parameters

    batch_size =  16 # batch size
    num_workers = 4# how many workers for loading data
    train_epoch = 1000

    Adam_lr_refine = 1e-4     # initial learning rate
    Adam_lr_classfy = 3e-6

    SGD_lr = 1e-4
    weight_decay = 0  # Adam
    SGD_mo = 0.95

    test_freq = 1        #test model every 1 epoch
    train_print_freq = 200 #print for each 50 steps

    save_image =False
    test = False

    def parse(self,kwarge):
        for key , values in kwarge.items():
            if not ( hasattr(self , key) ) :
                print("opt has not attr %s"%key)
                break
            else:
                setattr(self , key , values)
        print("             user config       ")
        for k , v in self.__class__.__dict__.items():
            if not k.startswith("__"):
                print(k , getattr(self , k))






