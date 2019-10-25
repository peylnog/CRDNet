from networks import *
from Utils.utils import *
from torch.nn.functional import interpolate as Inter
class Rain_steaks_with_BG(nn.Module):
    '''get rain steaks with background from RGBrain_img
    return img-----512 512 3*3    '''
    def __init__(self):
        super(Rain_steaks_with_BG, self).__init__()
        self.conv_3_0 = conv_blocks_size_3(in_dim=3 , out_dim=4 , Use_pool=True) #256
        weights_init_xavier(self.conv_3_0)
        self.conv_3_1 = conv_blocks_size_3(4,8,Use_pool=True) #128
        weights_init_xavier(self.conv_3_1)
        self.conv_3_2 = conv_blocks_size_3(8,16,True) #64
        weights_init_xavier(self.conv_3_2)



        self.deconv_3_0 = deconv_blocks_size_3(16,8,True) #128
        weights_init_xavier(self.deconv_3_0)
        self.deconv_3_1 = deconv_blocks_size_3(8, 4, True) #256
        weights_init_xavier(self.deconv_3_1)
        self.deconv_3_2 = deconv_blocks_size_3(4,3,True) #512
        weights_init_xavier(self.deconv_3_2)




        self.conv_5_0 = conv_blocks_size_5(in_dim=3, out_dim=4, Use_pool=True)  # 256
        weights_init_xavier(self.conv_5_0)
        self.conv_5_1 = conv_blocks_size_5(4, 8, Use_pool=True)  # 128
        weights_init_xavier(self.conv_5_1)
        self.conv_5_2 = conv_blocks_size_5(8,16,True) #64
        weights_init_xavier(self.conv_5_2)



        self.conv_7_0 = conv_blocks_size_7(in_dim=3, out_dim=4, Use_pool=True)  # 256
        weights_init_xavier(self.conv_7_0)
        self.conv_7_1 = conv_blocks_size_7(4,8, Use_pool=True)  # 128
        weights_init_xavier(self.conv_7_1)
        self.conv_7_2 = conv_blocks_size_7(8,16, True)  # 64
        weights_init_xavier(self.conv_7_2)




    def forward(self,x):
        input_3 = x
        input_3 = self.conv_3_0(input_3)
        input_3 = self.conv_3_1(input_3)
        input_3 = self.conv_3_2(input_3)
        input_3 = self.deconv_3_0(input_3)
        input_3 = self.deconv_3_1(input_3)
        input_3 = self.deconv_3_2(input_3)

        input_5 = x
        input_5 = self.conv_5_0(input_5)
        input_5 = self.conv_5_1(input_5)
        input_5 = self.conv_5_2(input_5)
        input_5 = self.deconv_3_0(input_5)
        input_5 = self.deconv_3_1(input_5)
        input_5 = self.deconv_3_2(input_5)


        input_7 = x
        input_7 = self.conv_7_0(input_7)
        input_7 = self.conv_7_1(input_7)
        input_7 = self.conv_7_2(input_7)
        input_7 = self.deconv_3_0(input_7)
        input_7 = self.deconv_3_1(input_7)
        input_7 = self.deconv_3_2(input_7)




        return torch.cat([input_3,input_5,input_7] ,dim=1)



class Low_BackGround(nn.Module):
    '''get low background from RGB rain_Img
    in order to move rain steaks ,wo decide to insert some residual img
    to make CNN do his jod better  input 512 512 3 retrun 32 32 64'''
    def __init__(self):
        super(Low_BackGround, self).__init__()
        self.encode1 = conv_blocks_size_3(in_dim=4 , out_dim=8 ,Use_pool=True)  #256
        weights_init_xavier(self.encode1)
        self.encode2 = conv_blocks_size_3(in_dim=8 ,out_dim=16,Use_pool=True)   #128
        weights_init_xavier(self.encode2)
        self.encode3 = conv_blocks_size_3(in_dim=16 ,out_dim=32,Use_pool=True)   #64
        weights_init_xavier(self.encode3)
        self.encode4 = conv_blocks_size_3(in_dim=32,out_dim=64,Use_pool=True)   #32
        weights_init_xavier(self.encode4)
        self.encode5 = conv_blocks_size_3(in_dim=64,out_dim=128,Use_pool=True)  #16
        weights_init_xavier(self.encode5)
        self.encode6 = conv_blocks_size_5(in_dim=128,out_dim=256,Use_pool=True)#8
        weights_init_xavier(self.encode5)


        self.decode1 = deconv_blocks_size_3(in_dim=256,out_dim=128,Use_pool=True) #16
        weights_init_xavier(self.decode1)
        self.decode2 =deconv_blocks_size_3(128,64,Use_pool=True) #32
        weights_init_xavier(self.decode2)
        self.decode3 = deconv_blocks_size_3(64,32,Use_pool=True) #64
        weights_init_xavier(self.decode3)
        self.decode4 = deconv_blocks_size_3(32,16,Use_pool=True) #128
        weights_init_xavier(self.decode4)
        self.decode5 = deconv_blocks_size_3(16,8,True) #256
        weights_init_xavier(self.decode5)
        self.decode6 = deconv_blocks_size_3(8,4,True) #512
        weights_init_xavier(self.decode6)

        self.eencode1 = conv_blocks_size_3(4,8,True) #256
        weights_init_xavier(self.eencode1)
        self.eencode2 = conv_blocks_size_3(8,16,True)#128
        weights_init_xavier(self.eencode2)
        self.eencode3 = conv_blocks_size_3(16,32,True)#64
        weights_init_xavier(self.eencode3)
        self.eencode4 = conv_blocks_size_3(32, 64, True)  # 32
        weights_init_xavier(self.eencode4)
        self.eencode5 = conv_blocks_size_3(64, 128, True)  # 16
        weights_init_xavier(self.eencode5)
        self.eencode6 = conv_blocks_size_3(128, 256, True)  # 8
        weights_init_xavier(self.eencode6)

        self.ddcode1 = deconv_blocks_size_3(256,128)
        self.ddcode2 = deconv_blocks_size_3(128,64)
        self.ddcode3 = deconv_blocks_size_3(64, 32)
        self.ddcode4 = deconv_blocks_size_3(32, 16)
        self.ddcode5 = deconv_blocks_size_3(16, 8)
        self.ddcode6 = deconv_blocks_size_3(8, 4)

    def forward(self,x):
        # x  512 512 3
        """insert img to use residual_img as a auxiliary"""
        insert_img = 0.8 * get_residue(x) + 0.2 * rgb2grad(x)  #insert image
        x = torch.cat([x , insert_img], 1) #512
        encode1 = self.encode1(x)      #256
        encode2 = self.encode2(encode1) #128
        encode3 = self.encode3(encode2) #64
        encode4 = self.encode4(encode3) #32
        encode5 = self.encode5(encode4) #16
        encode6 = self.encode6(encode5) #8

        #super_U-net

        decode = torch.add(encode5 , self.decode1(encode6))  #16
        decode = torch.add(encode4 , self.decode2(decode))   #32
        decode = torch.add(encode3 , self.decode3(decode))  #64
        decode = torch.add(encode2 ,self.decode4(decode))  #128
        decode = torch.add(encode1 , self.decode5(decode)) #256
        decode = torch.add(x , self.decode6(decode)) #512

        del encode1
        del encode2
        del encode3
        del encode4
        del encode5
        del encode6

        eencode1 = self.eencode1(decode)
        eencode2 = self.eencode2(eencode1)
        eencode3 = self.eencode3(eencode2)
        eencode4 = self.eencode4(eencode3)
        eencode5 = self.eencode5(eencode4)
        eencode6 = self.eencode6(eencode5)

        ddecode = torch.add(self.ddcode1(eencode6) , eencode5)
        del eencode6,eencode5
        ddecode = torch.add(self.ddcode2(ddecode) , eencode4)
        del eencode4
        ddecode = torch.add(self.ddcode3(ddecode), eencode3)
        del eencode3
        ddecode = torch.add(self.ddcode4(ddecode), eencode2)
        del eencode2
        ddecode = torch.add(self.ddcode5(ddecode), eencode1)
        del eencode1
        ddecode = torch.add(self.ddcode6(ddecode), decode)

        return  ddecode


class Classfication(nn.Module):
    '''classfication to classfy rain dense ,make sure Rain_with_Background success extract
    rain steaks (with some high level background will be fine )
    input 512 512 9'''
    def __init__(self):
        super(Classfication, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(9,16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),        #256


            nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),    #128

            nn.Conv2d(32, 2*32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(2*32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64

            nn.Conv2d(32*2, 4 * 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d( 4 * 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32

            nn.Conv2d(32 * 4, 8 * 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(8 * 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16

            nn.Conv2d(32 * 8, 16 * 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16 * 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8

            nn.Conv2d(32 * 16, 32 * 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32 * 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4

        )

        self.classfier = nn.Sequential(

            nn.Linear( 4*4*32*32, 4096 ),
            nn.ReLU(True),
            nn.Linear(4096 , 1000 ),
            nn.ReLU(True),
            nn.Linear(1000 ,3),
            nn.ReLU(True),
        )
        weights_init_xavier(self.features)
        weights_init_xavier(self.classfier)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(x)
        x = self.features(x)
        x = x.view( x.size(0) , -1)
        x = self.classfier(x)
        return  x

class Derain(nn.Module):
    '''derain input 512 512 9   512 512 4    512 512 3 '''
    def __init__(self):
        super(Derain, self).__init__()

        self.conv1 =conv_block_size_3_without_bn(16,8)
        weights_init_xavier(self.conv1)
        self.conv2 = conv_block_size_3_without_bn(8, 4)
        weights_init_xavier(self.conv2)
        self.conv3 = conv_block_size_3_without_bn(4, 3)
        weights_init_xavier(self.conv3)

        self.relu = Nonlinear_layer(relu= True)
    def forward(self,High, Low , rain_img):
        '''High         512 512 9
        low             512 512 4
        rain            512 512 3              '''


        rain_img = torch.cat([Low , High , rain_img],dim=1)  #512 512 16
        rain_img = self.conv1(rain_img)
        rain_img = self.conv2(rain_img)
        rain_img = self.conv3(rain_img)

        return rain_img