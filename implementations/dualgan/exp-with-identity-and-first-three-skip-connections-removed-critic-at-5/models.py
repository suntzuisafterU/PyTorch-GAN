##############################
#           rando STIO util...
##############################
from io import StringIO
import sys
from torchsummary import summary

class Capturing(list):
    """
    Capture strings printed within a context block....

    source: https://stackoverflow.com/a/16571630
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        print('args...:', args)
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout





import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init_glorot(m):
    """ alternative based on Bengio and ? Glorot initialization method """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, skip_connection=True):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.skip_connection = skip_connection
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # Skip connections, how does a skip_input like this work???
        # What does torch.cat do in this case???
        if self.skip_connection:
            x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        # self.up4 = UNetUp(1024, 256)
        # self.up5 = UNetUp(512, 128)
        # self.up6 = UNetUp(256, 64)
        self.up4 = UNetUp(1024, 512, skip_connection=False)
        self.up5 = UNetUp(512, 256, skip_connection=False)
        self.up6 = UNetUp(256, 128, skip_connection=False)

        # TODO: Here we ConvTranspose2d to 128.... maybe this should be same as image size??
        #       NO! 128 is the input size of the number of channels!
        # TODO: Figure out what the output size of this function is... Is it controlled by the arguments?
        # The output size is invaraint, it matches the input size
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

        ## Doesn't work because we would have to pass the 'skip_input' but we can't....
        # temp_model = nn.Sequential(
        #     self.down1,
        #     self.down2,
        #     self.down3,
        #     self.down4,
        #     self.down5,
        #     self.down6,
        #     self.down7,

        #     self.up1,
        #     self.up2,
        #     self.up3,
        #     self.up4,
        #     self.up5,
        #     self.up6,
        #     self.final
        # )

        # input_shape = (3, 256, 256)
        # ## Unfortunately, this crashes the process because of GPU jank...
        # with Capturing() as summary_str:
        #     summary(temp_model.cuda(), input_shape)

        # with open('Generator_summary.txt', 'w') as f:
        #     f.writelines(summary_str)
        # print('\n'.join(summary_str))


    def forward(self, x):
        # Propogate noise through fc layer and reshape to img shape
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discrimintor_block(in_features, out_features, normalize=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discrimintor_block(in_channels, 64, normalize=False),
            *discrimintor_block(64, 128),
            *discrimintor_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, kernel_size=4)
        )

    def forward(self, img):
        # TODO: what is the output size? 
        # DONE: Output size is: (-1, 1, 30, 30)
        #       So this is a patchGAN?? Seems like it. Yes
        return self.model(img)

