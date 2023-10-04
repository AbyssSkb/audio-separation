import torch.nn as nn
import torch
from torch.nn import functional as F 

#Down  
class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.conv_relu1 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU()
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU()
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        c1 = self.conv_relu1(x)
        c2 = self.conv_relu2(c1)
        x = self.pool(c2)
        return x, c2
#Up
class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channel, out_channel, 2, 2)
        self.conv_relu1 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU()
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU()
        )
    
    def forward(self, x, res):
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.conv_relu1(x)
        x = self.conv_relu2(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channel : int, out_channel : int):
        super(Unet, self).__init__()
        #Down
        self.down1 = Down(in_channel, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.conv_relu1 = nn.Sequential(
            nn.Conv1d(512, 1024, 3, 1, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU()
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU()
        )

        #Up
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)

        self.conv = nn.Sequential(
            nn.Conv1d(64, out_channel, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x, res1 = self.down1(x)
        x, res2 = self.down2(x)
        x, res3 = self.down3(x)
        x, res4 = self.down4(x)

        x = self.conv_relu1(x)
        x = self.conv_relu2(x)
        
        x = self.up4(x, res4)
        x = self.up3(x, res3)
        x = self.up2(x, res2)
        x = self.up1(x, res1)

        x = self.conv(x)
        return x