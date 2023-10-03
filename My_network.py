import torch.nn as nn
import torch
from torch.nn import functional as F 

#Down  
class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.conv_relu1 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 3, 1, 1),
            nn.ReLU()
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
            nn.ReLU()
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 3, 1, 1),
            nn.ReLU()
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
        self.down1 = Down(in_channel, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)

        self.conv_relu1 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_relu2 = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

        #Up
        self.up4 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.up1 = Up(16, 8)

        self.conv = nn.Sequential(
            nn.Conv1d(8, out_channel, 3, 1, 1),
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