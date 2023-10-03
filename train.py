from My_dataset import My_Dataset
from torch.utils.data import DataLoader
from My_network import Unet
import torch.nn as nn
import torch
import torchaudio
import os

Epoch = 10
Batch_size = 25
LR = 0.001

train_loader = DataLoader(dataset=My_Dataset(), batch_size=Batch_size, shuffle=True)

def main():
    unet = Unet(in_channel=1, out_channel=2)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
    unet = unet.cuda()
    loss_function = loss_function.cuda()

    for epoch in range(Epoch):
        for data in train_loader:
            audio_merge, audio_spilt = data
            audio_merge = audio_merge.cuda()
            audio_spilt = audio_spilt.cuda()
            optimizer.zero_grad()
            output = unet(audio_merge)
            loss = loss_function(output, audio_spilt)
            loss.backward()
            optimizer.step()
            print('Train Loss: ', loss)

    torch.save(unet.state_dict(), "model.pt")
    
if __name__ == '__main__':
    main()