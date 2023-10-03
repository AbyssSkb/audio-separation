from My_dataset import My_Dataset, My_Dataset2
from torch.utils.data import DataLoader
from My_network import Unet
import torch.nn as nn
import torch
import torchaudio
import os

Epoch = 1000
Batch_size = 5
LR = 0.001

train_loader = DataLoader(dataset=My_Dataset(), batch_size=Batch_size, shuffle=True)
valid_loader = DataLoader(dataset=My_Dataset2())
save_loader = DataLoader(dataset=My_Dataset())

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

    i = 1
    for data in save_loader:
        audio_merge, _ = data
        audio_merge = audio_merge.cuda()
        output = unet(audio_merge).cpu().detach()
        audio_merge = audio_merge.cpu().detach()
        torchaudio.save(os.path.join('Predict', 'spilt-' + str(i) + '_1.wav'), output[:, 0, :], 16000)
        torchaudio.save(os.path.join('Predict', 'spilt-' + str(i) + '_2.wav'), output[:, 1, :], 16000)
        torchaudio.save(os.path.join('Predict', 'merge-' + str(i) + '.wav'), audio_merge[0, :, :], 16000)
        i += 1

    torch.save(unet.state_dict(), "model1.pt")
    
if __name__ == '__main__':
    main()