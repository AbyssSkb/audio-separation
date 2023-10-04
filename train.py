from My_dataset import trainDataset
from torch.utils.data import DataLoader
from My_network import Unet
import torch.nn as nn
import torch

Epoch = 100
Batch_size = 5
LR = 0.0003

train_loader = DataLoader(dataset=trainDataset(), batch_size=Batch_size, shuffle=True)

def main():
    model = Unet(in_channel=1, out_channel=2)
    model.load_state_dict(torch.load("model2.pt"))
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model = model.cuda()
    loss_function = loss_function.cuda()

    for epoch in range(Epoch):
        for data in train_loader:
            audio_merge, audio_spilt = data
            audio_merge = audio_merge.cuda()
            audio_spilt = audio_spilt.cuda()
            optimizer.zero_grad()
            output = model(audio_merge)
            loss = loss_function(output, audio_spilt)
            loss.backward()
            optimizer.step()
            print('Train Loss: %.1e' %loss.item())

        torch.save(model.state_dict(), "model2.pt")

if __name__ == '__main__':
    main()