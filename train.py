from My_dataset import trainDataset
from torch.utils.data import DataLoader
from My_network import CNN
import torch.nn as nn
import torch

Epoch = 100
Batch_size = 5
LR = 0.001

train_loader = DataLoader(dataset=trainDataset(), batch_size=Batch_size, shuffle=True)

def main():
    model = CNN(in_channel=1, out_features=5)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model = model.cuda()
    criterion = criterion.cuda()

    for _ in range(Epoch):
        for data in train_loader:
            input, label = data
            label = label.unsqueeze(-1)
            input = input.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print('Train Loss: %.1e' %loss.item())

        torch.save(model.state_dict(), "model.pt")

if __name__ == '__main__':
    main()