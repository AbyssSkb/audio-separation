import torch
from My_network import Unet
from My_dataset import testDataset
from torch.utils.data import DataLoader
import torchaudio
import os

test_loader = DataLoader(dataset=testDataset(), shuffle=True)

def test():
    model = Unet(in_channel=1, out_channel=2)
    model.load_state_dict(torch.load("model.pt"))
    model = model.cuda()

    i = 1
    for data in test_loader:
        if i > 50:
            break
        audio_merge, _ = data
        audio_merge = audio_merge.cuda()
        output = model(audio_merge).cpu().detach()
        audio_merge = audio_merge.cpu().detach()
        torchaudio.save(os.path.join('C:\\Users\\Abyss\\Music\Test', str(i) + '_1.wav'), output[:, 0, :], 16000)
        torchaudio.save(os.path.join('C:\\Users\\Abyss\\Music\Test', str(i) + '_2.wav'), output[:, 1, :], 16000)
        torchaudio.save(os.path.join('C:\\Users\\Abyss\\Music\Test', str(i) + '_0.wav'), audio_merge[0, :, :], 16000)
        i += 1

if __name__ == '__main__':
    test()