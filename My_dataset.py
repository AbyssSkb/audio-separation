from torch.utils.data import Dataset
import torchaudio
import torch
import os

class trainDataset(Dataset):
    def __len__(self):
        return 1225
    
    def __getitem__(self, index):
        audio_merge, sample_rate = torchaudio.load(os.path.join('C:\\Users\\Abyss\\Music\\Merge', 'Merge-' + str(index + 1) + '.wav'))
        audio1, _ = torchaudio.load(os.path.join('C:\\Users\\Abyss\\Music\\1_Split', '1_Split-' + str(index + 1) + '.wav'))
        audio2, _ = torchaudio.load(os.path.join('C:\\Users\\Abyss\\Music\\2_Split', '2_Split-' + str(index + 1) + '.wav'))
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_merge = transform(audio_merge)[:, : 4*16000]
        audio1 = transform(audio1)[:, : 4*16000]
        audio2 = transform(audio2)[:, : 4*16000]
        return audio_merge, torch.cat((audio1, audio2), dim=0)
    
class testDataset(Dataset):
    def __len__(self):
        return 1225
    
    def __getitem__(self, index):
        audio_merge, sample_rate = torchaudio.load(os.path.join('C:\\Users\\Abyss\\Music\\Merge', 'Merge-' + str(index + 1) + '.wav'))
        audio1, _ = torchaudio.load(os.path.join('C:\\Users\\Abyss\\Music\\1_Split', '1_Split-' + str(index + 1) + '.wav'))
        audio2, _ = torchaudio.load(os.path.join('C:\\Users\\Abyss\\Music\\2_Split', '2_Split-' + str(index + 1) + '.wav'))
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_merge = transform(audio_merge)[:, : 2*16000]
        audio1 = transform(audio1)[:, : 2*16000]
        audio2 = transform(audio2)[:, : 2*16000]
        return audio_merge, torch.cat((audio1, audio2), dim=0)
    