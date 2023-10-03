from torch.utils.data import Dataset
import torchaudio
import torch
import os

class My_Dataset(Dataset):
    def __len__(self):
        return 35
    
    def __getitem__(self, index):
        audio_merge, sample_rate = torchaudio.load(os.path.join('Marge', 'Marge-' + str(index + 1) + '.wav'))
        audio1, _ = torchaudio.load(os.path.join('Angelina', 'Angelina-' + str(index + 1) + '.wav'))
        audio2, _ = torchaudio.load(os.path.join('Typhon', 'Typhon-' + str(index + 1) + '.wav'))
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_merge = transform(audio_merge)[:, : 4*16000]
        audio1 = transform(audio1)[:, : 4*16000]
        audio2 = transform(audio2)[:, : 4*16000]
        return audio_merge, torch.cat((audio1, audio2), dim=0)
    
class My_Dataset2(Dataset):
    def __len__(self):
        return 35
    
    def __getitem__(self, index):
        audio_merge, sample_rate = torchaudio.load(os.path.join('Merge', 'marge-' + str(index + 1) + '.wav'))
        audio1, _ = torchaudio.load(os.path.join('Swire', 'Swire-' + str(index + 1) + '.wav'))
        audio2, _ = torchaudio.load(os.path.join('Ejafjalla', 'Ejafjalla-' + str(index + 1) + '.wav'))
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_merge = transform(audio_merge)[:, : 2*16000]
        audio1 = transform(audio1)[:, : 2*16000]
        audio2 = transform(audio2)[:, : 2*16000]
        return audio_merge, torch.cat((audio1, audio2), dim=0)
    