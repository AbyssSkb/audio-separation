from torch.utils.data import Dataset
import torchaudio
import torch
import os

class trainDataset(Dataset):
    def __init__(self):
        self.audio_folder = "C:/Users/Abyss/Music/samples/train"
        self.audio_list = [audio for audio in os.listdir(self.audio_folder)]
        self.labels = torch.load("labels.pt").float()

    def __len__(self):
        return len(self.audio_list)
    
    def __getitem__(self, index):
        audio = self.audio_list[index]
        waveform, sample_rate = torchaudio.load(os.path.join(self.audio_folder, audio))
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform(waveform)
        label_index = int(audio.split('.')[0].split('-')[1]) - 1
        return waveform, self.labels[label_index]
        
    
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
    