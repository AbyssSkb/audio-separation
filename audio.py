import torch
import torchaudio
import matplotlib.pyplot as plt
import os


def plot_waveform(path):
    waveform, sample_rate = torchaudio.load(path)
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)
    waveform = waveform[:, : 5*16000]
    print(waveform.shape)
    torchaudio.save('1.wav', waveform, 16000)
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / 16000

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show()

def getitem(index):
    audio_merge, sample_rage = torchaudio.load(os.path.join('Marge', 'Marge-' + str(index + 1) + '.wav'))
    print(sample_rage)
    return audio_merge

plot_waveform('Marge/Marge-1.wav')