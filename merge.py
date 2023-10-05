import wave
import random
import torch
import os
from scipy.io.wavfile import read, write
import numpy as np

audio_folders = ["samples/1", "samples/2", "samples/3", "samples/4", "samples/5"]

num_files = 10000

labels = torch.zeros(num_files, 5, dtype=torch.int)

for i in range(num_files):
    num_voices = random.randint(1, 5)
    selected_folders = random.sample(audio_folders, num_voices)
   
    selected_audio_files = [random.choice([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".wav")]) for folder in selected_folders]

    rates = []
    data = []
    for audio_file in selected_audio_files:
        rate, audio = read(audio_file)
        rates.append(rate)
        data.append(audio)

    merged_data = np.zeros_like(data[0])
    for audio in data:
        merged_data += audio

    output_file = f"samples/Merge/merge-{i+1}.wav"
    write(output_file, rates[0], merged_data)
    for folder in selected_folders:
       labels[i, audio_folders.index(folder)] = 1

print(labels)
torch.save(labels, "labels.pt")