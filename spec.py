import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd

audio_path = "C:/Users/haris/Desktop/spectrogram/wav/"
audio_clips = os.listdir(audio_path)
print("No. of .wav files in audio files = ",len(audio_clips))

for i in audio_clips:
    print(i)

    x, sr = librosa.load(audio_path+i, sr=44100)

    print(type(x), type(sr))
    print(x.shape, sr)

    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
#spectrogram
    X = librosa.stft(x)
    z = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(z, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title(i)


#mfcc spectrogram
    y, sr = librosa.load(audio_path+i, sr=44100)
    librosa.feature.mfcc(y=y, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
    librosa.feature.mfcc(S=librosa.power_to_db(S))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=i)