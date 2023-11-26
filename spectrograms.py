from scipy.io import wavfile
import scipy.signal as signal
import numpy as np
import json

def get_spectrogram(audio):
    spectrogram = []
    window_size = 1024
    overlap = int(3/4 * window_size)
    hamming = np.hamming(window_size)

    for i in range(0, len(audio) - window_size, overlap):
        audio_window = np.matmul(np.diag(hamming), audio[i : i + window_size])
        spectrum = np.fft.fft(audio_window)
        spectrogram.append(spectrum)

    spectrogram = np.abs(spectrogram) + 1e-6
    spectrogram = np.log(spectrogram)
    spectrogram = np.array(spectrogram).T
    rows, cols = spectrogram.shape
    spectrogram = spectrogram[0:int(rows/2), :]

    return spectrogram

def scipy_spectrogram(audio, rate):
    frequencies, times, spectrogram = signal.spectrogram(audio, rate)
    return spectrogram

def create_spectrograms(file = "labels.json", num_files = 50):
    with open(file, "r") as f:
        labels = json.load(f)
        for label in labels:
            files = labels[label]
            label_train_data = np.array([])
            label_test_data = np.array([])
            if len(files) > num_files:
              files = np.random.choice(files, num_files, replace = False)
            train = files[:int(0.8 * len(files))]
            test = files[int(0.8 * len(files)):]
            for file in train:
                path = "./FSD50K.dev_audio/" + file + ".wav"
                try:
                  rate, audio = wavfile.read(path)
                  spectrogram = get_spectrogram(audio)
                  if label_train_data.shape[0] == 0:
                    label_train_data = spectrogram
                  else:
                    label_train_data = np.concatenate((label_train_data, spectrogram), axis = 1)
                except:
                  print("Error reading file: ", path)
                  continue
            for file in test:
                path = "./FSD50K.dev_audio/" + file + ".wav"
                try:
                  rate, audio = wavfile.read(path)
                  spectrogram = get_spectrogram(audio)
                  if label_test_data.shape[0] == 0:
                    label_test_data = spectrogram
                  else:
                    label_test_data = np.concatenate((label_test_data, spectrogram), axis = 1)
                except:
                  print("Error reading file: ", path)
                  continue
            print(label, label_train_data.shape)
            np.save("./spectrogram_data/train/" + label + ".npy", label_train_data)
            np.save("./spectrogram_data/test/" + label + ".npy", label_test_data)
            with open("spectrogram_per_class.txt", "a") as f:
                f.write(label + "," + str(len(train)) + "," + str(len(test)) + "\n")

create_spectrograms()