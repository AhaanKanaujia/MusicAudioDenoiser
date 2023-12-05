import os
import numpy as np
from sklearn.decomposition import PCA
import pickle as pk

def create_subset_training_data():
    # train_data = np.array([])
    # labels_shape = []
    noise_dir = "./scipy_spectrogram_data/noise/train/"
    for file in os.listdir(noise_dir):
        print(file)
        # label_data = np.load(noise_dir + file)
        # cols = label_data.shape[1]
        # train_data = np.concatenate((train_data, label_data), axis = 1) if train_data.shape[0] != 0 else label_data
        # labels_shape.append(cols)
        # print(train_data.shape)
    music_dir = "./scipy_spectrogram_data/music/train/"
    for file in os.listdir(music_dir):
        print(file)
    #     label_data = np.load(music_dir + file)
    #     cols = label_data.shape[1]
    #     train_data = np.concatenate((train_data, label_data), axis = 1) if train_data.shape[0] != 0 else label_data
    #     labels_shape.append(cols)
    #     print(train_data.shape)
    # np.save("subset_train_data.npy", train_data)
    # np.save("subset_labels_shape.npy", labels_shape)
    # return labels_shape

def pca_sklearn(dims = 20):
    train = np.load("subset_train_data.npy")
    train = np.abs(train).T
    pca = PCA(n_components = dims)
    pca.fit(train)
    Z = pca.transform(train)
    return pca, Z

# create_subset_training_data()

pca, transformed_data = pca_sklearn()
np.save("transformed_data_sklearn20.npy", transformed_data)
pk.dump(pca, open("pca20.pkl", "wb"))
