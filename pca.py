import os
import numpy as np
import scipy.sparse.linalg as la
from sklearn.decomposition import PCA

def create_training_data(files = "./spectrogram_data/train"):
    train_data = np.array([])
    labels_shape = []
    for file in os.listdir(files):
        label_data = np.load(files + '/' + file)
        cols = label_data.shape[1]
        train_data = np.concatenate((train_data, label_data), axis = 1) if train_data.shape[0] != 0 else label_data
        labels_shape.append(cols)
        print(label_data.shape, train_data.shape)
    np.save("train_data.npy", train_data)
    print("labels shape: ", len(labels_shape))
    return labels_shape

def create_subset_training_data(file = "important_classes.txt"):
    train_data = np.array([])
    labels_shape = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip()
            print(label)
            label_data = np.load("./spectrogram_data/train/" + label + ".npy")
            cols = label_data.shape[1]
            train_data = np.concatenate((train_data, label_data), axis = 1) if train_data.shape[0] != 0 else label_data
            labels_shape.append(cols)
            print(label_data.shape)
            print(train_data.shape)
    np.save("subset_train_data.npy", train_data)
    print("labels shape:", len(labels_shape))
    return labels_shape
    
def pca_manual(dims = 20):
    train = np.load("train_data.npy")

    CovX = np.cov(train) + 1e-6
    eigenvalues, eigenvectors = la.eigs(CovX, dims)
    eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
    sqrt_eigenvalues = np.real(np.sqrt(eigenvalues))
    sqrt_inverse_eigenvalues = np.diag(1 / sqrt_eigenvalues)

    W = sqrt_inverse_eigenvalues @ eigenvectors.T

    Z = W @ train

    return W, Z

def pca_sklearn(dims = 20):
    train = np.load("subset_train_data.npy")
    pca = PCA(n_components = dims)
    pca.fit(train)
    Z = pca.transform(train)
    return pca, Z

# features, transformed_data = pca_manual()
# np.save("transformed_data.npy", transformed_data)

# subset_labels_shape = create_subset_training_data()
# print(subset_labels_shape)

pca, transformed_data = pca_sklearn()
np.save("transformed_data_sklearn.npy", transformed_data)
np.save("pca_sklearn.npy", pca.components_)
