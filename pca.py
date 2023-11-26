import os
import numpy as np
import scipy.sparse.linalg as la

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
    
def pca(dims = 25):
    train = np.load("train_data.npy")

    CovX = np.cov(train) + 1e-6
    eigenvalues, eigenvectors = la.eigs(CovX, dims)
    eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
    sqrt_eigenvalues = np.real(np.sqrt(eigenvalues))
    sqrt_inverse_eigenvalues = np.diag(1 / sqrt_eigenvalues)

    W = sqrt_inverse_eigenvalues @ eigenvectors.T

    Z = W @ train

    return W, Z

labels_shape = create_training_data()

# features, transformed_data = pca()
# np.save("transformed_data.npy", transformed_data)
