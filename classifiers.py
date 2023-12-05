import numpy as np
import os
import pickle as pk
import json
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
# some helper functions

def multivar_gaussian_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Given the mean and covariance matrix of a multivariate normal distribution, compute the probability density of x.

    Args:
        x: the input point to compute the probability density of
        mu: the mean of the multivariate normal distribution
        sigma: the covariance matrix of the multivariate normal distribution
    
    Returns:
        The probability density of x

    Throws:
        ValueError: if the dimensions of mu and sigma do not match
        ValueError: if sigma is not positive definite
    """
    if mu.shape[0] != sigma.shape[0] or mu.shape[0] != sigma.shape[1]:
        raise ValueError("Dimensions of mu and sigma do not match")
    if mu.shape[0] != x.shape[0]:
        raise ValueError("Dimensions of mu and x do not match")
    if not np.all(np.linalg.eigvals(sigma) > 0):
        print("Sigma is not positive definite... Using diagonal")
        sigma = np.diag(np.diag(sigma))

    d = x.shape[0]
    # handle case when multiple points given as columns in x
    if len(x.shape) > 1:
        mu = mu.reshape(-1, 1)
        return 1 / (np.sqrt((2*np.pi)**d * np.linalg.det(sigma)) + 1e-5) * np.exp(-0.5 * np.diag((x - mu).T @ np.linalg.inv(sigma) @ (x - mu)))
    
    return 1 / (np.sqrt((2*np.pi)**d * np.linalg.det(sigma)) + 1e-5) * np.exp(-0.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu))


def fit_gaussian(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a set of data points D (d x n), fit a multivariate normal distribution to the data.

    Args:
        D: the data matrix. Each column is a data point.

    Returns:
        Mean and covariance matrix of the fitted distribution in a tuple.
    """
    mu = np.mean(D, axis=1)
    sigma = np.cov(D).reshape(mu.shape[0], mu.shape[0])
    return mu, sigma


# the classifiers...

class GaussianClassifier():
    """
    Gaussian classifier for n classes. Fits a multivariate normal distribution to each class in training data
    and classifies new data points based on the posterior probability density of the point under each distribution
    using the Baye's decision rule.
    """

    def __init__(self, nclass) -> None:
        self.nclass = nclass     # list of possible labels to classify as
        self.class_models = []   # gaussian model fit to each class
    
    def fit(self, train_data: np.ndarray, train_labels: np.ndarray) -> None:
        """
        Fit a gaussian model to each class in the training data.

        Args:
            train_data: d x n matrix of training data. Each column is a data point.
            train_labels: 1 x n vector of labels for each data point in train_data.
                          Assumes all elements of train_labels are one of 0..(nclass-1)
        """
        self.class_models = [fit_gaussian(train_data[:, train_labels == i]) for i in range(self.nclass)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Return p(x | class) for each class in the model
        """
        return np.array([multivar_gaussian_pdf(x, mu, sigma) for mu, sigma in self.class_models])

    def predict(self, x: np.ndarray) -> int:
        return np.argmax(self.predict_proba(x), axis=0)
    
    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Evaluate the accuracy of the classifier on the given dataset.
        Compare what the model classifies each data point as to what the actual label is.
        """
        acc = np.mean(self.predict(data) == labels)
        return acc


    
class PCAClassifier():
    """
    Classifier that computes the low dimensional PCA subspace for each class in training data and
    classifies data points based on which subspace they are most aligned with
    """

    def __init__(self, nclass: int) -> None:
        self.nclass = nclass        # number of classes 0..(nclass-1)
        self.class_proj_mats = []   # store PCA projection matrix for each class
        self.mean = 0               # mean vector of training data

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray, m: int = None) -> None:
        """
        Compute PCA projection matrix for each class in the training data.

        Args:
            train_data: d x n matrix of training data. Each column is a data point.
            train_labels: 1 x n vector of labels for each data point in train_data.
                          Assumes all elements of train_labels are one of 0..(nclass-1)
            m: number of principal components to use in projection. m <= d
        """
        if train_labels.max() >= self.nclass:
            raise ValueError(f"train_labels contains labels greater than or equal to nclass ({self.nclass})")
        if  train_labels.min() < 0:
            raise ValueError("train_labels contains negative labels")
        
        if (m is None):
            m = train_data.shape[0]

        self.mean = train_data.mean(axis=1, keepdims=True)
        centered_train_data = train_data - self.mean
        for i in range(self.nclass):
            U, S, Vt = np.linalg.svd(centered_train_data[:, train_labels == i].T, full_matrices=False)
            self.class_proj_mats.append(Vt[:m])

    def predict(self, x: np.ndarray) -> int:
        """
        Classify the given data points based on which subspace it is most aligned with (i.e. has highest magnitude projection onto).

        Args:
            x: d x n data point to classify

        Returns:
            array of predicted class labels of the data point.
        """
        x = x - self.mean
        return np.argmax(np.array([np.linalg.norm(Vt @ x, axis=0) for Vt in self.class_proj_mats]), axis=0)

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Evaluate the accuracy of the classifier on the given dataset.
        Compare what the model classifies each data point as to what the actual label is.
        """
        acc = np.mean(self.predict(data) == labels)
        return acc

# def gaussian_quadratic_boundary(label_training_data):
#   mu = np.mean(label_training_data, axis = 1)
#   sigma = np.cov(label_training_data)

#   inv_sigma = np.linalg.inv(sigma)

#   W = -1/2 * inv_sigma
#   w = inv_sigma @ mu
#   b = -1/2 * (mu.T @ inv_sigma @ mu) - 1/2 * np.log(np.linalg.det(sigma))

#   return W, w, b

noise = set()
with open("output/noise.txt", "r") as f:
    for line in f:
        noise.add(line.strip())

music = set()
with open("output/music.txt", "r") as f:
    for line in f:
        music.add(line.strip())

classes = []
noises = 0
noise_dir = "./scipy_spectrogram_data/noise/train/"
for file in os.listdir(noise_dir):
    classes.append(file)
    noises += 1
music_dir = "./scipy_spectrogram_data/music/train/"
music = 0
for file in os.listdir(music_dir):
    classes.append(file)
    music += 1
# print(classes, len(classes))

mapping = {}
for i in range(len(classes)):
    mapping[classes[i]] = i
# print(mapping)

train = np.load("transformed_data_sklearn.npy")
train = train.T
# print(train.shape)

label_shapes = np.load("subset_labels_shape.npy")
mid = 0
for i in range(noises):
    mid += label_shapes[i]

train_noise = train[:, :mid]
train_music = train[:, mid:]

labels = []
for i in range(mid):
    labels.append(0)
for i in range(mid, train.shape[1]):
    labels.append(1)

labels = np.array(labels)

# for i in range(len(label_shapes)):
#     start = sum(label_shapes[:i])
#     end = sum(label_shapes[:i+1])
#     label = mapping[classes[i]]
#     # print(start, end, label)
#     for j in range(start, end):
#         labels.append(label)

train = np.concatenate((train_noise, train_music), axis = 1)
labels = np.array(labels)
classifier = KNeighborsClassifier(2)
train = train.T
classifier.fit(train, labels)
pk.dump(classifier, open("knn_classifier.pkl", "wb"))

# bootstrap_samples = 10000

# for i in range(10):
#     sample = np.random.choice(train_noise.shape[1], size = bootstrap_samples, replace = True)
#     pred = classifier.predict(train_noise[:, sample].T)
#     acc = np.mean(pred == 0)
#     print("noise accuracy", acc)
#     sample = np.random.choice(train_music.shape[1], size = bootstrap_samples, replace = True)
#     pred = classifier.predict(train_music[:, sample].T)
#     acc = np.mean(pred == 1)
#     print("music accuracy", acc)

# pca = pk.load(open("pca.pkl", "rb"))

# classifiers = {}

# classifier = PCAClassifier(len(classes))
# classifier.fit(train, np.array(labels))

# bootstrap_samples = 10000

# for i in range(len(label_shapes)):
#     start = sum(label_shapes[:i])
#     end = sum(label_shapes[:i+1])
#     label = mapping[classes[i]]
#     subset = train[:, start:end].T
#     print(start, end, subset.shape, label)
#     accuracy = []
#     sample = np.random.choice(subset.shape[1], size = bootstrap_samples, replace = False)
#     pred = classifier.predict(subset[:, sample].T)
#     acc = np.mean(pred == label)
#     print(classes[i], np.mean(acc))

# for i in range(100):
#     x = train[:, i]
#     label = labels[i]
#     pred = classifier.predict(x)
#     print(label, pred)

# directory = "./scipy_spectrogram_data/noise/test/"
# for file in os.listdir(directory):
#     correct = str(file)
#     test = np.load(directory + file)
#     test = np.abs(test).T
#     test = pca.transform(test)
#     test = test.T

#     labels = [mapping[correct]] * test.shape[1]
#     labels = np.array(labels)

#     accuracy = []

#     for _ in tqdm(range(100)):
#         sample = np.random.choice(test.shape[1], size = bootstrap_samples, replace = True)
#         a = classifier.evaluate(test[:, sample], labels[sample])
#         accuracy.append(a)

#     print(file, np.mean(accuracy))

# # print("Gaussian Classifier Accuracy: ", classifier.evaluate(train, np.array(labels)))
