import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

(X, Y) = make_blobs(n_samples=5, n_features=2, centers=2, random_state=50)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
plt.axis([-5, 10, -12, -1])
plt.show()

postiveX = []
negativeX = []
for i, v in enumerate(y):
    if v == 0:
        negativeX.append(X[i])
    else:
        postiveX.append(X[i])

# our data dictionary
data_dict = {-1: np.array(negativeX), 1: np.array(postiveX)}
