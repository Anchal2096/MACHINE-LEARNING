import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target


def plotSVC(title):
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min()-1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min()-1, X[:, 1].max() + 1
  h = (x_max / x_min)/100
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  plt.subplot(1, 1, 1)
  Z = svm.SVC.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
  plt.xlabel("Sepal length")
  plt.ylabel("Sepal width")
  plt.xlim(xx.min(), xx.max())
  plt.title(title)
  plt.show()


kernels = ["linear", "rbf", "poly"]
for kernel in kernels:
  svc = svm.SVC(kernel=kernel).fit(X, y)
  plotSVC("kernel=" + str(kernel))


gammas = [0.1, 1, 10, 100]
for gamma in gammas:
   svc = svm.SVC(kernel="rbf", gamma=gamma).fit(X, y)
   plotSVC("gamma=" + str(gamma))


cs = [0.1, 1, 10, 100, 1000]
for c in cs:
   svc = svm.SVC(kernel="rbf", C=c).fit(X, y)
   plotSVC("C=" + str(c))


degrees = [0, 1, 2, 3, 4, 5, 6]
for degree in degrees:
   svc = svm.SVC(kernel="poly", degree=degree).fit(X, y)
   plotSVC("degree=" + str(degree))
