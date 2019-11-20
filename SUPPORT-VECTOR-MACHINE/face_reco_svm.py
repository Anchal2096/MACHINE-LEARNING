from sklearn import svm
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

faces = datasets.fetch_olivetti_faces()


# print(faces.data.shape)
# print(faces.DESCR)
def print_faces(images, target, top_n):
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=0)

pca = decomposition.PCA(n_components=150, whiten=True)
print(X_train.shape, X_test.shape)
pca.fit(X_train)
print()
plt.imshow(pca.mean_.reshape(faces.images[0].shape), cmap=plt.cm.bone)
print(pca.components_.shape)

fig = plt.figure(figsize=(16, 6))

for i in range(30):
    ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(faces.images[0].shape), cmap=plt.cm.bone)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)

print(X_test_pca.shape)

clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)

fig = plt.figure(figsize=(8, 6))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape),
              cmap=plt.cm.bone)
    y_pred = clf.predict(X_test_pca[i, np.newaxis])[0]
    color = ('black' if y_pred == y_test[i] else 'red')
    ax.set_title(faces.target[y_pred],
                 fontsize='small', color=color)


y_pred = clf.predict(X_test_pca)
print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test, y_pred))

print(print_faces(faces.images, faces.target, 20))

print(metrics.accuracy_score(y_test, y_pred))
