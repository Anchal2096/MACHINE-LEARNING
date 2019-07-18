from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

faces = datasets.fetch_olivetti_faces()

# print(faces.data.shape)
print(faces.DESCR)


def print_faces(images, target, top_n):
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))
    plt.show()


"""X here are the dimensions or the components for differentiation between the classes y here the classes in which the 
given data is to be classified """

"""faces.data = total details of faces, faces.target = classes for which the data is to be classified"""
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, train_size=0.8, random_state=85)

"""this function takes 'the number of components needed' as an argument"""
pca = decomposition.PCA(n_components=150, whiten=True)
print(X_train.shape, X_test.shape)
pca.fit(X_train)

# choose the mean face from all the faces
plt.imshow(pca.mean_.reshape(faces.images[95].shape), cmap=plt.cm.bone)
print(pca.components_.shape)

# plotting the figure with given argument of figure size in inches
fig = plt.figure(figsize=(16, 6))

# subplots are the plots in the main image
# arguments are given as (height, width, subplot number, xticks, yticks = markings on the x and y axes)
for i in range(30):
    ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(faces.images[0].shape), cmap=plt.cm.bone)

"""the attribute transform helps to reduce the dimensions of the training set and the test set"""
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)

print(X_test_pca.shape)

print(print_faces(faces.images, faces.target, 20))
print('Accuracy : ', accuracy_score(y_true=y_train, y_pred=y_test))
print(confusion_matrix(y_true=y_train, y_pred=y_test))
