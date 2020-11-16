import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


# 画坐标格
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# 画分界线
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# 选择萼片长度和萼片宽度为样本属性
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# 训练模型
model = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(X, y)
title = 'Non-linear Classifier of Iris'

fig, ax = plt.subplots()
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

# 画图
plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_title(title)

plt.legend(["blue: Iris-setosa",
            "white:Iris-versicolor",
            "red:   Iris-virginica"])
plt.show()
