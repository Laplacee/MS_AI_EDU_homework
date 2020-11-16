import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("mlm.csv")

# 训练的样本
x, y, z = [], [], []
for i in range(0, 999):
    x.append([data.values[i][0], data.values[i][1]])
    z.append(data.values[i][2])

# 训练过程
regr = linear_model.LinearRegression()
regr.fit(x, z)

# 打印训练参数
print("拟合结果为：Z=%f*X+%f*Y+%f" % (regr.coef_[0], regr.coef_[1], regr.intercept_))

# 测试样本
x_test, y_test, z_test = [], [], []
for i in range(0, 999, 10):
    x_test.append(data.values[i][0])
    y_test.append(data.values[i][1])
    z_test.append(data.values[i][2])

# 绘图
ax = plt.axes(projection="3d")
ax.scatter3D(x_test, y_test, z_test)

x_drawing = np.linspace(0, 100)
y_drawing = np.linspace(0, 100)
X_drawing, Y_drawing = np.meshgrid(x_drawing, y_drawing)
ax.plot_surface(X=X_drawing,
                Y=Y_drawing,
                Z=X_drawing * regr.coef_[0] + Y_drawing * regr.coef_[1] + regr.intercept_,
                color='r',
                alpha=0.5)

# 调整视角
ax.view_init(elev=30, azim=30)
plt.show()
