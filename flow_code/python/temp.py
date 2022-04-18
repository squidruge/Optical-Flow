import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import math

# 创建一个函数模型用来生成数据
def func1(x, a, b, c, d):
    r = a * np.exp(-((x[0] - b) ** 2 + (x[1] - d) ** 2) / (2 * c ** 2))
    return r.ravel()


# 生成原始数据
x1 = np.linspace(0, 10, 10).reshape(1, -1)
x2 = np.linspace(0, 10, 10).reshape(1, -1)
x = np.append(x1, x2, axis=0)
X, Y = np.meshgrid(x1, x2)
XX = np.expand_dims(X, 0)
YY = np.expand_dims(Y, 0)
xx = np.append(XX, YY, axis=0)
y = func1(xx, 10, 5, 2, 5)
# 对原始数据增加噪声
yn = y + 0.002 * np.random.normal(size=xx.shape[1] * xx.shape[2])

# 使用curve_fit函数拟合噪声数据
t0 = timeit.default_timer()
popt, pcov = curve_fit(func1, xx, yn)
elapsed = timeit.default_timer() - t0
print('Time: {} s'.format(elapsed))

# popt返回最拟合给定的函数模型func的参数值
print(popt)

fig = plt.figure('拟合图')
ax = Axes3D(fig)
X, Y = np.meshgrid(x1, x2)
XX = np.expand_dims(X, 0)
YY = np.expand_dims(Y, 0)
xx = np.append(XX, YY, axis=0)
R = func1(xx, *popt)
# R, _ = np.meshgrid(R, x1)
# y = func1(xx, 10, 5, 2, 5)
# # 对原始数据增加噪声
# yn = y + 0.002 * np.random.normal(size=xx.shape[1] * xx.shape[2])
R = R.reshape(10, 10)
yn = yn.reshape(10, 10)
ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap='rainbow')
ax.plot_surface(X, Y, yn, rstride=1, cstride=1, color='red')

plt.show()
y_predict_1 = func1(x, *popt)
indexes_1 = getIndexes(y_predict_1, yn)
print(indexes_1)