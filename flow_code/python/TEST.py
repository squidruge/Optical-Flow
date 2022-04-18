from scipy.optimize import curve_fit
import numpy as np


def curve(x, var1):
    y = x[0] * x[1] * var1
    return y


x = np.array([np.arange(10), np.arange(100, 110)])
y = 0.1 * x[0] * x[1]
popt, _ = curve_fit(curve, x, y)
print(popt)

