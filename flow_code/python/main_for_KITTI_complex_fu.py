import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from math import sin, cos, tan, pi

# P_rect_03: f=721.5377 u0=6.095593 v0=172.8540

# get paths

# phi 比较大: 320~340 830~840 905~934
pic_num = 313#924

u0 = 609.5593
v0 = 172.8540
fx = fy = f = 721.5377

# v0 = 146  # 922
# v0 = 157  # 910
# v0 = 180  # 561
# pic_num = 116 330 331 410 411 532 560 561 699 700 875 876
path = "E:\\Program Files\\dataset\\KITTI\\2011_09_26_drive_0101_sync"
original_img_path = path + f"\\image_2\\0000000{pic_num}.png"
optical_flow_path = path + f"\\flow\\0000000{pic_num}.png"
semantic_path = path + f"\\semantic\\0000000{pic_num}.png"

# depth_path = path + r"\\depth\\00000040.png"

# ==================== 图1 原始RGB =========================================
rgb_img = Image.open(original_img_path)
plt.figure("raw RGB")
plt.imshow(rgb_img)
plt.axis('off')
plt.show()

original_img_path = path + f"\\image_2\\0000000{pic_num + 1}.png"
rgb_img = Image.open(original_img_path)
plt.figure("raw RGB next")
plt.imshow(rgb_img)
plt.axis('off')
plt.show()

# ==================== 获取路面的mask =========================================
semantic_img = cv2.imread(semantic_path)

# road: (0, 0, 255)
mask = cv2.inRange(semantic_img, (0, 0, 255), (0, 0, 255))
mask = mask / 255
mask.astype(int)

# ==================== 图2 光流图 =========================================
optical_flow_img = cv2.imread(optical_flow_path, -1)
# valid = optical_flow_img[:, :, 0]
fu = optical_flow_img[:, :, 2].astype(np.float32)
fv = optical_flow_img[:, :, 1].astype(np.float32)
fu = (fu - 2 ** 15) / 64.0
fv = (fv - 2 ** 15) / 64.0

plt.figure("fu fv")
plt.subplot(2, 1, 1)
plt.imshow(fu)
plt.title("original fu")
plt.axis('off')

plt.subplot(2, 1, 2)

plt.imshow(fv)
plt.title("original fv")
plt.axis('off')

# ==================== 图3 光流fv图带掩码 =========================================
# fv_origin = fv
# fu_origin = fu

# 底部去除
v_max = image_h = fv.shape[0]
u_max = image_w = fv.shape[1]
#
for i in range(v_max):
    for j in range(u_max):
        if fv[i, j] + i > v_max:
            mask[i, j] = 0

fu = fu * mask
fv = fv * mask

# plt.figure("flow with mask")
# plt.subplot(2, 1, 1)
# plt.imshow(fu)
# plt.title("fu")
# plt.axis('off')
# plt.subplot(2, 1, 2)
# plt.imshow(fv)
# plt.title("fv")
# plt.axis('off')
# plt.show()

# ==================== 图4 v-fv曲线图 =========================================

# v_fv_map = np.zeros((v_max, round(fv.max()) + 1))
# for i in range(v_max):
#     for j in range(u_max):
#         if round(fv[i, j]) != 0:
#             v_fv_map[i, round(fv[i, j])] += 1
#
# plt.figure("v-fv-map")
# plt.imshow(v_fv_map)
# plt.show()

# ==================== v-fv公式计算图 =========================================
# fv = Zd(v1-v0)/(hf/(v1-v0)-Zd)
# Y = Zd*X^2 / (hf - Zd*X)
# f = fx = fy = image_w / 2.0


v1 = np.arange(0, v_max, 0.1)
v1 = v1[v1 - v0 > 0]
# v1 = np.arange(250, v_max, 0.1)


# ==================== 拟合数据预处理 =========================================

# 选取中间非零点
# v_head = v0
# v_tail = v_max
# for i in np.arange(round(v0) + 1, v_max):
#     if np.argmax(v_fv_map[i, :]) != 0:
#         v_head = i
#         break
# for i in np.arange(v_max - 1, v_head, -1):
#     if np.argmax(v_fv_map[i, :]) != 0:
#         v_tail = i + 1
#         break

# 选取mask所在点作为拟合的数据
vu = np.nonzero(mask)
# fv_val = [fv[vu[0, i], vu[1, i]] for i in np.arange(vu.shape[1])]
fv_val = fv[vu[0], vu[1]]
fv_val = np.array(fv_val)
# fu_val = [fu[vu[0, i], vu[1, i]] for i in np.arange(vu.shape[1])]
fu_val = fu[vu[0], vu[1]]
fu_val = np.array(fu_val)


flow_val = np.hstack((fv_val, fu_val))
# ==================== fv拟合 =========================================
"""
Parameter Define:

    phi: 车辆与Z方向夹角
    theta: 相机系统绕Z轴旋转
    delta_f: 前轮偏角
    vr: 后轴中心沿车辆轴线的速度
    vf: 前轴中心沿车辆轴线的速度
    h: 相机高度
    l：前后轴距离

    lambda_1 = (u1-u0)/fx*sin(theta)+(v1-v0)/fy*cos(theta)
    lambda_2 = (u1-u0)/fx*cos(theta)-(v1-v0)/fy*sin(theta)
    lambda_3 = lambda_2*h-lambda_1*Xd
    lambda_4 = h-lambda_1*Zd


Formula:


"""


def flow_func(x, theta, Xd, Zd, phi, h):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)
    lambda_3 = lambda_2 * h - lambda_1 * Xd
    lambda_4 = h - lambda_1 * Zd

    lambda_5 = (lambda_3 * cos(phi) - lambda_4 * sin(phi)) / \
               (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_2
    lambda_6 = (lambda_1 * h) / (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_1

    fv_e = fy * (-lambda_5 * sin(theta) + lambda_6 * cos(theta))
    fu_e = fx * (lambda_5 * cos(theta) + lambda_6 * sin(theta))

    output = np.hstack((fv_e, fu_e))
    return output


def fv_func(x, theta, Xd, Zd, phi, h):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)
    lambda_3 = lambda_2 * h - lambda_1 * Xd
    lambda_4 = h - lambda_1 * Zd

    lambda_5 = (lambda_3 * cos(phi) - lambda_4 * sin(phi)) / \
               (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_2
    lambda_6 = (lambda_1 * h) / (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_1

    output = fy * (-lambda_5 * sin(theta) + lambda_6 * cos(theta))

    return output


def fu_func(x, theta, Xd, Zd, phi, h):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)
    lambda_3 = lambda_2 * h - lambda_1 * Xd
    lambda_4 = h - lambda_1 * Zd

    lambda_5 = (lambda_3 * cos(phi) - lambda_4 * sin(phi)) / \
               (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_2
    lambda_6 = (lambda_1 * h) / (lambda_3 * sin(phi) + lambda_4 * cos(phi)) - lambda_1

    output = fx * (lambda_5 * cos(theta) + lambda_6 * sin(theta))

    return output


popt, _ = curve_fit(flow_func, vu, flow_val,
                    p0=[0, 0, 0, 0, 1.5],
                    bounds=[[-0.5 * pi, 0, 0, -0.5 * pi, 1], [0.5 * pi, 10, 10, 0.5 * pi, 3]])

# 构造验证拟合效果的数组
u = np.arange(u_max)
# v = np.arange(v_head, v_tail)
v = np.arange(v_max)
U, V = np.meshgrid(u, v)
VV = np.expand_dims(V, 0)
UU = np.expand_dims(U, 0)
vu_valid = np.append(VV, UU, axis=0)

# parameter estimation
theta_est = popt[0]
Xd_est = popt[1]
Zd_est = popt[2]
phi_est = popt[3]
h_est = popt[4]

# 拟合得到的fv
# flow_est = flow_func(vu_valid, vr_est, theta_est, delta_f_est)
fu_est = fu_func(vu_valid, theta_est, Xd_est, Zd_est, phi_est, h_est)
fv_est = fv_func(vu_valid, theta_est, Xd_est, Zd_est, phi_est, h_est)
fu_est = fu_est.reshape(v_max, u_max) * mask
fv_est = fv_est.reshape(v_max, u_max) * mask
# 绝对误差
fu_diff = fu_est - fu
fv_diff = fv_est - fv
# 相对误差
with np.errstate(divide='ignore', invalid='ignore'):
    fu_error = fu_diff / fu
    fu_error[~ np.isfinite(fu_error)] = 0  # 对 -inf, inf, NaN进行修正，置为0

with np.errstate(divide='ignore', invalid='ignore'):
    fv_error = fv_diff / fv
    fv_error[~ np.isfinite(fv_error)] = 0  # 对 -inf, inf, NaN进行修正，置为0

plt.figure("flow estimation")
plt.subplot(4, 2, 1)
plt.title("fv estimate")
plt.axis('off')
plt.imshow(fv_est)
plt.subplot(4, 2, 3)
plt.title("fv with mask")
plt.axis('off')
plt.imshow(fv)
plt.subplot(4, 2, 5)
plt.title("relative error")
plt.axis('off')
plt.imshow(fv_error)
plt.subplot(4, 2, 7)
plt.title("absolute error")
plt.axis('off')
plt.imshow(fv_diff)

# plt.figure("fu estimate")
plt.subplot(4, 2, 2)
plt.title("fu estimate")
plt.axis('off')
plt.imshow(fu_est)
plt.subplot(4, 2, 4)
plt.title("fu with mask")
plt.axis('off')
plt.imshow(fu)
plt.subplot(4, 2, 6)
plt.title("relative error")
plt.axis('off')
plt.imshow(fu_error)
plt.subplot(4, 2, 8)
plt.title("absolute error")
plt.axis('off')
plt.imshow(fu_diff)

print(f"theta:{theta_est * 180 / pi:0.6f}, phi:{phi_est * 180 / pi:0.6f}, Xd:{Xd_est:0.6f},"
      f" Zd:{Zd_est:0.6f}, h:{h_est:0.6f}")
