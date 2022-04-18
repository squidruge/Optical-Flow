import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# hyper params
theta = 0  # degree  0 / 5

# [1065,786] [1053,757]

# get paths
pic_num = 55  # t10_s0_r0的img在20之后h=2.4m
# path = f"C:\\dataset\\CARLA\\t10_s0_r{theta}"
path = "E:\\Program Files\\dataset\\CARLA\\t10_s0_r0"

original_img_path = path + f"\\rgb\\000000{pic_num}.png"
optical_flow_path = path + f"\\optical_flow\\000000{pic_num}.tif"
optical_flow_path_png = path + f"\\optical_flow\\000000{pic_num}.png"
semantic_path = path + f"\\semantic\\000000{pic_num}.png"
depth_path = path + f"\\depth\\000000{pic_num}.png"

# ==================== 图1 原始RGB =========================================
rgb_img = Image.open(original_img_path)
plt.figure(f"raw RGB {pic_num}")
plt.imshow(rgb_img)
plt.axis('off')
plt.show()

# pic_num = 39
# original_img_path = path + f"\\rgb\\000000{pic_num}.png"
# rgb_img = Image.open(original_img_path)
# plt.figure(f"raw RGB {pic_num}")
# plt.imshow(rgb_img)
# plt.axis('off')
# plt.show()
# ==================== 获取路面的mask =========================================
semantic_img = cv2.imread(semantic_path)
# road: (128,64,128)
# road line: (157, 234, 50)
mask_road = cv2.inRange(semantic_img, (128, 64, 128), (128, 64, 128))
mask_roadline = cv2.inRange(semantic_img, (50, 234, 157), (50, 234, 157))
mask = mask_road | mask_roadline
mask = mask / 255
# plt.figure("mask")
# plt.imshow(mask)
# plt.axis('off')
# plt.show()

# ==================== 图2 光流图 =========================================
# optical_flow_img = Image.open(optical_flow_path_png)
# plt.figure('optical flow img')
# plt.imshow(optical_flow_img)
# plt.axis('off')
# plt.show()

# ==================== 图3 光流fv图带掩码 =========================================
flow_img = Image.open(optical_flow_path)
flow_img = np.reshape(np.array(flow_img), (1080, 1920, 2))

CH0 = flow_img[:, :, 0]
CH1 = flow_img[:, :, 1]

fu = flow_img[:, :, 0] * 1920 * mask
fv = -flow_img[:, :, 1] * 1080 * mask
# plt.figure("fu with mask")
# plt.imshow(fu)
# plt.axis('off')
# plt.show()

v_max = image_h = 1080
u_max = image_w = 1920
u0 = int(image_w / 2)
v0 = int(image_h / 2)

# 去除底部
for i in range(v_max):
    for j in range(u_max):
        if fv[i, j] + i > v_max:
            fv[i, j] = 0

plt.figure("fv with mask")
plt.imshow(fv)
plt.axis('off')
plt.show()

# ==================== 图4 v-fv曲线图 =========================================


v_fv_map = np.zeros((v_max, round(fv.max()) + 1))
for i in range(v_max):
    for j in range(u_max):
        if round(fv[i, j]) != 0:
            v_fv_map[i, round(fv[i, j])] += 1

plt.figure("v-fv-map")
plt.imshow(v_fv_map)
plt.show()

# ==================== v-fv公式计算 =========================================
# fv = Zd(v1-v0)/(hf/(v1-v0)-Zd)
# Y  = Zd*X^2 / (hf - Zd*X)
#    = X^2 / (hf/Zd - X)
# hf_div_Zd = np.linspace(1000,3000,10)
f = fx = fy = image_w / 2.0
h = 2.4
Zd = 1.25


# v1 = np.arange(0, 1080, 0.1)
# v1 = v1[v1 - v0 > 0]


# Y = Zd * (v1 - v0) / (h * f / (v1 - v0) - Zd)
# for hfzd in hf_div_Zd:
#     Y = (v1 - v0)**2 / (hfzd - (v1 - v0))
#     plt.plot(Y, v1)


# for Zd in np.arange(0, 2, 0.2):
#     #     Y = Zd * (v1 - v0) ** 2 / (h * f - Zd * (v1 - v0))
#     Y = Zd * (v1 - v0) / (h * f / (v1 - v0) - Zd)
#     plt.plot(Y, v1)
# plt.figure()

# ax = plt.gca()  # 获取到当前坐标轴信息
# ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
# ax.invert_yaxis()
# plt.show()


# ==================== 曲线拟合 =========================================

def fitting_curve(v, var1):
    # fv = var1(v1 - v0) / (var2 / (v1 - v0) - var1)
    # fv = Zd(v1-v0)/(hf/(v1-v0)-Zd)

    # output = var1 * (v - var3) ** 2 / (var2 - (v - var3))
    # output = var1 * (v - v0) ** 2 / (var2 - (v - v0))
    output = (v - v0) ** 2 / (var1)
    return output


start_num = 600
v1 = np.arange(start_num, v_max, 0.1)

# 选取中间非零点
v_head = v0
v_tail = v_max
for i in np.arange(v0, v_max):
    if np.argmax(v_fv_map[i, :]) != 0:
        v_head = i
        break
for i in np.arange(v_max - 1, v_head, -1):
    if np.argmax(v_fv_map[i, :]) != 0:
        v_tail = i + 1
        break

# popt, _ = curve_fit(fitting_curve, np.arange(start_num, v_max), np.argmax(v_fv_map, 1)[start_num:])
popt, _ = curve_fit(fitting_curve, np.arange(v_head, v_tail), np.argmax(v_fv_map, 1)[v_head:v_tail],
                    p0=[3000], bounds=[[0], [10000]])
# Y = popt[0] * (v1 - v0) ** 2 / (popt[1] - (v1 - v0))
# Y = (v1 - v0) ** 2 / (popt[0] - (v1 - v0))
Y = (v1 - v0) ** 2 / (popt[0])

# Y = popt[0] * (v1 - popt[2]) ** 2 / (popt[1] - (v1 - popt[2]))
plt.plot(Y, v1, 'r')
print(f"popt[0]:{popt[0]}")
print(f"Zd=f*h/popt[0]=:{f * h / popt[0]}")
# print(f"popt[0]:{popt[0]}, popt[1]:{popt[1]}, popt[2]:{popt[2]}")

# ==================== 车的位姿 =========================================
pose = np.loadtxt(path + "\\poses.txt")
# # timestamp, x, y, z, qx, qy, qz, qw
# timestamp = pose[:, 0]
# x = pose[:, 1]
# y = pose[:, 2]
# z = pose[:, 3]
# qx = pose[:, 4]
# qy = pose[:, 5]
# qz = pose[:, 6]
# qw = pose[:, 7]
# # (x, y, z, w) format
# r1 = R.from_quat([qx[39], qy[39], qz[39], qw[39]])
# r2 = R.from_quat([qx[40], qy[40], qz[40], qw[40]])
# rotation_matrix1 = r1.as_matrix()
# rotation_matrix2 = r2.as_matrix()
#
# pos1 = np.array([x[39], y[39], z[39]])
# pos2 = np.array([x[40], y[40], z[40]])
# print(pos2 - pos1)
# print(rotation_matrix1)
# print(rotation_matrix2)
#
# K = np.array([[fx, 0, u0],
#               [0, fy, v0],
#               [0, 0, 1]])
# theta = 0
# R1 = np.array([[np.cos(theta), np.sin(theta), 0],
#                [-np.sin(theta), np.cos(theta), 0],
#                [0, 0, 1]])
#
# X_w = h * ((u1 - u0) * np.cos(theta) - (v1 - v0) * np.sin(theta)) / (
#             (u1 - u0) * np.sin(theta) + (v1 - v0) * np.cos(theta))
# Y_w = h
# Z_w = f * h / ((u1 - u0) * np.sin(theta) + (v1-v0)*np.cos(theta))
#

# ==================== 深度计算 =========================================
# depth_img = cv2.imread(depth_path).astype(np.float32)
# [B, G, R] = depth_img[:, :, 0], depth_img[:, :, 1], depth_img[:, :, 2]
#
# depth_in_meters = (R + G * 256.0 + B * 256 * 256) / (256.0 * 256 * 256 - 1) * 1000
# # cv2.imshow("depth_in_meters", depth_in_meters)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# plt.figure()
# plt.imshow(depth_in_meters)
# plt.show()

plt.figure("fu_u0")
plt.plot(range(v_max), fu[:, u0], 'r')
plt.show()

plt.figure("fu")
plt.imshow(fu)
plt.show()
