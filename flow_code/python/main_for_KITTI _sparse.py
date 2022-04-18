import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# get paths
pic_num = 7
# 000002_10.png
# pic_num = 409
path = "E:\\Program Files\\dataset\\data_scene_flow"
original_img_path = path + f"\\training\\image_2\\000{pic_num:0>3d}_10.png"
optical_flow_path = path + f"\\training\\flow_noc\\000{pic_num:0>3d}_10.png"
semantic_path = path + f"\\training\\semantic\\000{pic_num:0>3d}_11.png"
# depth_path = path + r"\\depth\\00000040.png"

# ==================== 图1 原始RGB =========================================
rgb_img = Image.open(original_img_path)
plt.figure("raw RGB")
plt.imshow(rgb_img)
plt.axis('off')
plt.show()

# original_img_path = path + f"\\image_2\\0000000{pic_num+1}.png"
# rgb_img = Image.open(original_img_path)
# plt.figure("raw RGB next")
# plt.imshow(rgb_img)
# plt.axis('off')
# plt.show()
# ==================== 获取路面的mask =========================================
semantic_img = cv2.imread(semantic_path)
# road: [128,64,128]
# road line: (157, 234, 50)
mask_road = cv2.inRange(semantic_img, (0, 0, 255), (0, 0, 255))
# mask_roadline = cv2.inRange(semantic_img, (50, 234, 157), (50, 234, 157))
# mask = mask_road | mask_roadline
# mask = mask / 255
mask = mask_road / 255
plt.figure("mask")
plt.imshow(mask)
plt.axis('off')
plt.show()

# ==================== 图2 光流图 =========================================
# optical_flow_img = Image.open(optical_flow_path,-1)
optical_flow_img = cv2.imread(optical_flow_path, -1)
valid = optical_flow_img[:, :, 0]
fu = optical_flow_img[:, :, 2].astype(np.float32)
fv = optical_flow_img[:, :, 1].astype(np.float32)
fu = (fu - 2 ** 15) / 64.0
fv = (fv - 2 ** 15) / 64.0
fu[valid == 0] = np.nan
fv[valid == 0] = np.nan
fv = np.nan_to_num(fv, nan=-1)

plt.figure("fv")
# plt.subplot(2, 1, 1)
plt.imshow(fv)
plt.axis('off')

# plt.subplot(2, 1, 2)
# plt.imshow(fv)
# plt.title("original fv")
# plt.axis('off')

# ==================== 图3 光流fv图带掩码 =========================================

fu = optical_flow_img[:, :, 0] * 1920 * mask
# fv = -optical_flow_img[:, :, 1] * 1080 * mask
fv = fv * mask

plt.figure("fv with mask")
plt.imshow(fv)
plt.axis('off')
plt.show()

# ==================== 图4 v-fv曲线图 =========================================
v_max = image_h = fu.shape[0]
u_max = image_w = fu.shape[1]
u0 = 696.02
v0 = 224.18

# 去除底部
for i in range(v_max):
    for j in range(u_max):
        if fv[i, j]+i>v_max:
            fv[i, j]=-1

v_fv_map = np.zeros((v_max, round(fv.max()) + 1))

for i in range(v_max):
    for j in range(u_max):
        v_fv_map[i, round(fv[i, j])] += 1

plt.figure("v-fv-map")
plt.imshow(v_fv_map)
plt.show()

# index = np.argmax(v_fv_map, 1)  # 按行取最大值的位置
# points_set = np.arange(1080)
# plt.plot(np.arange(1080), index)


# ==================== v-fv公式计算图 =========================================
# fv = Zd(v1-v0)/(hf/(v1-v0)-Zd)
# Y = Zd*X^2 / (hf - Zd*X)
# f = fx = fy = image_w / 2.0

# h = 2.4
v0 = np.argmax(v_fv_map[:, 1])
print(v0)
v1 = np.arange(0, v_max, 0.1)
v1 = v1[v1 - v0 > 0]


# Zd = 1.25
# Y = Zd * (v1 - v0) / (h * f / (v1 - v0) - Zd)
# plt.plot(Y, v1, 'r')

# for hfzd in np.linspace(300, 700, 10):
#     Y = (v1 - v0) ** 2 / (hfzd - (v1 - v0))
#     plt.plot(Y, v1, 'r')


# ax = plt.gca()  # 获取到当前坐标轴信息
# ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
# ax.invert_yaxis()
# plt.show()


# ==================== 曲线拟合 =========================================
# def fitting_curve(x, var1, var2, var3):
#     # fv = var1(v1 - v0) / (var2 / (v1 - v0) - var1)
#     output = var1 * (x - var3) ** 2 / (var2 - (x - var3))
#     return output
#
#
# popt, _ = curve_fit(fitting_curve, np.arange(250, v_max), np.argmax(v_fv_map, 1)[250:]
#                     , bounds=[np.zeros(3), [3, 5000, 600]])
# Y = popt[0] * (v1 - popt[2]) ** 2 / (popt[1] - (v1 - popt[2]))
# plt.plot(Y, v1, 'r')
# print("popt[0]=", popt[0], "  popt[1]=", popt[1], "  popt[2]=", popt[2])


def fitting_curve(x, var1, var2):
    # fv = var1(v1 - v0) / (var2 / (v1 - v0) - var1)
    output = (x - var2) ** 2 / (var1 - (x - var2))
    # output =  (x - var2) ** 2 / (var1 - x)
    return output


popt, _ = curve_fit(fitting_curve, np.arange(180, v_max), np.argmax(v_fv_map, 1)[180:]
                    , bounds=[[400, 160], [1000, 230]])
v1 = np.arange(0, v_max, 0.1)
v1 = v1[v1 - popt[1] > 0]
Y = (v1 - popt[1]) ** 2 / (popt[1] - (v1 - popt[0]))
plt.plot(Y, v1, 'r')
print("popt[0]=", popt[0], "  popt[1]=", popt[1])

# ==================== 车的位姿 =========================================
# pose = np.loadtxt(r"F:\dataset\forFY\t10_s1_r0\poses.txt")
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

# X_w = h * ((u1 - u0) * np.cos(theta) - (v1 - v0) * np.sin(theta)) / (
#             (u1 - u0) * np.sin(theta) + (v1 - v0) * np.cos(theta))
# Y_w = h
# Z_w = f * h / ((u1 - u0) * np.sin(theta) + (v1-v0)*np.cos(theta))


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
