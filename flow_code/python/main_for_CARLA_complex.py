import os

import cv2
import matplotlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit
from math import sin, cos, tan, pi
import matplotlib.transforms as mtransforms

# hyper params
theta_real = 0  # degree  0 / 5

# [1065,786] [1053,757]

# get paths
pic_num = 70
# t10_s0_r0的img在20之后h=2.4m

# path = f"C:\\dataset\\CARLA\\t10_s0_r{theta_real}"
path = f"E:\\Program Files\\dataset\\CARLA\\t10_s0_r{theta_real}"

original_img_path = path + f"\\rgb\\000000{pic_num}.png"
optical_flow_path = path + f"\\optical_flow\\000000{pic_num}.tif"
optical_flow_path_png = path + f"\\optical_flow\\000000{pic_num}.png"
semantic_path = path + f"\\semantic\\000000{pic_num}.png"
depth_path = path + f"\\depth\\000000{pic_num}.png"

fig_save_path = f"E:\\Program Files\\flow\\tits2022\\figs\\CARLA\\t10_s0_r{theta_real}" + f"\\000000{pic_num}"
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
mask.astype(int)

# ==================== 图2 光流图 =========================================
# optical_flow_img = Image.open(optical_flow_path_png)
# plt.figure('optical flow img')
# plt.imshow(optical_flow_img)
# plt.axis('off')
# plt.show()

# ==================== 图3 光流fv图带掩码 =========================================
flow_img = Image.open(optical_flow_path)
flow_img = np.reshape(np.array(flow_img), (1080, 1920, 2))

fu_origin = flow_img[:, :, 0] * 1920
fv_origin = -flow_img[:, :, 1] * 1080
# plt.figure("fu with mask")
# plt.imshow(fu)
# plt.axis('off')
# plt.show()

v_max = image_h = 1080
u_max = image_w = 1920
u0 = int(image_w / 2)
v0 = int(image_h / 2)

# 去除底部
# for i in range(v_max):
#     for j in range(u_max):
#         if fv_origin[i, j] + i > v_max:
#             mask[i, j] = 0

# # 去除左右边界
# for i in range(v_max):
#     for j in range(u_max):
#         if fv_origin[i, j] + i > u_max or fv_origin[i, j] + i <0:
#             mask[i, j] = 0

fu = fu_origin * mask
fv = fv_origin * mask

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

# ==================== pre-set parameter ===================================
f = fx = fy = image_w / 2.0
h = 2.4
Zd = 1.25
l = 2.7  # 2500~2700 mm
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
fv_val = fv[vu[0], vu[1]]
fv_val = np.array(fv_val)
# fv_val = (fv_val - np.min(fv_val)) / (np.max(fv_val) - np.min(fv_val))
fu_val = fu[vu[0], vu[1]]
fu_val = np.array(fu_val)
# fu_val = (fu_val - np.min(fu_val)) / (np.max(fu_val) - np.min(fu_val))
fv_min = np.min(fv_val)
fv_max = np.max(fv_val)
fv_val = (fv_val - fv_min) / (fv_max - fv_min)
fu_min = np.min(fu_val)
fu_max = np.max(fu_val)
fu_val = (fu_val - fu_min) / (fu_max - fu_min)
# fv_std = np.std(fv_val)
# fv_mean = np.mean(fv_val)
# fu_std = np.std(fu_val)
# fu_mean = np.mean(fu_val)
# fv_val = (fv_val - fv_mean) / fv_std
# fu_val = (fu_val - fu_mean) / fu_std

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

    fu = fx*cos(theta)*vr/h * (lambda_1*lambda_2 - (tan(delta_f)/l)*(1-lambda_2**2)*h)
            +fx*sin(theta)*vr/h * (lambda_1**2-lambda_1*lambda_2*h*tan(delta_f)/l)
    fv = -fy*sin(theta)*vr/h * (lambda_1*lambda_2 - (tan(delta_f)/l)*(1-lambda_2**2)*h)
            +fy*cos(theta)*vr/h * (lambda_1**2-lambda_1*lambda_2*h*tan(delta_f)/l)

"""


def flow_func(x, vr, theta, delta_f):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)

    # output = vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * math.tan(delta_f) / l)

    fv_e = -fy * sin(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
           + fy * cos(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    fu_e = fx * cos(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
           + fx * sin(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    # fv_e = -fy * sin(theta) * vr / h * (lambda_1 * lambda_2) \
    #        + fy * cos(theta) * vr / h * (lambda_1 ** 2)
    # fu_e = fx * cos(theta) * vr / h * (lambda_1 * lambda_2) \
    #        + fx * sin(theta) * vr / h * (lambda_1 ** 2)
    # fv_e = (fv_e - np.min(fv_e)) / (np.max(fv_e) - np.min(fv_e))
    # fu_e = (fu_e - np.min(fu_e)) / (np.max(fu_e) - np.min(fu_e))
    # fv_e = (fv_e - np.mean(fv_e)) / np.std(fv_e)
    # fu_e = (fu_e - np.mean(fu_e)) / np.std(fu_e)
    # fv_e = (fv_e - fv_mean) / fv_std
    # fu_e = (fu_e - fu_mean) / fu_std
    fv_e = (fv_e - fv_min) / (fv_max - fv_min)
    fu_e = (fu_e - fu_min) / (fu_max - fu_min)
    output = np.hstack((fv_e, fu_e))
    return output
    # return output.ravel()


def fv_func(x, vr, theta, delta_f):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)

    # output = vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * math.tan(delta_f) / l)

    output = -fy * sin(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
             + fy * cos(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    # output = -fy * sin(theta) * vr / h * (lambda_1 * lambda_2) \
    #          + fy * cos(theta) * vr / h * (lambda_1 ** 2)
    return output.ravel()


def fu_func(x, vr, theta, delta_f):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)

    # output = vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * math.tan(delta_f) / l)
    output = fx * cos(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
             + fx * sin(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    # output = fx * cos(theta) * vr / h * (lambda_1 * lambda_2) \
    #          + fx * sin(theta) * vr / h * (lambda_1 ** 2)
    return output.ravel()


popt, _ = curve_fit(flow_func, vu, flow_val,
                    p0=[1, 5 / 180 * 3.14, 0],
                    bounds=[[0, -0.5 * pi, -0.5 * pi], [5, 0.5 * pi, 0.5 * pi]])

# 构造验证拟合效果的数组
u = np.arange(u_max)
v = np.arange(v_max)
U, V = np.meshgrid(u, v)
VV = np.expand_dims(V, 0)
UU = np.expand_dims(U, 0)
vu_valid = np.append(VV, UU, axis=0)

# parameter estimation
vr_est = popt[0]
theta_est = popt[1]
delta_f_est = popt[2]

# 拟合得到的fv
# flow_est = flow_func(vu_valid, vr_est, theta_est, delta_f_est)
fu_est = fu_func(vu_valid, vr_est, theta_est, delta_f_est)
fv_est = fv_func(vu_valid, vr_est, theta_est, delta_f_est)
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
#



print(f"vr:{vr_est}, theta:{theta_est * 180 / pi}, delta_f:{delta_f_est * 180 / pi}")

# 保存图片
# plt.savefig()
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

fv_diff_mean = np.sum(fv_diff) / (np.sum(mask))
fv_error_mean = np.sum(fv_error) / (np.sum(mask))
fu_diff_mean = np.sum(fu_diff) / (np.sum(mask))
fu_error_mean = np.sum(fu_error) / (np.sum(mask))

theta_est = theta_est * 180 / pi
theta_absolute_error = theta_est - theta_real

file_handle = open(fig_save_path + '\\error.txt', mode='w')
file_handle.write(f'fv_relative_mean:{fv_error_mean}\n')
file_handle.write(f'fv_absolute_mean:{fv_diff_mean}\n')
file_handle.write(f'fu_relative_mean:{fu_error_mean}\n')
file_handle.write(f'fu_absolute_mean:{fu_diff_mean}\n\n')
file_handle.write(f'theta_absolute_error:{theta_absolute_error}\n')
if theta_real != 0:
    theta_relative_error = abs(theta_est - theta_real) / theta_real
    file_handle.write(f'theta_relative_error:{theta_relative_error}\n')
file_handle.close()
# def add_right_cax(ax, pad, width):
#
#     # 在一个ax右边追加与之等高的cax.
#     # pad是cax与ax的间距,width是cax的宽度.
#
#     axpos = ax.get_position()
#     caxpos = mtransforms.Bbox.from_extents(
#         axpos.x1 + pad,
#         axpos.y0,
#         axpos.x1 + pad + width,
#         axpos.y1
#     )
#     cax = ax.figure.add_axes(caxpos)
#
#     return cax
#
#
# norm = matplotlib.colors.Normalize(vmin=min(fv.min(), fv_est.min()),
#                                    vmax=max(fv.max(), fv_est.max()))
#
# # plot fv
# fig = plt.figure()
# ax = plt.axes()
# h1 = plt.imshow(fv_est, norm=norm)
# plt.axis('off')
#
# cax = add_right_cax(ax, pad=0.01, width=0.02)
# # cbar_ax = fig.add_axes(rect)
# cb = plt.colorbar(h1, cax=cax, ticks=None)
# # 设置colorbar标签字体等
# cb.ax.tick_params(labelsize=10)  # 设置色标刻度字体大小。
# font = {'family': 'serif',
#         #       'color'  : 'darkred',
#         'color': 'black',
#         'weight': 'normal',
#         'size': 16,
#         }
#
# plt.savefig(fig_save_path + "\\fv_estimation.png", bbox_inches='tight', pad_inches=0)
#
# fig = plt.figure()
# ax = plt.axes()
# h1 = plt.imshow(fv, norm=norm)
# plt.axis('off')
#
# cax = add_right_cax(ax, pad=0.01, width=0.02)
# # cbar_ax = fig.add_axes(rect)
# cb = plt.colorbar(h1, cax=cax, ticks=None)
# # 设置colorbar标签字体等
# cb.ax.tick_params(labelsize=10)  # 设置色标刻度字体大小。
# font = {'family': 'serif',
#         #       'color'  : 'darkred',
#         'color': 'black',
#         'weight': 'normal',
#         'size': 16,
#         }
#
# plt.savefig(fig_save_path + "\\fv_truth.png", bbox_inches='tight', pad_inches=0)
#
# fig = plt.figure()
# ax = plt.axes()
# h1 = plt.imshow(fv_error)
# plt.axis('off')
# cax = add_right_cax(ax, pad=0.01, width=0.02)
# cb = plt.colorbar(h1, cax=cax, ticks=None)
# cb.ax.tick_params(labelsize=10)
# plt.savefig(fig_save_path + "\\fv_relative.png", bbox_inches='tight', pad_inches=0)
#
# fig = plt.figure()
# ax = plt.axes()
# h1 = plt.imshow(fv_diff)
# plt.axis('off')
# cax = add_right_cax(ax, pad=0.01, width=0.02)
# cb = plt.colorbar(h1, cax=cax, ticks=None)
# cb.ax.tick_params(labelsize=10)
# plt.savefig(fig_save_path + "\\fv_absolute.png", bbox_inches='tight', pad_inches=0)
#
# # plot fu
# fig = plt.figure()
# ax = plt.axes()
# h1 = plt.imshow(fu_est, norm=norm)
# plt.axis('off')
#
# cax = add_right_cax(ax, pad=0.01, width=0.02)
# # cbar_ax = fig.add_axes(rect)
# cb = plt.colorbar(h1, cax=cax, ticks=None)
# # 设置colorbar标签字体等
# cb.ax.tick_params(labelsize=10)  # 设置色标刻度字体大小。
# font = {'family': 'serif',
#         #       'color'  : 'darkred',
#         'color': 'black',
#         'weight': 'normal',
#         'size': 16,
#         }
#
# plt.savefig(fig_save_path + "\\fu_estimation.png", bbox_inches='tight', pad_inches=0)
#
# fig = plt.figure()
# ax = plt.axes()
# h1 = plt.imshow(fu, norm=norm)
# plt.axis('off')
#
# cax = add_right_cax(ax, pad=0.01, width=0.02)
# # cbar_ax = fig.add_axes(rect)
# cb = plt.colorbar(h1, cax=cax, ticks=None)
# # 设置colorbar标签字体等
# cb.ax.tick_params(labelsize=10)  # 设置色标刻度字体大小。
# font = {'family': 'serif',
#         #       'color'  : 'darkred',
#         'color': 'black',
#         'weight': 'normal',
#         'size': 16,
#         }
#
# plt.savefig(fig_save_path + "\\fu_truth.png", bbox_inches='tight', pad_inches=0)
#
# fig = plt.figure()
# ax = plt.axes()
# h1 = plt.imshow(fu_error)
# plt.axis('off')
# cax = add_right_cax(ax, pad=0.01, width=0.02)
# cb = plt.colorbar(h1, cax=cax, ticks=None)
# cb.ax.tick_params(labelsize=10)
# plt.savefig(fig_save_path + "\\fu_relative.png", bbox_inches='tight', pad_inches=0)
#
# fig = plt.figure()
# ax = plt.axes()
# h1 = plt.imshow(fu_diff)
# plt.axis('off')
# cax = add_right_cax(ax, pad=0.01, width=0.02)
# cb = plt.colorbar(h1, cax=cax, ticks=None)
# cb.ax.tick_params(labelsize=10)
# plt.savefig(fig_save_path + "\\fu_absolute.png", bbox_inches='tight', pad_inches=0)

#
#
# plt.figure("fv_est")
# plt.axis('off')
# h1=plt.imshow(fv_est)
# plt.colorbar(h1,pad=0.05)
# # plt.clim(-200, 200)
# plt.savefig(fig_save_path + "\\fv_estimation.png", bbox_inches='tight', pad_inches = -0.1)
# plt.figure("fv_truth")
# plt.axis('off')
# plt.imshow(fv)
# plt.colorbar(pad=0.05)
# plt.savefig(fig_save_path + "\\fv_truth.png", bbox_inches='tight', pad_inches = -0.1)
# plt.figure("fv_relative")
# plt.axis('off')
# plt.imshow(fv_error)
# plt.colorbar(pad=0.05)
# plt.savefig(fig_save_path + "\\fv_relative.png", bbox_inches='tight', pad_inches = -0.1)
# plt.figure("fv_absolute")
# plt.axis('off')
# plt.imshow(fv_diff)
# plt.colorbar(pad=0.05)
# plt.savefig(fig_save_path + "\\fv_absolute.png", bbox_inches='tight', pad_inches = -0.1)
#
# plt.figure("fu_estimation")
# plt.axis('off')
# plt.imshow(fu_est)
# plt.colorbar()
# # plt.clim(-200, 200)
# plt.savefig(fig_save_path + "\\fu_estimation.png", bbox_inches='tight', pad_inches = -0.1)
# plt.figure("fu_truth")
# plt.axis('off')
# plt.imshow(fu)
# plt.colorbar()
# plt.savefig(fig_save_path + "\\fu_truth.png", bbox_inches='tight', pad_inches = -0.1)
# plt.figure("fu_relative")
# plt.axis('off')
# plt.imshow(fu_error)
# plt.colorbar()
# plt.savefig(fig_save_path + "\\fu_relative.png", bbox_inches='tight', pad_inches = -0.1)
#
# # plt.figure("fu_absolute")
# plt.axis('off')
# plt.imshow(fu_diff)
# # plt.colorbar()
# plt.savefig(fig_save_path + "\\fu_absolute.png", bbox_inches='tight', pad_inches = -0.1)

# cv2.imwrite(fig_save_path + "\\fv_estimation.png", fv_est)
# cv2.imwrite(fig_save_path + "\\fv_truth.png", fv)
# cv2.imwrite(fig_save_path + "\\fv_relative.png", fv_error)
# cv2.imwrite(fig_save_path + "\\fv_absolute.png", fv_diff)
# cv2.imwrite(fig_save_path + "\\fu_estimation.png", fu_est)
# cv2.imwrite(fig_save_path + "\\fu_truth.png", fu)
# cv2.imwrite(fig_save_path + "\\fu_relative.png", fu_error)
# cv2.imwrite(fig_save_path + "\\fu_absolute.png", fu_diff)

#
# # 验证 phi
# pose = np.loadtxt(path + "\\poses.txt")
#
# # # timestamp, x, y, z, qx, qy, qz, qw
# timestamp = pose[:, 0]
# x = pose[:, 1]
# y = pose[:, 2]
# z = pose[:, 3]
# qx = pose[:, 4]
# qy = pose[:, 5]
# qz = pose[:, 6]
# qw = pose[:, 7]
# # (x, y, z, w) format
# r1 = R.from_quat([qx[pic_num], qy[pic_num], qz[pic_num], qw[pic_num]])
# r2 = R.from_quat([qx[pic_num + 1], qy[pic_num + 1], qz[pic_num + 1], qw[pic_num + 1]])
# rotation_matrix1 = r1.as_matrix()
# rotation_matrix2 = r2.as_matrix()
# rot_mat = np.matmul(rotation_matrix1, np.linalg.pinv(rotation_matrix2))
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


'''
def fv_func(x, vr, theta, delta_f):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)

    # output = vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * math.tan(delta_f) / l)

    output = -fy * sin(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
             + fy * cos(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    # output = -fy * sin(theta) * vr / h * (lambda_1 * lambda_2) \
    #          + fy * cos(theta) * vr / h * (lambda_1 ** 2)
    return output.ravel()




popt, _ = curve_fit(fv_func, vu, fv_val,
                    p0=[10, 0, 0],
                    bounds=[[0, 0, 0], [100000, 0.5 * pi, 0.5 * pi]])

# 构造验证拟合效果的数组
u = np.arange(u_max)
v = np.arange(v_head, v_tail)
U, V = np.meshgrid(u, v)
VV = np.expand_dims(V, 0)
UU = np.expand_dims(U, 0)
vu_valid = np.append(VV, UU, axis=0)

# parameter estimation
vr_est = popt[0]
theta_est = popt[1]
delta_f_est = popt[2]

# 拟合得到的fv
fv_est = fv_func(vu_valid, vr_est, theta_est, delta_f_est)
fv_est = fv_est.reshape(v_tail - v_head, u_max)
# fv 误差
fv_diff = fv_est - fv_origin[v_head:v_tail]

plt.figure("fv estimate")
plt.subplot(3, 1, 1)
plt.title("fv estimate")
plt.axis('off')
plt.imshow(fv_est * mask[v_head:v_tail])
plt.subplot(3, 1, 2)
plt.title("fv with mask")
plt.axis('off')
plt.imshow(fv[v_head:v_tail, 0:u_max])
plt.subplot(3, 1, 3)
plt.title("estimation error")
plt.axis('off')
plt.imshow(fv_diff[:, 0:u_max] * mask[v_head:v_tail])
print(f"vr:{vr_est}, theta:{theta_est * 180 / 3.14}, delta_f:{delta_f_est * 180 / 3.14}")


# ==================== fu拟合 =========================================

def fu_func(x, vr, theta, delta_f):
    v = x[0]
    u = x[1]

    lambda_1 = (u - u0) / fx * sin(theta) + (v - v0) / fy * cos(theta)
    lambda_2 = (u - u0) / fx * cos(theta) - (v - v0) / fy * sin(theta)

    # output = vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * math.tan(delta_f) / l)
    output = fx * cos(theta) * vr / h * (lambda_1 * lambda_2 - (tan(delta_f) / l) * (1 + lambda_2 ** 2) * h) \
             + fx * sin(theta) * vr / h * (lambda_1 ** 2 - lambda_1 * lambda_2 * h * tan(delta_f) / l)
    # output = fx * cos(theta) * vr / h * (lambda_1 * lambda_2) \
    #          + fx * sin(theta) * vr / h * (lambda_1 ** 2)
    return output.ravel()


popt, _ = curve_fit(fu_func, vu, fu_val,
                    p0=[10, 0, 0],
                    bounds=[[0, -0.5 * pi, -0.5 * pi], [100000, 0.5 * pi, 0.5 * pi]])

# parameter estimation
vr_est = popt[0]
theta_est = popt[1]
delta_f_est = popt[2]

# 拟合得到的fv
fu_est = fu_func(vu_valid, vr_est, theta_est, delta_f_est)
fu_est = fu_est.reshape(v_tail - v_head, u_max)
# fv 误差
fu_diff = fu_est - fu_origin[v_head:v_tail]

plt.figure("fu estimate")
plt.subplot(3, 1, 1)
plt.title("fu estimate")
plt.axis('off')
plt.imshow(fu_est * mask[v_head:v_tail])
plt.subplot(3, 1, 2)
plt.title("fu with mask")
plt.axis('off')
plt.imshow(fu[v_head:v_tail, 0:u_max])
plt.subplot(3, 1, 3)
plt.title("estimation error")
plt.axis('off')
plt.imshow(fu_diff[:, 0:u_max] * mask[v_head:v_tail])
print(f"vr:{vr_est}, theta:{theta_est * 180 / 3.14}, delta_f:{delta_f_est * 180 / 3.14}")
'''
