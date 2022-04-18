import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import math

# P_rect_03: f=721.5377 u0=6.095593 v0=172.8540

# get paths

# phi 比较大: 320~340 830~840 905~934
pic_num = 922
#v0 = 146  # 922
# v0 = 157  # 910
# v0 = 180  # 561
# pic_num = 116 330 331 410 411 532 560 561 699 700 875 876
path = "E:\Program Files\dataset\\KITTI\\2011_09_26_drive_0101_sync"
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
# fu[valid == 0] = np.nan
# fv[valid == 0] = np.nan


plt.figure("fu fv")
plt.subplot(2, 1, 1)
plt.imshow(fu)
plt.title("original fu")
plt.axis('off')

plt.subplot(2, 1, 2)

plt.imshow(fv)
plt.axis('off')

# plt.subplot(2, 1, 2)
# plt.imshow(fv)
# plt.title("original fv")
# plt.axis('off')

# ==================== 图3 光流fv图带掩码 =========================================
fv_origin = fv
fu = fu * mask
fv = fv * mask

plt.figure("fv with mask")
plt.imshow(fv)
plt.axis('off')
plt.show()

# ==================== 图4 v-fv曲线图 =========================================
v_max = image_h = fu.shape[0]
u_max = image_w = fu.shape[1]
u0 = 609.5593
v0 = 172.8540

# plt.figure("fu_u0")
# plt.plot(range(v_max), fu[:, round(u0)], 'r')
# plt.show()

for i in range(v_max):
    for j in range(u_max):
        if fv[i, j] + i > v_max:
            fv[i, j] = 0

v_fv_map = np.zeros((v_max, round(fv.max()) + 1))
for i in range(v_max):
    for j in range(u_max):
        if round(fv[i, j]) != 0:
            v_fv_map[i, round(fv[i, j])] += 1

plt.figure("v-fv-map")
plt.imshow(v_fv_map)
plt.show()

# ==================== v-fv公式计算图 =========================================
# fv = Zd(v1-v0)/(hf/(v1-v0)-Zd)
# Y = Zd*X^2 / (hf - Zd*X)
# f = fx = fy = image_w / 2.0


v1 = np.arange(0, v_max, 0.1)
v1 = v1[v1 - v0 > 0]
# v1 = np.arange(250, v_max, 0.1)


# ==================== 曲线拟合 =========================================

# 选取中间非零点
v_head = v0
v_tail = v_max
for i in np.arange(round(v0) + 1, v_max):
    if np.argmax(v_fv_map[i, :]) != 0:
        v_head = i
        break
for i in np.arange(v_max - 1, v_head, -1):
    if np.argmax(v_fv_map[i, :]) != 0:
        v_tail = i + 1
        break

# original
# k1=cos(phi)*Zd-sin(phi)*Xd
# k2=h*sin(phi)
# k3=h*cos(phi)
# fv=(f*h*v)/(f*k3-k2*v-k1*v)-v

# physical meaning
# k1=Zd-tan(phi)*Xd
# k2=phi
# k3=h
# fv=(f*h*v)/(f*h*cos(k2)-h*sin(k2)-k1*v*cos(k2))-v

vu = []
v_x = []
u_x = []
fv_val = []
for i in np.arange(v_head, v_tail):
    for j in np.arange(u_max):
        if fv[i, j] > 0:
            v_x.append(i)
            u_x.append(j)
            fv_val.append(fv[i, j])
# vu = np.append(v_x,u_x)
fv_val = np.array(fv_val)
vu = np.empty((2, fv_val.shape[0]), dtype="int32")
vu[0] = v_x
vu[1] = u_x
fv_val = fv_val[0:(v_tail - v_head) * u_max]

f = 958


# u = np.arange(u_max)
# v = np.arange(v_head, v_tail)
# x = np.append(u, v, axis=0)
# U, V = np.meshgrid(u, v)
# VV = np.expand_dims(V, 0)
# UU = np.expand_dims(U, 0)
# vu = np.append(VV, UU, axis=0)
# fv_val = fv.ravel()
# fv_val = fv_val[0:(v_tail - v_head) * u_max]


def fitting_curve(x, k1, k2, h):
    v = x[0]
    u = x[1]
    # fv = var1(v1 - v0) / (var2 / (v1 - v0) - var1)
    # output = var1 * (x - v0) ** 2 / (var2 - (x - v0))
    # output = var1 * (x - var3) ** 2 / (var2 - (x - var3))
    output = (f * h * (v - v0)) / \
             (f * h * math.cos(k2) - h * (u - u0) * math.sin(k2) - k1 * (v - v0) * math.cos(k2)) - (v - v0)

    return output
    #return output.ravel()

popt, _ = curve_fit(fitting_curve, vu, fv_val
                    , p0=[0.5, -0.2, 1.5], bounds=([0, -np.pi, 0], np.array([np.inf, 0, 2.3])))

# popt, _ = curve_fit(fitting_curve, np.arange(250, 300), np.argmax(v_fv_map, 1)[250:300],
#                     p0=[1000, 200], bounds=(np.zeros(2), np.array([5000, 600])))

# p0=[0, 1000, 200], bounds=(np.zeros(3), np.array([10, 5000, 600])))
# Y = popt[0] * (v1 - popt[2]) ** 2 / (popt[1] - (v1 - popt[2]))
# fv_est = np.zeros_like(fv)

# 构造验证拟合效果的数组
u = np.arange(u_max)
v = np.arange(v_head, v_tail)
# x = np.append(u, v, axis=0)
U, V = np.meshgrid(u, v)
VV = np.expand_dims(V, 0)
UU = np.expand_dims(U, 0)
vu1 = np.append(VV, UU, axis=0)

# parameter estimation
k1_est = popt[0]
k2_est = popt[1]
h_est = popt[2]
# u0_est = popt[3]

# 拟合得到的fv
fv_est = fitting_curve(vu1, k1_est, k2_est, h_est)
fv_est = fv_est.reshape(v_tail - v_head, u_max)
# fv 误差
fv_diff = fv_est - fv_origin[v_head:v_tail]
# for x in vu:
#     fv_est[x[0], x[1]] = (f * h_est * (x[0] - v0)) / \
#                          (f * h_est * math.cos(k2_est) - h_est * (x[1] - u0) * math.sin(k2_est) - k1_est * (
#                                  x[0] - v0) * math.cos(k2_est)) - (x[0] - v0)
# Y = (v1 - popt[1]) ** 2 / (popt[0] - (v1 - popt[1]))
plt.figure("fv estimate")
plt.subplot(3, 1, 1)
plt.imshow(fv_est)
plt.subplot(3, 1, 2)
plt.imshow(fv[v_head:v_tail, 0:u_max])
plt.subplot(3, 1, 3)
plt.imshow(fv_diff[:, 0:u_max])
# print(f"popt[0]:{popt[0]}, popt[1]:{popt[1]}, popt[2]:{popt[2]}")
print(f"k1:{k1_est}, k2:{k2_est}, h:{h_est}, phi:{k2_est*180/3.14}")
# plt.plot(np.argmax(v_fv_map, 1)[250:], np.arange(250, v_max), '-b')

# 通过特殊点估计参数
# ftan = fu[round(v0), round(u0)]
# print(f"f*tan(phi)={ftan:.6f}")
