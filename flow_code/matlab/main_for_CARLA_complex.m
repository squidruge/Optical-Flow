clear; close all; clc;

% hyper params
theta = 0;  % degree  0 / 5

pic_num = 60;  %t10_s0_r0的img在20之后h=2.4m

% path = 'C:/dataset/CARLA/t10_s0_r'+num2str(theta);
path = "E:/Program Files/dataset/CARLA/t10_s0_r"+num2str(theta);
original_img_path = path+'/rgb/000000'+num2str(pic_num)+'.png';
optical_flow_path = path+'/optical_flow/000000'+num2str(pic_num)+'.tif';
optical_flow_path_png = path+'/optical_flow/000000'+num2str(pic_num)+'.png';
semantic_path = path + '/semantic/000000'+num2str(pic_num)+'.png';
depth_path = path + '/depth/000000'+num2str(pic_num)+'.png';

% 图1 原始RGB
rgb_img = imread(original_img_path);
fig1 = figure;
imshow(rgb_img)
title("raw RGB")
axis off
addToolbarExplorationButtons(gcf)


% 图2 光流图

% optical_flow_img = imread(optical_flow_path);

% flow_u = (double(optical_flow_img(:,:,1))-2^15)/64.0; flow_v =
% (double(optical_flow_img(:,:,2))-2^15)/64.0; flow_valid =
% optical_flow_img(:,:,3); flow_u(flow_valid==0) = nan;
% flow_v(flow_valid==0) = nan;
%
% fig2 = figure; subplot(2,1,1); imshow(flow_u, [], 'Colormap',
% jet(4096));title("original fu") subplot(2,1,2); imshow(flow_v, [],
% 'Colormap', jet(4096));title("original fv")
%
% addToolbarExplorationButtons(gcf)


% 获取路面的mask
semantic_img = imread(semantic_path);
%imshow(semantic_img)
% road: (128,64,128) road line: (157, 234, 50)
mask_road=zeros(size(semantic_img,1),size(semantic_img,2),"int8");
mask_roadline=zeros(size(semantic_img,1),size(semantic_img,2),"int8");
for i = 1 : size(semantic_img,1)
    for j = 1 : size(semantic_img,2)
        if semantic_img(i,j,1)==128 ...
                && semantic_img(i,j,2)==64 ...
                && semantic_img(i,j,3)==128
            mask_road(i,j)=1;
        end
        if semantic_img(i,j,1)==157 ...
                && semantic_img(i,j,2)==234 ...
                && semantic_img(i,j,3)==50
            mask_road(i,j)=1;
        end
    end
end

mask = mask_road | mask_roadline;

% mask = single(mask);


%图3 光流fv图带掩码

image_h = 1080;
image_w = 1920;

flow_u=ones(1,image_h*image_w);
flow_v=ones(1,image_h*image_w);
flow_img_origin = imread(optical_flow_path);
for i =1:size(flow_img_origin,2)/2
    flow_u(i)=flow_img_origin(2*i-1);
    flow_v(i)=flow_img_origin(2*i);
end

flow_u= reshape(flow_u, image_w, image_h);
flow_u=flow_u';
flow_v= reshape(flow_v, image_w, image_h);
flow_v=flow_v';

% flow_img = reshape(flow_img, 1080, 1920, 2); flow_img=flow_img'
%
% CH0 = flow_img(:, :, 2); CH1 = flow_img(:, :, 1);

flow_u = flow_u.* 1920 .* mask;
flow_v = -flow_v.* 1080 .* mask;

figure;
imshow(flow_v, [], 'Colormap', jet(4096));title("original fu")
title("fv with mask")
axis('off')


% pre-set parameter
v_max = 1080;
u_max  = 1920;

u0 = int16(image_w / 2);
v0 = int16(image_h / 2);

f = image_w / 2.0;
fx = image_w / 2.0;
fy = image_w / 2.0;
h = 2.4;
Zd = 1.25;
l = 2.7; %2500~2700 mm

% 选取mask所在点作为拟合的数据
[mask_v,mask_u] = find(mask~=0); % 非零的索引
fv_val=zeros(size(mask_v));
fu_val=zeros(size(mask_v));

for i = 1: size(mask_v,1)
    fv_val(i)=flow_v(mask_v(i), mask_u(i));
    fu_val(i)=flow_u(mask_v(i), mask_u(i));
end

mask_vu=horzcat(mask_v,mask_u);
% flow_val =vertcat(fv_val, fu_val);
% mask_vu=mask_vu';
% flow_val=flow_val';

% 曲线拟合
popt=lsqcurvefit('flow_CARLA',[1 0 0],mask_vu,[fv_val,fu_val],[0 -0.5 * pi -0.5 * pi],[5 0.5 * pi 0.5 * pi]);
vr_est = popt(1);
theta_est = popt(2);
delta_f_est = popt(3);
fprintf("vr=%f,theta=%f,delta_f=%f",vr_est,theta_est*180/pi,delta_f_est*180/pi)

% 构造验证拟合效果的数组
u_valid=1:u_max;
v_valid=v0:v_max;
[v_square, u_square]=meshgrid(v_valid,u_valid);
u_array=reshape(u_square,[],1);
v_array=reshape(v_square,[],1);
valid_vu=double(horzcat(v_array,u_array));

flow_est=flow_CARLA(popt,valid_vu);
fv_est=flow_est(:,1);
fu_est=flow_est(:,2);
fv_est=reshape(fv_est,u_max,[]);
fv_est=fv_est'.*mask(v0:v_max,:);
fu_est=reshape(fu_est,u_max,[]);
fu_est=fu_est'.*mask(v0:v_max,:);

fig_estimation = figure;
ax = subplot(3,2,1); imshow(fv_est, [], 'Colormap', jet(4096));title('flow v estimation')
ax = subplot(3,2,2); imshow(fu_est, [], 'Colormap', jet(4096));title('flow u estimation')
ax = subplot(3,2,3); imshow(flow_v(v0:v_max,:), [], 'Colormap', jet(4096));title('flow v truth')
ax = subplot(3,2,4); imshow(flow_u(v0:v_max,:), [], 'Colormap', jet(4096));title('flow u truth')
ax = subplot(3,2,5); imshow(fv_est-flow_v(v0:v_max,:), [], 'Colormap', jet(4096));title('flow v error')
ax = subplot(3,2,6); imshow(fu_est-flow_u(v0:v_max,:), [], 'Colormap', jet(4096));title('flow v error')
axis off
addToolbarExplorationButtons(gcf)

hold on;
% plot(fv_tmp/dmax*100,V_tmp+oy2,'g*')
hold off;

% hold on Y=cfun(x); plot(Y, x, 'r','LineWidth',1)


