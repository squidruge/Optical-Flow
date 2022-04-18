clear; close all; clc;

% pic_num = 409;
pic_num = 924;
path = "E:/Program Files/dataset/KITTI/2011_09_26_drive_0101_sync";
original_img_path = path+'/image_2/0000000'+num2str(pic_num)+'.png';
optical_flow_path = path+'/flow/0000000'+num2str(pic_num)+'.png';
semantic_path = path + '/semantic/0000000'+num2str(pic_num)+'.png';

% 图1 原始RGB
rgb_img = imread(original_img_path);
fig1 = figure;
imshow(rgb_img)
title("raw RGB")
axis off
addToolbarExplorationButtons(gcf)


% 图2 光流图

optical_flow_img = imread(optical_flow_path);

flow_u = (double(optical_flow_img(:,:,1))-2^15)/64.0;
flow_v = (double(optical_flow_img(:,:,2))-2^15)/64.0;
flow_valid = optical_flow_img(:,:,3);
flow_u(flow_valid==0) = nan;
flow_v(flow_valid==0) = nan;

fig2 = figure;
subplot(2,1,1); imshow(flow_u, [], 'Colormap', jet(4096));title("original fu")
subplot(2,1,2); imshow(flow_v, [], 'Colormap', jet(4096));title("original fv")

addToolbarExplorationButtons(gcf)


% 获取路面的mask
semantic_img = imread(semantic_path);
%imshow(semantic_img)
mask=zeros(size(semantic_img,1),size(semantic_img,2),"double");
for i = 1 : size(semantic_img,1)
    for j = 1 : size(semantic_img,2)
        if semantic_img(i,j,1)==255 ...
                && semantic_img(i,j,2)==0 ...
                && semantic_img(i,j,3)==0
            
            mask(i,j)=1;
        end
    end
end


%图3 光流fv图带掩码

flow_u = flow_u .* mask;
flow_v = flow_v .* mask;

figure;
imshow(flow_v, [], 'Colormap', jet(4096));title("original fu")
title("fv with mask")
axis('off')


% 图4 v-fv曲线图

[v_max, u_max, ~] = size(flow_u);
image_h = v_max;
image_w = u_max;

% pre-set parameter
u0 =610; %609.5593;
v0 =173; %172.8540;
fx = 721.5377;
fy = 721.5377;
f = 721.5377;
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
popt=lsqcurvefit('flow_KITTI',[0 0 0 0 1.5],mask_vu,[fv_val,fu_val],[-0.5 * pi  0  0 -0.5 * pi 1],[0.5 * pi 10 10 0.5 * pi 3]);
                    
theta_est = popt(1);
Xd_est= popt(2);
Zd_est = popt(3);
phi = popt(4);
h = popt(5);

fprintf("theta=%f,Xd=%f,Zd=%f,phi=%f,h=%f",theta_est*180/pi, Xd_est, Zd_est, phi*180/pi, h)

% 构造验证拟合效果的数组
u_valid=1:u_max;
v_valid=v0:v_max;
[v_square, u_square]=meshgrid(v_valid,u_valid);
u_array=reshape(u_square,[],1);
v_array=reshape(v_square,[],1);
valid_vu=double(horzcat(v_array,u_array));

flow_est=flow_KITTI(popt,valid_vu);
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

