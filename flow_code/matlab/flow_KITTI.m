function output=flow_KITTI(a,x)
% pre-set parameter
u0 = 609.5593;
v0 = 172.8540;
fx = 721.5377;
fy = 721.5377;
f = 721.5377;
l = 2.7; %2500~2700 mm

v = x(:,1);
u = x(:,2);

theta=a(1);
Xd=a(2);
Zd=a(3);
phi=a(4);
h=a(5);

lambda_1 = (u - u0) ./ fx .* sin(theta) + (v - v0) ./ fy .* cos(theta);
lambda_2 = (u - u0) ./ fx .* cos(theta) - (v - v0) ./ fy .* sin(theta);
lambda_3 = lambda_2 .* h - lambda_1 .* Xd;
lambda_4 = h - lambda_1 .* Zd;

lambda_5 = (lambda_3 .* cos(phi) - lambda_4 .* sin(phi)) ./ (lambda_3 .* sin(phi) + lambda_4 .* cos(phi)) - lambda_2;
lambda_6 = (lambda_1 .* h) ./ (lambda_3 .* sin(phi) + lambda_4 .* cos(phi)) - lambda_1;

fv_e = fy .* (-lambda_5 .* sin(theta) + lambda_6 .* cos(theta));
fu_e = fx .* (lambda_5 .* cos(theta) + lambda_6 .* sin(theta));

output = horzcat(fv_e, fu_e);

end