function output=flow_CARLA(a,x)
v = x(:,1);
u = x(:,2);
vr=a(1);
theta=a(2);
delta_f=a(3);
u0=960;
v0=540;
l=2.7;
h=2.4;
fx=960;
fy=960;

lambda_1 = (u - u0) ./ fx .* sin(theta) + (v - v0) ./ fy .* cos(theta);
lambda_2 = (u - u0) ./ fx .* cos(theta) - (v - v0) ./ fy .* sin(theta);


fv_e = -fy .* sin(theta) .* vr / h .* (lambda_1 .* lambda_2 - (tan(delta_f) / l) .* (1 + lambda_2.^2) .* h) ...
    + fy .* cos(theta) .* vr / h .* (lambda_1 .^2 - lambda_1 .* lambda_2 .* h .* tan(delta_f) / l);
fu_e = fx .* cos(theta) .* vr ./ h .* (lambda_1 .* lambda_2 - (tan(delta_f) / l) .* (1 + lambda_2 .^2) .* h) ...
    + fx .* sin(theta) .* vr ./ h .* (lambda_1 .^ 2 - lambda_1 .* lambda_2 .* h .* tan(delta_f) ./ l);

output = horzcat(fv_e, fu_e);
end