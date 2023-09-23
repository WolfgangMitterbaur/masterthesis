% --------------------------------------
% Wolfgang Mitterbaur
% 23.09.2023
% IU International University
% Master Course: Artificial Intelligence
%
% Masterthesis: Artificial Intelligence Controls
%               Comparison with Control Technology
%               Investigated on the Inverted Pendulum Problem
% Matriculation ID: 31914987
% --------------------------------------
% non-linear model of the inverted pendulum
% --------------------------------------

function dx = invertedPendelnonlinear(t, x, u)
   
    % parameters
    M =  1.0;           % Mass of cart
    m   = 0.1;          % mass of pendulum
    I   = 0.05833333;   % MOI of Pendulum I = 1/3*m*l^2 + m * (l/2)^2
    l   = 0.5;          % COM of Pendulum
    g   = 9.81;         % Gravity Constant
    b   = 0.00007892;   % viscous damping at pivot of Pendulum
    c   = 0.63;         % friction coefficient of cart

    x1 = x(1);          % x1
    theta = x(2);       % theta x2
    x_dot = x(3);       % x_dot x3
    theta_dot = x(4);   % theta_dot x4
    
    F = u(1);           % input force

    Fstor = u(2);       % disturbance force

    dx1 = x(3);

    dx2 = x(4);

    alpha_a = ((m^2)*(l^2)*((sin(theta))^2)+ M*m*l^2 +(M+m)*I); 

    dx3  = (b*m*l*theta_dot*cos(theta) + (m^2)*(l^2)*g*sin(theta)*cos(theta) + (I + m*(l^2))*(F-c*x_dot + m*l*sin(theta)*theta_dot^2) )/alpha_a;

    dx4 = -(F*m*l*cos(theta)-c*m*l*x_dot*cos(theta) + (m^2)*(l^2)*(theta_dot^2)*sin(theta)*cos(theta)+ (M+m)*(b*theta_dot + m*g*l*sin(theta) + Fstor*2*l*cos(theta) ))/alpha_a;

    dx = [dx1; dx2; dx3; dx4];
