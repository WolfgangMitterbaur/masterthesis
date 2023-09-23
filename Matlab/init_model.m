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
% system model of the inverted pendulum
% --------------------------------------

clear all
close all
clc

%% parameters of the system
M   = 1.0;                  % Mass of cart
m   = 0.1;                  % mass of pendulum
I   = 0.05833333;           % MOI of Pendulum I = 1/3*m*l^2 + m * (l/2)^2
l   = 0.5;                  % COM of Pendulum
g   = 9.81;                 % Gravity Constant
b   = 0.00007892;           % viscous damping at pivot of Pendulum
c   = 0.63;                 % friction coefficient of cart

%% system model of the inverted pendulum
% matrices for the contoller design
% calculation of Sate Space Matrix/System Matrix (A,B,C,D)

%         0             0             1               0
%     
%         0             0             0               1
% A =
%         0      m^2l^2g/alfa    -(I+ml^2)c/alfa   -bml/alfa
%
%         0      mgl(M+m)/alfa    -mlc/alfa      -b(M+m)/alfa
%

%          0 
%
%          0 
%   B = 
%         (I+ml^2)/alfa
%
%         ml/alfa
%

AA = I*(M+m) + M*m*(l^2);                           % alpha (denominator)

aa = (((m*l)^2)*g)/AA;
bb = (I +m*(l^2))*c/AA;
cc = (b*m*l)/AA;
dd = (m*g*l*(M+m))/AA;
ee = (m*l*c)/AA;
ff = ((M+m)*b)/AA;
mm = (I +m*(l^2))/AA;
nn = (m*l)/AA;

oo = (2*l*(M+m))/AA;

A = [0 0 1 0; 0 0 0 1; 0 aa -bb -cc; 0 dd -ee -ff];
B = [0;0; mm; nn]; 
C = [1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1];
D = [0; 0; 0; 0];

%% simulation model of the inverted pendulum
% matrices for the simulation of the model
% using this, the simulation model can differ from the real model used for
% the controller desing
% calculation of Sate Space Matrix/System Matrix (A,B,C,D)

Msim =  1.0;                % Mass of cart
msim   = 0.1;               % mass of pendulum
Isim   = 0.05833333;        % MOI of Pendulum I = 1/3*m*l^2 + m * (l/2)^2
lsim   = 0.5;               % COM of Pendulum
gsim   = 9.81;              % Gravity Constant
bsim   = 0.00007892;        % viscous damping at pivot of Pendulum
csim   = 0.63;              % friction coefficient of cart

AAsim = Isim*(Msim + msim) + Msim*msim*(lsim^2);   % alpha (denominator)

aasim = (((msim*lsim)^2)*gsim)/AAsim;
bbsim = (Isim +msim*(lsim^2))*csim/AAsim;
ccsim = (bsim*msim*lsim)/AAsim;
ddsim = (msim*gsim*lsim*(Msim+msim))/AAsim;
eesim = (msim*lsim*csim)/AAsim;
ffsim = ((Msim+msim)*bsim)/AAsim;
mmsim = (Isim +msim*(lsim^2))/AAsim;
nnsim = (msim*lsim)/AAsim;

oosim = (2*lsim*(Msim+msim))/AAsim;

Asim = [0 0 1 0; 0 0 0 1; 0 aasim -bbsim -ccsim; 0 ddsim -eesim -ffsim];
Bsim = [0, 0; 0, 0; mmsim, 0; nnsim, oosim];
Csim = [1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1];
Dsim = [0, 0; 0, 0; 0, 0; 0, 0];

%% start condition
% start condition for the simulation run
xstart = 0.0;
xstartdelta = 0.0;
thetastart = pi;
thetastartdelta = 10*(pi/180);

