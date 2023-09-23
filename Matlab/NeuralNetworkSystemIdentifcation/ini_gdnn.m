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
% Initialization of a GDNN-Network
% --------------------------------------

% clear all data at start
clear all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      General Initialization                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sample time for simulation
T_sim = 0.01;

% sample time for neural network
T = 0.01;

% maximum simulation time in seconds
train_length = 5000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Parameter Optimization                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% window length (in sample steps)
Q = 500;

% Levenberg-Marquardt-Parameter
mue = 3;
theta = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           APRBS-Excitation                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% minimum and maximum amplitude of the training data
min_ampl = -5;  
max_ampl =  5;  

% minimum and maximum time range of the APRBS signal in seconds
t_min = 1*T;  
t_max = 25*T;  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Network Construction                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number the inputs
numofinputs = 1;

% number of layers
numoflayers = 3;

% outputs of the weights (and management matrices) after the simulation
print_weights = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Do not change this!                           %

DI = zeros(numoflayers, numofinputs);              
DL = zeros(numoflayers);                           
idel = zeros(numoflayers, numofinputs,1);          
ldel = zeros(numoflayers, numoflayers,1);          
DIS = zeros(numoflayers, numofinputs);             
DLS = zeros(numoflayers);                          
idels = zeros(numoflayers, numofinputs,1);         
ldels = zeros(numoflayers, numoflayers,1);         
bs = zeros(1, numoflayers);                         
DIB = zeros(numoflayers, numofinputs);             
DLB = zeros(numoflayers);                          
idelb = zeros(numoflayers, numofinputs, 1);         
ldelb = zeros(numoflayers, numoflayers, 1);         
bb = zeros(1, numoflayers);                         
U_target = zeros(1, numoflayers);                   
U_out = zeros(1, numoflayers);                      
tp = tp_input(train_length/T, [t_min,t_max],[min_ampl,max_ampl], T);        
tp2 = tp_input_PRBS(train_length/T, [t_min,t_max], [min_ampl,max_ampl], T);  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Network Construction                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% delay Lines at the inputs
%[DI, idel] = setDI(toLayer, fromLayer, delayVector, DI, idel)
[DI,idel] = setDI(1, 1, [1:5], DI, idel);

% Delay Lines between the layers
%[DL, ldel] = setDL(toLayer, fromLayer, delayVector, DL, ldel)
[DL,ldel] = setDL(1, 1, [1:3], DL, ldel);
[DL,ldel] = setDL(1, 2, [1:3], DL, ldel);
[DL,ldel] = setDL(1, 3, [1:3], DL,ldel);
[DL,ldel] = setDL(2, 1, [0], DL, ldel);         % normal forward connection
[DL,ldel] = setDL(2, 2, [1:3], DL, ldel);
[DL,ldel] = setDL(2, 3, [1:3], DL, ldel);
[DL,ldel] = setDL(3, 2, [0], DL, ldel);         % normale forward connection
[DL,ldel] = setDL(3, 3, [1:3], DL, ldel);

% dimenstion of the inputs
R(1) = 1;

% number of neurons each layer
S(1) = 4;
S(2) = 3;
S(3) = 1;

% layer outputs compared to target
U_target(3) = 1;

% layer outputs lead out of S-Function
U_out(3) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Pruning                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% minimal number of weights when pruning
min_weights = 0;

% number of time steps between pruning actions
prune_interval = 100;

% max error (* old error) when taking a pruning step
prune_maxerr_OBS = 2;

% max error (* old error) when taking a pruning step
prune_maxerr_exact = 1; % only effecitve at Pruning 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           do not change                                 %

% maximal length of delays
D_max = max(max(max(max(idel))),max(max(max(ldel))));

% set of input layers
% a layer is an input layer if it has an input weight, of if it contains any
% delays with any of its weight matrices
X = zeros(1,numoflayers);
for i = 1:numoflayers
    for j = 1:numofinputs
        % if it has an input weight
        if DI(i,j) > 0
            X(i) = 1;
        end
    end
end

for i = 1:numoflayers
    for j = 1:numoflayers
        % if it contains any delays with any of its weight matrices
        % if (i > j && DL(i,j) > 1) || (i <= j && DL(i,j) > 0)
        if (DL(i,j) > 1) || (DL(i,j) == 1 && ldel(i,j,1) > 0)
            X(i) = 1;
        end
    end
end

% set of output layers
% a layer is an output layer if its output will be compared to a target
% during training, or if it is connected to an input layer through a matrix
% that has any delays associated with it
U = U_target;
for i = 1:numoflayers
    for j = 1:numoflayers
        % if (i > j && DL(i,j) > 1) || (i <= j && DL(i,j) > 0)
        if (DL(i,j) > 1) || (DL(i,j) == 1 && ldel(i,j,1) > 0)
            U(j) = 1;
        end
    end
end

% input connections
I=zeros(1,numoflayers*numofinputs);
for i = 1:numoflayers
    for j = 1:numofinputs
        if DI(i,j) > 0
            I((i-1)*numofinputs+j) = 1;
        end
    end
end

% forward connections
% Lfm is the set of indices of layers that directly connect forward to layer m
Lf = zeros(1,numoflayers^2);
for i = 1:numoflayers
    for j = 1:numoflayers
        if DL(i,j) > 0
            Lf((i-1)*numoflayers+j) = 1;
        end
    end
end

% backward connections
% Lbm is the set of indices of layers that are directly connected backwards to layer m
% (or to which layer m connects forward) and that contain no delays in the connection
Lb = zeros(1,numoflayers^2);
for i = 1:numoflayers
    for j = 1:numoflayers
        if DL(i,j) == 1 && ldel(i,j,1) == 0
            Lb((j-1)*numoflayers+i) = 1;
        end
    end
end

% dimension of the output of the S function
S_out = 0;
for i = 1:numoflayers
    if U_out(i) == 1
        S_out = S_out + S(i);
    end
end

% dimension of the signal to be learned and the error
S_target = 0;
for i = 1:numoflayers
    if U_target(i) ==  1
        S_target = S_target + S(i);
    end
end

% Set E_LW_U(x)
% E_LW_U(x) = {u e U mit der Eigenschaft (LW(x,u) ~= 0)}
E_LW_U = zeros(1,numoflayers^2);
for i = 1:numoflayers
    for j = 1:numoflayers
        if DL(i,j) > 0 && U(j) == 1
            E_LW_U((i-1)*numoflayers+j) = 1;
        end
    end
end

% convert matrices with length of delay vectors to row vectors
for i = 1:numoflayers
    if i == 1
        DI_vec = DI(1,:);
        DL_vec = DL(1,:);
        DIS_vec = DIS(1,:);
        DLS_vec = DLS(1,:);
        DIB_vec = DIB(1,:);
        DLB_vec = DLB(1,:);
    else
        DI_vec = [DI_vec DI(i,:)];
        DL_vec = [DL_vec DL(i,:)];
        DIS_vec = [DIS_vec DIS(i,:)];
        DLS_vec = [DLS_vec DLS(i,:)];
        DIB_vec = [DIB_vec DIB(i,:)];
        DLB_vec = [DLB_vec DLB(i,:)];
    end
end

% convert matrices to row vectors using the delay vectors for the inputs
h = 1;
for i = 1:numoflayers
    for j = 1:numofinputs
        for d = 1:DI(i,j)
            idel_vec(h) = idel(i,j,d);
            h = h+1;
        end
    end
end

h = 1;
for i = 1:numoflayers
    for j = 1:numofinputs
        for d = 1:DIS(i,j)
            idels_vec(h) = idels(i,j,d);
            h = h+1;
        end
    end
end

h = 1;
for i = 1:numoflayers
    for j = 1:numofinputs
        for d = 1:DIB(i,j)
            idelb_vec(h) = idelb(i,j,d);
            h = h+1;
        end
    end
end

% convert matrices to row vectors using the delay vectors for the layers
h = 1;
for i = 1:numoflayers
    for j = 1:numoflayers
        for d = 1:DL(i,j)
            ldel_vec(h) = ldel(i,j,d);
            h = h+1;
        end
    end
end

h = 1;
for i = 1:numoflayers
    for j = 1:numoflayers
        for d = 1:DLS(i,j)
            ldels_vec(h) = ldels(i,j,d);
            h = h+1;
        end
    end
end

h = 1;
for i = 1:numoflayers
    for j = 1:numoflayers
        for d = 1:DLB(i,j)
            ldelb_vec(h) = ldelb(i,j,d);
            h = h+1;
        end
    end
end

if(~exist('idels_vec','var'))
    idels_vec = 0;
end

if(~exist('ldels_vec','var'))
    ldels_vec = 0;
end

if(~exist('idelb_vec','var'))
    idelb_vec = 0;
end

if(~exist('ldelb_vec','var'))
    ldelb_vec = 0;
end

% determine matrices with the maximum delays for all inputs and layers
dimax = zeros(numoflayers,numofinputs);
dlmax = zeros(numoflayers,numoflayers);
for i = 1:numoflayers
    for j = 1:numofinputs
        dimax(i,j) = max(idel(i,j,:));
    end
    
    for j = 1:numoflayers
        dlmax(i,j) = max(ldel(i,j,:));
    end
end
dimax = max(dimax);
dlmax = max(dlmax);

% size of the weight vector
L = 0;
for i = 1:numoflayers
    % input weights
    for j = 1:numofinputs
        L = L + DI(i,j)*S(i)*R(j);
    end
    
    % layer weights
    for j = 1:numoflayers
        L = L + DL(i,j)*S(i)*S(j);
    end
    
    % biases
    L = L + S(i);
end

net_params = [Q, mue, theta, T, L, D_max, numoflayers, numofinputs, max(S), sum(R), S_out, S_target, print_weights, min_weights, prune_interval, train_length, prune_maxerr_OBS, prune_maxerr_exact];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      Initial initialization                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W_ini = rand(1,L) - 0.5;
W_alpha_ini = ones(1,L);


