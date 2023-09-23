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
% collect data to train the neural network using simulation of LQR controller
% --------------------------------------

%% Run Simulink model and collect data
% set the solver and simulation options 

%% Generate training data
opt = simset('solver','ode4','FixedStep', 0.001);
% opt = simset('solver','ode45');                           % test an alternative

for i=1:1000
    rng(1);                                                 % set the seed for reproducible results random number between 0...1
    %xstart = rand - 0.5;                                   % have a inital position between [-0.5,0.5]
    xstartdelta = rand - 0.5;                               % have a inital position between [-0.5,0.5]
    
    rng(1);                                                 % set the seed for reproducible results
    
    thetastartdelta = rand * 0.6 - 0.3;                     % have a initial angle between [-0.2, 0.2]
    %thetastart = rand * 0.6 - 0.3;                         % have a initial angle between [-0.2, 0.2]
    
    [T,X] = sim('simulink_collect_data_NN',[0 15], opt);    % simulate system and record data
    
    % Create data object for each iteration
    states = [xx xtheta xxdot xthetadot];
    force = usim;
    
    % Create a data object to encapsulate the input/output data and their properties
    data = iddata(states, force, 0.005); 
    
    % Merge all simulations together
    if i > 1
        dataf = merge(dataf,data);
    else
        dataf = merge(data);
    end
end

%clearvars -except dataf;
save('collect_data_NN','dataf'); 

%% Generate testing data
%
% note: this is only used for the design process
%
% opt = simset('solver','ode4','SrcWorkspace','Current');
%
% for i=1:10
%    rng(2);                         % set the seed for reproducible results
%    x0 = rand-0.5;                  % have a inital position between [-0.5,0.5]
%    
%    rng(2);                         % set the seed for reproducible results
%    theta0 = rand * 0.4 - 0.2;      % have a initial angle between [-0.2,0.2]
%    
%    [T,X] = sim('ClosedLoop_LQRcont',[0 15],opt);     % simulate system and record data
%    
% Create data object for each iteration
%    states = [xx xxdot xtheta xthetadot];
%    force = u;
%    
% Create a data object to encapsulate the input/output data and their properties
%    data = iddata(states,force,0.005);
% Merge all simulations together
%    if i > 1
%        datatest = merge(datatest,data);
%    else
%        datatest = merge(data);
%    end
% end
%
% clearvars -except datatest;
% save('miw_invpend_data_test','datatest'); 

