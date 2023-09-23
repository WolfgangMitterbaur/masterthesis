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
% cart and pole done function
% --------------------------------------

function isDone = cartPoleIsDoneFunction(obs,action,nextObs)
% Compute termination signal based on next observation.

    if iscell(nextObs)
        nextObs = nextObs{1};
    end

    % Angle at which to fail the episode
    thetaThresholdRadians = 12 * pi/180;
    %thetaThresholdRadians = 11 * pi/180;

    % Distance at which to fail the episode
    xThreshold = 2.4;
    %xThreshold = 10.0;

    x = nextObs(1,:);
    theta = nextObs(3,:);
    
    isDone = abs(x) > xThreshold | abs(theta) > thetaThresholdRadians;
    
end