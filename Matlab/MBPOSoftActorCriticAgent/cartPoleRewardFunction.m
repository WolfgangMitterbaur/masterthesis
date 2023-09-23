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
% cart and pole reward function
% --------------------------------------

function reward = cartPoleRewardFunction(obs,action,nextObs)
% Compute reward value based on the next observation.

    if iscell(nextObs)
        nextObs = nextObs{1};
    end

    % Distance at which to fail the episode
    xThreshold = 2.4;
    %xThreshold = 10.0;

    % Reward each time step the cart-pole is balanced
    rewardForNotFalling = 1; % original
    %rewardForNotFalling = 1.0;
    
    % Penalty when the cart-pole fails to balance
    penaltyForFalling = -50;

    x = nextObs(1,:);
    distReward = 1 - abs(x)/xThreshold;

    isDone = cartPoleIsDoneFunction(obs,action,nextObs);

    reward = zeros(size(isDone));
    reward(logical(isDone)) = penaltyForFalling;
    reward(~logical(isDone)) = ...
        0.5 * rewardForNotFalling + 0.5 * distReward(~logical(isDone));

    %reward(~logical(isDone)) = ...
    %    0.9 * rewardForNotFalling + 0.1 * distReward(~logical(isDone));

end
