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
% train mbpo soft actor critic agent
% --------------------------------------

%clear all
%clc

% NN agent for own model

% https://de.mathworks.com/help/reinforcement-learning/ug/train-mbpo-agent-to-balance-cart-pole-system-example.html

% This example shows how to train a model-based policy optimization (MBPO). 
% MBPO agents use an environment model to generate more experiences 
% while training a base agent. In this example, the base agent 
% is a soft actor-critic (SAC) agent.

% The build-int MBPO agent is based on a model-based policy optimization 
% algorithm in [1]. The original MBPO algorithm trains an ensemble of 
% stochastic models. In contrast, this example trains an ensemble of 
% deterministic models.



%% parameters
M =  1.0;                   % Mass of cart
m   = 0.1;                  % mass of pendulum
I   = 0.05833333;           % MOI of Pendulum I = 1/3*m*l^2 + m * (l/2)^2
l   = 0.5;                  % COM of Pendulum
g   = 9.81;                 % Gravity Constant
b   = 0.00007892;           % viscous damping at pivot of Pendulum
c   = 0.63;                 % friction coefficient of cart

%% LQR control design 

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

AA = I*(M+m) + M*m*(l^2);                           % alpha (nenner)

aa = (((m*l)^2)*g)/AA;
bb = (I +m*(l^2))*c/AA;
cc = (b*m*l)/AA;
dd = (m*g*l*(M+m))/AA;
ee = (m*l*c)/AA;
ff = ((M+m)*b)/AA;
mm = (I +m*(l^2))/AA;
nn = (m*l)/AA;

oo = (2*l)/AA;

A  =  [0 0 1 0; 0 0 0 1; 0 aa -bb -cc; 0 dd -ee -ff];
B  = [0;0; mm; nn]; 
C = [1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1];
D = [0; 0; 0; 0];

xstartdelta = 0;
x0 = 0;
thetastartdelta = 0;
theta0 = 0;

% start conditions of the model
xstart = 0;
thetastart = pi;

% observations
obsInfo = rlNumericSpec([4 1]);
obsInfo.Name = "observations";
obsInfo.Description = "x theta xdot thetadot";
% actoins
actInfo = rlNumericSpec([1 1]);
actInfo.Name = "force";
actInfo.Description = "u";

% Create a predefined environment interface for the cart-pole system
open_system('simulink_model_mbpo.slx')
env = rlSimulinkEnv("simulink_model_mbpo","simulink_model_mbpo/RL Agent", obsInfo, actInfo);
% reset function
env.ResetFcn = @(in)localResetFcn(in);

obsInfo = getObservationInfo(env);
numObservations = obsInfo.Dimension(1);
actInfo = getActionInfo(env);

% Fix the random generator seed for reproducibility.
rng(0)

%% Define Model-Free Off-Policy Agent
% Create a SAC base agent with a default network structure. 
% For more information on SAC agents, see Soft Actor-Critic Agents. 
% For an environment with a continuous action space, you can also use a 
% DDPG or TD3 base agent. For discrete environments, you can use a DQN base agent. 

agentOpts = rlSACAgentOptions;
agentOpts.MiniBatchSize = 256;
initOpts = rlAgentInitializationOptions("NumHiddenUnit",64);
baseagent = rlSACAgent(obsInfo,actInfo,initOpts,agentOpts);
baseagent.AgentOptions.ActorOptimizerOptions.LearnRate = 4e-5; %1e-4;
baseagent.AgentOptions.CriticOptimizerOptions(1).LearnRate = 4e-5; %1e-4;
baseagent.AgentOptions.CriticOptimizerOptions(2).LearnRate = 4e-5; %1e-4;
baseagent.AgentOptions.NumGradientStepsPerUpdate = 5;

% Create three deterministic transition functions
net1 = createDeterministicTransitionNetwork(4,1);
transitionFcn = rlContinuousDeterministicTransitionFunction(net1,...
    obsInfo,...
    actInfo,...
    ObservationInputNames="state",...
    ActionInputNames="action",...
    NextObservationOutputNames="nextObservation");

net2 = createDeterministicTransitionNetwork(4,1);
transitionFcn2 = rlContinuousDeterministicTransitionFunction(net2,...
    obsInfo,...
    actInfo,...
    ObservationInputNames="state",...
    ActionInputNames="action",...
    NextObservationOutputNames="nextObservation");

net3 = createDeterministicTransitionNetwork(4,1);
transitionFcn3 = rlContinuousDeterministicTransitionFunction(net3,...
    obsInfo,...
    actInfo,...
    ObservationInputNames="state",...
    ActionInputNames="action",...
    NextObservationOutputNames="nextObservation");

useGroundTruthReward = true;
if useGroundTruthReward
    rewardFcn = @cartPoleRewardFunction;
else
    % This neural network uses action and next observation as inputs.
    rewardnet = createRewardNetworkActionNextObs(4,1);
    rewardFcn = rlContinuousDeterministicRewardFunction(rewardnet,...
        obsInfo,...
        actInfo, ...
        ActionInputNames="action",...
        NextObservationInputNames="nextState");
end

useGroundTruthIsDone = true;
if useGroundTruthIsDone
    isdoneFcn = @cartPoleIsDoneFunction;
else
    % This neural network uses only next obesrvation as inputs.
    isdoneNet = createIsDoneNetwork(4);
    isdoneFcn = rlIsDoneFunction(isdoneNet,...
        obsInfo,...
        actInfo,...
        NextObservationInputNames="nextState");    
end

generativeEnv = rlNeuralNetworkEnvironment(obsInfo,actInfo, ...
    [transitionFcn,transitionFcn2,transitionFcn3],rewardFcn,isdoneFcn);
% Reset model environment.
reset(generativeEnv);

MBPOAgentOpts = rlMBPOAgentOptions;
MBPOAgentOpts.NumEpochForTrainingModel = 1;
MBPOAgentOpts.NumMiniBatches = 15;
MBPOAgentOpts.MiniBatchSize = 256;
MBPOAgentOpts.ModelExperienceBufferLength = 60000;
MBPOAgentOpts.RealSampleRatio = 0.2; %0.2;
MBPOAgentOpts.ModelRolloutOptions.NumRollout = 20000;
MBPOAgentOpts.ModelRolloutOptions.HorizonUpdateSchedule = "piecewise";
MBPOAgentOpts.ModelRolloutOptions.HorizonUpdateFrequency = 100;
MBPOAgentOpts.ModelRolloutOptions.Horizon = 1;
MBPOAgentOpts.ModelRolloutOptions.HorizonMax = 3;

transitionOptimizerOptions1 = rlOptimizerOptions(...
    LearnRate= 4e-5,...                                  %  1e-4
    GradientThreshold=1.0);
transitionOptimizerOptions2 = rlOptimizerOptions(...
    LearnRate= 4e-5,...                                  %  1e-4
    GradientThreshold=1.0);
transitionOptimizerOptions3 = rlOptimizerOptions(...
    LearnRate= 4e-5,...                                  %  1e-4
    GradientThreshold=1.0);
MBPOAgentOpts.TransitionOptimizerOptions = ...
    [transitionOptimizerOptions1,...
    transitionOptimizerOptions2,...
    transitionOptimizerOptions3];

rewardOptimizerOptions = rlOptimizerOptions(...
    LearnRate= 4e-5,...                              %  1e-4
    GradientThreshold=1.0);
MBPOAgentOpts.RewardOptimizerOptions = rewardOptimizerOptions;

isdoneOptimizerOptions = rlOptimizerOptions(...
    LearnRate= 4e-5,...                          %  1e-4
    GradientThreshold=1.0);
MBPOAgentOpts.IsDoneOptimizerOptions = isdoneOptimizerOptions;

agent = rlMBPOAgent(baseagent,generativeEnv,MBPOAgentOpts);
agent.SampleTime = 0.20; % new 1.0

% run 2000 episodes
% run each eposode max. 1000 time steps
% save agent when average reward greater than 500 
% stop training when agent recieve cumulative reward greater than 600 over 20 consecutive episodes

trainOpts = rlTrainingOptions(...
    MaxEpisodes=2000, ...
    MaxStepsPerEpisode=950, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=850,...
    ScoreAveragingWindowLength=150,...
    SaveAgentCriteria="EpisodeReward",...
    SaveAgentValue=850,...
    ScoreAveragingWindowLength=150); 

%plot(env)

% good result:
% stop at 600 with length of 25
% sample time = 0.2
% LearnRate= 4e-5

doTraining = false;

%if doTraining
%    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
%else
%    % Load the pretrained agent for the example.
%    load("MATLABCartpoleMBPO_1.mat","agent");
%end
% save("MATLABCartpoleMBPO_1.mat","agent");
% 
% rng(1)
% agent.UseExplorationPolicy = false; % disable exploration during sim
% simOptions = rlSimulationOptions(MaxSteps=500);
% experience = sim(env,agent,simOptions);
% totalReward_MBPO = sum(experience.Reward)
% 
% rng(1)
% experience = sim(env,agent.BaseAgent,simOptions);
%
% totalReward_SAC = sum(experience.Reward)
% 
% 
% rng(1)
% agent.UseExplorationPolicy = true; % enable exploration during sim to create diverse data for model evaluation
% simOptions = rlSimulationOptions(MaxSteps=500);
% experience = sim(env,agent,simOptions);
% 
% agent.EnvModel.TransitionModelNum = 1;
% 
% numSteps = length(experience.Reward.Data);
% nextObsPrediction = zeros(4,1,numSteps);
% rewardPrediction = zeros(1,numSteps);
% isdonePrediction = zeros(1,numSteps);
% nextObsGroundTruth = zeros(4,1,numSteps);
% rewardGroundTruth = zeros(1,numSteps);
% isdoneGroundTruth = zeros(1,numSteps);
% for stepCt = 1:numSteps
%     % Extract the actual next observation, reward, and is-done value.
%     nextObsGroundTruth(:,:,stepCt) = ...
%         experience.Observation.CartPoleStates.Data(:,:,stepCt+1);
%     rewardGroundTruth(:, stepCt) = experience.Reward.Data(stepCt);
%     isdoneGroundTruth(:, stepCt) = experience.IsDone.Data(stepCt);
% 
%     % Predict the next observation, reward, and is-done value using the
%     % environment model.
%     obs = experience.Observation.CartPoleStates.Data(:,:,stepCt);
%     agent.EnvModel.Observation = {obs};
%     action = experience.Action.CartPoleAction.Data(:,:,stepCt);
%     [nextObs,reward,isdone] = step(agent.EnvModel,{action});
% 
%     nextObsPrediction(:,:,stepCt) = nextObs{1};
%     rewardPrediction(:,stepCt) = reward;
%     isdonePrediction(:,stepCt) = isdone;
% end
% 
% figure
% for obsDimensionIndex = 1:4
%     subplot(2,2,obsDimensionIndex)
%     plot(reshape(nextObsGroundTruth(obsDimensionIndex,:,:),1,numSteps))
%     hold on
%     plot(reshape(nextObsPrediction(obsDimensionIndex,:,:),1,numSteps))
%     hold off
%     xlabel('Step')
%     ylabel('Observation')   
%     if obsDimensionIndex == 1
%         legend('GroundTruth','Prediction','Location','southwest')
%     end
% end
% 
 
% reset function: at a reset of the environment a new start 
% condition is calculated
function in = localResetFcn(in)

% thetastart = 0.1;%= pi - 1.0; %- 0.05*(rand(1)-0.5); % pi +- 2.8 grad
% xstart = 0 - 0.2*(rand(1)-0.5); %  0 +- 0.1 m

% blk = 'simulink_model_mbpo/S-Funktion/xstart';
% in = setBlockParameter(in, blk, 'InitialCondition', num2str(xstart));
    
% blk = 'simulink_model_mbpo/S-Funktion/thetastart';
% in = setBlockParameter(in, blk, 'InitialCondition', num2str(thetastart));
 
    % Randomize reference signal
    blk = sprintf('simulink_model_mbpo/IP/x0');
    h = 0;% 0.2*(rand(1)-0.5);                  % +/- 0.1
    in = setBlockParameter(in,blk,'Value',num2str(h));

    % Randomize reference signal
    blk = sprintf('simulink_model_mbpo/IP/theta0');
    
    h = 0;%0.01*(rand(1)-0.5);             % +/- 1.4 grad
    in = setBlockParameter(in,blk,'Value',num2str(h));

end
