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
% create reward network action next obs
% --------------------------------------

function rewardNetwork = createRewardNetworkActionNextObs(numObservations, numActions)
    actionPath = featureInputLayer(numActions,Normalization="none",Name="action");
    nextStatePath = featureInputLayer(numObservations,Normalization="none",Name="nextState");
    commonPath = [concatenationLayer(1,2,Name="concat")
        fullyConnectedLayer(64,Name="FC1")
        reluLayer(Name="CriticRelu1")
        fullyConnectedLayer(64, "Name","FC2")
        reluLayer(Name="CriticCommonRelu2")
        fullyConnectedLayer(64, "Name","FC3")
        reluLayer(Name="CriticCommonRelu3")
        fullyConnectedLayer(1,Name="reward")];

    rewardNetwork = layerGraph(nextStatePath);
    rewardNetwork = addLayers(rewardNetwork,actionPath);
    rewardNetwork = addLayers(rewardNetwork,commonPath);

    rewardNetwork = connectLayers(rewardNetwork,"nextState","concat/in1");
    rewardNetwork = connectLayers(rewardNetwork,"action","concat/in2");

    rewardNetwork = dlnetwork(rewardNetwork);
end
