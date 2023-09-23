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
% Function to initialize of DLS(toLayer, fromLayer) with the length of delayVector
% and ldels(toLayer, fromLayer) with delayVector
% --------------------------------------

function [DL, ldel] = setDL(toLayer, fromLayer, delayVector, DL, ldel)
%function [DL, ldel] = setDL(toLayer, fromLayer, delayVector, DL, ldel)

delayVector = sort(delayVector);

DL(toLayer, fromLayer) = length(delayVector);

for i = 1:DL(toLayer, fromLayer)
    ldel(toLayer, fromLayer, i) = delayVector(i);
end