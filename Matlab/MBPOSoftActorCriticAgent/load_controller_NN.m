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
% loads an already trained and stored controller from file
% --------------------------------------
load ('nncontroller.mat', 'net');
load("MATLABCartpoleMBPO_1.mat","agent");