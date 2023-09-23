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
% s-function for simulink simulaton of non-linear model
% --------------------------------------

function [sys, x0, str, ts] = invertedPendelnonlinear_sfcn(t, x, u, flag, x10, x20, x30, x40 )

switch flag

    case 0
        str = []                ;
        ts = [0 0]              ;
        s = simsizes            ;
            s.NumContStates = 4;
            s.NumDiscStates = 0;
            s.NumOutputs = 4;
            s.NumInputs = 2;
            s.DirFeedthrough = 0;
            s.NumSampleTimes = 1;

        sys = simsizes(s);
        x0 = [x10, x20, x30, x40];

    case 1
        
        sys = invertedPendelnonlinear(t,x,u);

    case 3
      
        sys = x;     % angular position

    case {2 4 9}

        sys = [];
    otherwise
        error(['unhanled flag =', num2str(flag)]);
end