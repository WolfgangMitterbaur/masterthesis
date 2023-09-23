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
% LQR control design for Inverted Pendulum
% --------------------------------------

%% LQR control design 
% calculation of LQR gain

Q = diag([1200 1500 0 0]);
R  = 0.035;
KK = lqr(A,B,Q,R);

%% Close Loop Control simulation
%
% note: this is only used for the development process
%
% Ts = 0.01; % sample time for simulation 
% Tf = 10;   % simulation end time
%
% X0    = [0.2; 160*(pi/180); 0;0]; % initial state/ take the (Initial Value of theta) > 0 {MUST}
%                                   % such as angle will always measured from vertical downward axis(from 4th quadrant) 
% X_des = [0; pi; 0; 0];            % desired state
% f0    = 0;                        % start force
% i = 0;
% 
% for k = 0:Ts:Tf
%     i = i+1;
%     % solve the equation of movement
%     new_state = RK4_2nd_order(X0, Ts, f0, M, m, g, l, c, b, I);
%     Xp(i,:) = new_state'; % for plot 
%     t(i)    = k;
%     X0      = new_state;
%     
%     % LQR control Design 
%     f0 = -KK*(X0-X_des);
%   
% end

%% Animation plot of Inverted Pedulum System
%
% note: this is only used for the development process
%
% figure()
% for i = 1:12:length(Xp)
%    IP_Animation(Xp(i,1),Xp(i,2))
%    pause(0.01);
%     % movieVector(i) =  getframe(hf);
%    hold off
% end

%% plot results
%
% note: this is only used for the development process
%
% figure()
% axis(gca,'equal');
% subplot(2,2,1);
% plot(t,Xp(:,1));
% grid on;
% ylabel('X (m)');
% xlabel('time [sec]');
% 
% subplot(2,2,2);
% %plot(t, (180/pi.*Xp(:,2)));
% plot(t, (Xp(:,2)));
% grid on;
% ylabel('\theta (deg)');
% xlabel('time [sec]');
% 
% subplot(2,2,3);
% plot(t,Xp(:,3));
% grid on;
% ylabel('x dot (m/sec)');
% xlabel('time [sec]');
% 
% subplot(2,2,4);
% %plot(t,(180/pi.*Xp(:,4)));
% plot(t,(Xp(:,4)));
% grid on;
% ylabel('\theta dot (deg/sec)');
% xlabel('time [sec]');

