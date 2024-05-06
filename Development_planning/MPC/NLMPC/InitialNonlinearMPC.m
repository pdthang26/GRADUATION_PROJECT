function [nlobj,ekf] = InitialNonlinearMPC(x0,Ts)
nlobj = nlmpc(7,3,'MV',[1 2],'MD',3,'UD',4);


% Specify the controller sample time, prediction horizon, and control
% horizon. 

nlobj.Ts = Ts;
nlobj.PredictionHorizon = 10;
nlobj.ControlHorizon = 2;

 
% Specify the state function for the nonlinear plant model and its
% Jacobian.
nlobj.Model.StateFcn = @(x,u) LaneFollowingStateFcn(x,u);
nlobj.Jacobian.StateFcn = @(x,u) LaneFollowingStateJacFcn(x,u);


% Specify the output function for the nonlinear plant model and its
% Jacobian. The output variables are:
%
% * Longitudinal velocity
% * Lateral deviation
% * Sum of the yaw angle and yaw angle output disturbance
%
nlobj.Model.OutputFcn = @(x,u) [x(3);x(5);x(6)+x(7)];
nlobj.Jacobian.OutputFcn = @(x,u) [0 0 1 0 0 0 0;0 0 0 0 1 0 0;0 0 0 0 0 1 1];


% Set the constraints for manipulated variables.
nlobj.MV(1).Min = -3;      % Maximum acceleration 3 m/s^2
nlobj.MV(1).Max = 3;       % Minimum acceleration -3 m/s^2
nlobj.MV(2).Min = -1.13;   % Minimum steering angle -65 
nlobj.MV(2).Max = 1.13;    % Maximum steering angle 65


% Set the scale factors.
nlobj.OV(1).ScaleFactor = 15;   % Typical value of longitudinal velocity
nlobj.OV(2).ScaleFactor = 0.5;  % Range for lateral deviation
nlobj.OV(3).ScaleFactor = 0.5;  % Range for relative yaw angle
nlobj.MV(1).ScaleFactor = 6;    % Range of steering angle
nlobj.MV(2).ScaleFactor = 2.26; % Range of acceleration
nlobj.MD(1).ScaleFactor = 0.2;  % Range of Curvature


% Specify the weights in the standard MPC cost function. The third output,
% yaw angle, is allowed to float because there are only two manipulated
% variables to make it a square system. In this example, there is no
% steady-state error in the yaw angle as long as the second output, lateral
% deviation, reaches 0 at steady state.
nlobj.Weights.OutputVariables = [1 1 0];


% Penalize acceleration change more for smooth driving experience.
nlobj.Weights.ManipulatedVariablesRate = [0.3 0.1];


% Định nghĩa hàm trạng thái phi tuyến của bạn và hàm đo lường
stateTransitionFcn = @(x,u) LaneFollowingEKFStateFcn(x,u);
measurementFcn = @(x) LaneFollowingEKFMeasFcn(x);

% Create extended kalman filter for estimated state for NLMPC
ekf = extendedKalmanFilter(stateTransitionFcn,measurementFcn);

% Initial state value 
ekf.State = x0;
end

