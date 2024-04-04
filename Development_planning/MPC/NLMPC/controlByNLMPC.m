function mv = controlByNLMPC(nlmpcObject,ekfObject,x,lastmv,unMeasValue,curvature,Ts,reference)
% States x = [  lateral velocity (Vy)
%               yaw rate (psi_dot)
%               longitudinal velocity (Vx)
%               longitudinal acceleration (Vx_dot)
%               lateral deviation (e1)
%               relative yaw angle (e2)
%               output disturbance of relative yaw angle (xOD)];


% Outputs:
%           y: Output vector - [Vx e1 e2+x_od]
y = [x0(3);x0(5);x0(6)+x0(7)];

% Correct previous prediction
xk = correct(ekfObject,y);

% Compute optimal control moves
% mv = [  acceleration
%         steering angle]
mv = nlmpcmove(nlmpcObject,xk,lastmv,reference,curvature*x(3));

% input for Extend Kalman Filter
% Inputs uk = [  acceleration
%               steering angle
%               road curvature * Vx (measured disturbance)
%               white noise (unmeasured disturbance)
%               sample time ];
uk = [mv ;curvature*x(3); unMeasValue; Ts];
% Predict prediction model states for the next iteration
predict(ekfObject,uk);
end

