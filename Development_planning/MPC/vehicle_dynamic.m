m = 1575;   % Total vehicle mass (kg)
Iz = 2875;  % Yaw moment of inertia of the vehicle (mNs^2)
lf = 1.2;   % Longitudinal distance from the center of gravity to the front tires (m)
lr = 1.6;   % Longitudinal distance from center of gravity to the rear tires (m)
Cf = 19000; % Cornering stiffness of the front tires (N/rad)
Cr = 33000; % Cornering stiffness of the rear tires (N/rad)

% Uncomment the below code and set the longitudinal velocity to a constant value of 15 m/s 
Vx = 15;

% Uncomment the below code and specify the state-space matrices A, B, C and
% D shown above. Create a state-space model (vehicle) of the lateral
% vehicle dynamics using the specified matrices.
A = [-(2*Cf + 2*Cr)/(m*Vx)        0  -Vx-(2*Cf*lf - 2*Cr*lr)/(m*Vx)     0
              0                   0                 1                   0
    -(2*Cf*lf-2*Cr*lr)/(Iz*Vx)    0  -(2*Cf*lf^2 + 2*Cr*lr^2)/(Iz*Vx)   0
              1                   Vx               0                    0];
B = [2*Cf/m
       0    
   2*Cf*lf/Iz
       0  ];
C = [ 0 0 0 1
      0 1 0 0];
D = 0;

vehicle = ss(A,B,C,D);
load('reference.mat')