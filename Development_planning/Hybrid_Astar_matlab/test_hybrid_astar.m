close all; clear all;

x = load("map_test.csv");
x(200-146,90) = 1;
tic
startPose = [100 100 pi/2];
goalPose  = [79 199 1.5707963267948966];

% x = gpuArray(x);
result = Hybrid_Astar(x, startPose, goalPose,10);
toc

