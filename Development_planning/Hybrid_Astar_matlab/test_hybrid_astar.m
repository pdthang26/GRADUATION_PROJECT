close all; clear all;

x = load("map2.csv");

startPose = [20 22.5 pi/2];
goalPose  = [30 35 pi/2];
result = Hybrid_Astar(x, startPose, goalPose);

tic;
elapsed_time = toc;  % Lấy thời gian đã trôi qua
fprintf('Elapsed time: %.4f seconds\n', elapsed_time);
