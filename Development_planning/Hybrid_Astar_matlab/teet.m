close all; clear all;

x = load("map_test.csv");
x(200-146,90) = 1;
tic
startPose = [100 100 pi/2];
goalPose  = [79 199 1.5707963267948966];

% x = gpuArray(x);
size_map = size(x);
map = binaryOccupancyMap(x);
%show(map)
inflatemap = copy(map);
inflate(inflatemap,10);
%show(inflatemap)
validator = validatorOccupancyMap;
validator.Map = inflatemap;
% validator.ValidationDistance = 1;
validator.StateSpace.StateBounds = [0,size_map(2); 0,size_map(1); -pi*2,pi*2];
planner = plannerHybridAStar(validator, ...
                            'MinTurningRadius',35, ...
                            'MotionPrimitiveLength',10, ...
                            'NumMotionPrimitives',5 , ...
                            "DirectionSwitchingCost", 30, ...
                            "ReverseCost",30, ...
                            "InterpolationDistance",10, ...
                            "ForwardCost",1, ...
                            "AnalyticExpansionInterval",1 );

[refpath, direction] = plan(planner,startPose,goalPose);
path = refpath.States;
dir = direction;
result = [path, dir];
show(planner)
toc