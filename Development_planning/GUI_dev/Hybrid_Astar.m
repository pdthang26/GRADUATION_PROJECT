
function result= Hybrid_Astar(costVal, startPose, goalPose,Interpolation_Dis)
    size_map = size(costVal);
    map = binaryOccupancyMap(costVal);
    %show(map)
    inflatemap = copy(map);
    inflate(inflatemap,10);
    %show(inflatemap)
    validator = validatorOccupancyMap;
    validator.Map = inflatemap;
    validator.StateSpace.StateBounds = [0,size_map(2); 0,size_map(1); -pi*2,pi*2];
    planner = plannerHybridAStar(validator, ...
                                'MinTurningRadius',35, ...
                                'MotionPrimitiveLength',10, ...
                                'NumMotionPrimitives',7 , ...
                                "DirectionSwitchingCost", 30, ...
                                "ReverseCost",30, ...
                                "InterpolationDistance",Interpolation_Dis, ...
                                "ForwardCost",1, ...
                                "AnalyticExpansionInterval",1 );

    [refpath, direction] = plan(planner,startPose,goalPose);
    path = refpath.States;
    dir = direction;
    result = [path, dir];
    %show(planner)
end
