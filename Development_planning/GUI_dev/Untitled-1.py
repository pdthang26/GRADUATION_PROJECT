from math import *

def calulation_next_point(start_point_X,start_point_Y,distance_turn, angle_turn):
    x = start_point_X
    y = start_point_Y
    x_turn = sin(radians(-angle_turn)) * distance_turn 
    y_turn = cos(radians(-angle_turn)) * distance_turn 
    x_new =  (x_turn+x)
    y_new =  (y_turn+y)
    return x_new,y_new

x_0=y_0 = 0
b_p=0

while True:
    b = float(input('Nhập số b:'))
    c = float(input('Nhập goc c:'))
    # for i in range(len(a)):
    #     if b== a[i]:
    #         x_0 = x 
    #         y_0 = y11
    #         b_p = b
    x_n,y_n = calulation_next_point(x_0,y_0,b-b_p,c)
    x_0,y_0 = x_n,y_n
    b_p = b
    # x = b + 1
    # y = b + 2

    print(x_n)
    print(y_n)
    # print(b_p)

    