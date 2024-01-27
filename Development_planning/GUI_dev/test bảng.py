# # 
# import time

# a=(0,0)
# x=[a] 
# print(x)
# time.sleep(2)
# x = [a,(1,1),(2,2),(3,3),(4,4),(5,5)]
# print(x)
# time.sleep(2)
# a=x[len(x)-1]
# x=[a]
# print(a)
# print(x)

# a = 30
# a_p = 30
# b = 40
# c = 0
# while a<=b:
#     if a >= a_p and a <= a_p + 2:
#         c = 10
#     elif a > a_p+2 and a<b:
#         c = 20
#     else:
#         c = 0
#         a_p = a

#     print("a =", a, "c =", c)
#     a += 1
#     time.sleep(1)
# print(a_p)

# import math

# def move_car(points, angles):
#     current_poin = (0, 0)  # Điểm ban đầu
#     total_distance = 0

#     for i in range(len(points)):
#         next_point = points[i]
#         angle = angles[i]

#         distance = math.sqrt((next_point[0] - current_point[0]) ** 2 + (next_point[1] - current_point[1]) ** 2)
#         total_distance += distance

#         print(f"Di chuyển từ {current_point} đến {next_point}, quay {angle} độ, quãng đường: {distance}m")

#         current_point = next_point

#         if i < len(points) - 1:
#             print(f"Xe quay {angle} độ")

#     print(f"Tổng chiều dài đường đi: {total_distance}m")

# Các điểm và góc tương ứng
# distance = [5, 5, 5]
# angles = [0, 90, -90]

# move_car(points, angles)

a = [10,15.6,25.6] # length increment
b= [0,-45,90] # change angle
c = [0,-45,45] # deisred angle

d= 0 # actual_p
e = 2 # actual

f = 0 # desired_p
g = 25.6 # desired
h= g+f # total_length

speed = 0 # speed

desired_ang = 0
change_ang = 0
act_ang = -0.3


'''điều xung theo góc cho bánh trước'''
MAX_left_pulse = 20000 # bánh đánh hết sang bên trái
STRAIGHT_pulse = 10000 # bánh đánh thẳng
MIN_right_pulse = 0 # bánh đánh hết sang phải
Max_steering = 38 # góc quay tối đa qua một bên

def adjust_front_pulse(desired,actual,change):
    sub = desired - actual
    pulse= pulse_straight = 10000
    pulse_desired = int((change/38)*10000)+10000

    if pulse_desired>MAX_left_pulse:
        pulse_desired = MAX_left_pulse
    elif pulse_desired<0:
        pulse_desired = MIN_right_pulse

    #Hệ số trả góc
    coef = 0.5
    # điều xung cho quay bên trái
    if sub>= 0 and sub <= abs(change)*coef:
        if sub> 1:
            pulse  = int((sub/38)*10000)+10000
            return pulse
        else:
            pulse = pulse_straight
            return pulse
    elif sub > 1 and sub> abs(change)*coef:
        pulse = pulse_desired
        return pulse

    # điều xung cho quay bên phải
    if sub>= -abs(change)*coef and sub <= 0:
        if sub<-1:
            pulse  = int((sub/38)*10000)+10000
            return pulse
        else:
            pulse = pulse_straight
            return pulse
    elif sub<-1 and sub<-abs(change)*coef:
        pulse = pulse_desired
        return pulse

    return pulse
'''----oooo----'''


if e<g:
    step= 0
    for i in range(len(a)):
        if e<a[i]:
            step = i
            break
    desired_ang = c[step]
    change_ang = b[step]
               
    print ("góc quay mỗi điểm:",change_ang)
    print ('góc mong muốn:',desired_ang)

    if e>=d and e<=d+2:
        speed = 75
    if e> d+2 and e<=h-1:
        if act_ang-desired_ang>=-1 and act_ang-desired_ang<=1:
            speed = 30
        else:
            speed = 50
    print('tốc độ:',speed)
    print('tổng trước:',f)
    print('thực tế trước:',d)
else:
    speed = 0
    f = h
    d = e
    print('tốc độ:',speed)
    print('tổng trước:',f)
    print('thực tế trước:',d)

pwm = adjust_front_pulse(desired_ang,act_ang,change_ang)
print('xung bánh trước:',pwm)

# '''điều xung theo góc cho bánh trước'''
# MAX_left_pulse = 20000 # bánh đánh hết sang bên trái
# STRAIGHT_pulse = 10000 # bánh đánh thẳng
# MIN_right_pulse = 0 # bánh đánh hết sang phải

# def adjust_front_pulse(desired,actual,change):
#     sub = desired - actual
#     pulse_straight = 10000
#     pulse_desired = int((change/38)*10000)+10000

#     if pulse_desired>MAX_left_pulse:
#         pulse_desired = MAX_left_pulse
#     elif pulse_desired<0:
#         pulse_desired = MIN_right_pulse

#     # điều xung cho quay bên trái
#     if sub>= 0 and sub <= change/2:
#         pulse = pulse_straight
#         return pulse
#     elif sub > 2 and sub>change/2:
#         pulse = pulse_desired
#         return pulse

#     # điều xung cho quay bên phải
#     if sub>= abs() and sub <= 0:
#         pulse = pulse_straight
#         return pulse
#     elif sub<-2:
#         pulse = MIN_right_pulse
#         return pulse

#     return pulse

# a = 20 #change
# b = -20 #desired
# c = 0 #actual

# pulse = int(adjust_front_pulse(b,c,a))

# print(f'xung cần quay góc {a} là {pulse}')


# '''điều xung theo góc cho bánh trước'''
# MAX_left_pulse = 20000 # bánh đánh hết sang bên trái
# STRAIGHT_pulse = 10000 # bánh đánh thẳng
# MIN_right_pulse = 0 # bánh đánh hết sang phải
# Max_steering = 38 # góc quay tối đa qua một bên

# def adjust_front_pulse(desired,actual,change):
#     sub = desired - actual
#     pulse= pulse_straight = 10000
#     pulse_desired = int((change/38)*10000)+10000

#     if pulse_desired>MAX_left_pulse:
#         pulse_desired = MAX_left_pulse
#     elif pulse_desired<0:
#         pulse_desired = MIN_right_pulse

#     #Hệ số trả góc
#     coef = 0.5
#     # điều xung cho quay bên trái
#     if sub>= 0 and sub <= abs(change)*coef:
#         if sub> 1:
#             pulse  = int((sub/38)*10000)+10000
#             return pulse
#         else:
#             pulse = pulse_straight
#             return pulse
#     elif sub > 2 and sub> abs(change)*coef:
#         pulse = pulse_desired
#         return pulse

#     # điều xung cho quay bên phải
#     if sub>= -abs(change)*coef and sub <= 0:
#         if sub<-1:
#             pulse  = int((sub/38)*10000)+10000
#             return pulse
#         else:
#             pulse = pulse_straight
#             return pulse
#     elif sub<-2 and sub<-abs(change)*coef:
#         pulse = pulse_desired
#         return pulse

#     return pulse
# '''----oooo----'''

# '''điều tốc độ quay motor bánh trước'''
# MAX_steering_speed = 50
# Min_steering_speed = 35

# def adjust_front_speed(desired,actual):
#     sub = abs(desired-actual)
#     # khi góc quay tính ra lớn hơn góc tối đa mỗi bên 
#     if sub>10:
#         speed = MAX_steering_speed
#     elif sub>=0 and sub<=10:
#         speed = Min_steering_speed
#     return speed
# '''------oooo-----'''