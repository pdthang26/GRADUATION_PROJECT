
# '''điều khiển vận tốc'''

# #LOW speed if distance <= 10 m
# Max_Low_Speed = 60
# def adjust_low_speed():
#     speed = Max_Low_Speed
#     return speed

# # High speed if distance > 10 m
# Max_High_Speed = 100
# Satisfy_High_Speed = 20
# def adjust_high_speed(desired_dis,actual_dis):
#     actual_dis_p=0

#     if (actual_dis>=0) and (actual_dis<2):    #increase speed in first 2m
#         speed = int((actual_dis-actual_dis_p)*(Max_High_Speed-Satisfy_High_Speed)/2)+Satisfy_High_Speed
#         if speed>Max_High_Speed:
#             speed = Max_High_Speed
#         actual_dis_p = actual_dis 

#     elif (actual_dis>=2) and (desired_dis-actual_dis)>2:  # max speed while moving
#         speed = Max_High_Speed

#     elif (desired_dis-actual_dis)>=0 and (desired_dis-actual_dis)<=2: # decrease speed in 2m
#         speed = int((desired_dis-actual_dis)*Max_High_Speed/2)
    
#     return speed


# # Điều khiển bánh sau
# def back_wheel_control(desired,actual):
     
#     actual_p = 0
    
#     # actual = actual_rev
#     destiantion_p=0
#     destination = destiantion_p + desired
#     print(destination)
    
#     direction = b'T'
#     if  destination < 0:
#         direction = b'L'
      

#     if actual <= destination:
#         if desired >=0 and desired<=10:
#             speed = adjust_low_speed()
#         else:
#             speed = adjust_high_speed(desired,actual-actual_p)
        
#         actual_p = actual
#         print(actual_p)

#     destiantion_p = destination  
#     print(destiantion_p)

#     return speed

# destination = 50
# destination_p = 30
# desired = destination-destination_p
# pwm = back_wheel_control(desired,30)
# print(pwm)

# # biến cờ thứ tự di chuyển auto
# flag_mov = 0

# # hàm điều khiển đi bánh sau auto cho đi thẳng
# a=0
# def straight_control():
#     global flag_mov,a
     
#     distance_value = float(distance[1:].replace('\x00', ''))
#     direction = b'T'
    
#     if linear_entry.get() != '':
#         desired_straight_pos = float(linear_entry.get())
#         if  desired_straight_pos < 0:
#             direction = b'L'
    
#     if turn_entry.get() !='':
#         run = 1
#     else:
#         run = 0

#     if distance_value < desired_straight_pos:
#         flag = 1
#         flag_mov = 1
#         vel = 80
#         a= 10000
#     else: 
#         a= 10000
#         flag = 0
#         vel = 0
#         if run==1:
#             flag_mov = 2
#         else:
#            flag_mov = 5
           

#     #Gửi cho bánh sau
#     flag_transmit = str(flag)
#     uart_data_0 = b_start_bit + direction + flag_transmit.encode('utf-8') + stop_bit
#     b_uart.write(uart_data_0)

#     #Gửi cho bánh trước
#     b = str(a)
#     uart_data_1 = f_start_bit + chr(vel).encode('utf-8') + b.encode('utf-8') + stop_bit
#     f_uart.write(uart_data_1)


# MAX_left_pulse = 20000 # bánh đánh hết sang bên trái
# STRAIGHT_pulse = 10000 # bánh đánh thẳng
# MIN_right_pulse = 0 # bánh đánh hết sang phải
# Max_steering = 38 # góc quay tối đa qua một bên

# def adjust_pulse(desired,actual):
    
#     # điều xung cho quay bên trái
#     if desired - actual > 45:
#         pulse = MAX_left_pulse
#     elif (desired - actual)>=0 and (desired - actual)<= 45:
#         pulse = STRAIGHT_pulse

#     # điều xung cho quay bên phải
#     if desired - actual <-45:
#         pulse = MIN_right_pulse
#     elif (desired - actual)>=-45 and (desired - actual)<=0:
#         pulse = STRAIGHT_pulse

#     return pulse

# MAX_steering_speed = 100
# Min_teering_speed = 50

# def adjust_speed(desired,actual):

#     # khi góc quay tính ra lớn hơn góc tối đa mỗi bên 
#     if abs(desired-actual)>45:
#         speed = MAX_steering_speed
#     elif abs(desired - actual) >=0 and abs(desired-actual)<=45:
#         speed = Min_teering_speed

#     return speed

# def steering_control():
     
#     real_angle_value = float(angle[1:].replace('\x00',''))

#     if turn_entry.get()=='':
#         desired_angle = 0
#     elif int(turn_entry.get()) < 0:
#         desired_angle = -90
#     elif int(turn_entry.get()) > 0:
#         desired_angle = 90
   
#     pulse = str(adjust_pulse(desired_angle,real_angle_value))
#     speed = chr(adjust_speed(desired_angle,real_angle_value))

#     uart_data = f_start_bit + speed.encode('utf-8') + pulse.encode('utf-8') + stop_bit

#     f_uart.write(uart_data)

# # hàm điều khiển đi bảnh sau auto cho đi quẹp
# def turn_control():
#     global flag_mov

#     if turn_entry.get !='':
#         desired_pos = abs(float(turn_entry.get()))
    
#     direction = b'T'
#     distance_value = float(distance[1:].replace('\x00', ''))

#     if distance_value < desired_pos:
#         flag = 1
#         flag_mov = 2
#     else:
#         flag = 0
#         flag_mov = 0

#     flag_transmit = str(flag)
#     uart_data = b_start_bit + direction + flag_transmit.encode('utf-8') + stop_bit
#     b_uart.write(uart_data)


a = [5,5,5]
b = [0, -90, 90]
c = int(input('Nhập c:'))
d = 0  # Biến tạm để tính tổng các phần tử
e = sum(a)  # Chuyển đổi các phần tử của a thành số nguyên và tính tổng
f = 15
g = f + e
flag = False


if c <= g:
    for i in range(len(a)):
        for j in range(len(b)):
            d += int(a[i])  # Chuyển đổi a[i] thành số nguyên trước khi cộng vào d
            if c ==d:
                if j+1<len(b):
                    print('j =', j+1, 'b =', b[j+1])
else:
    f = e
    print("quãng đường đã đi được:",f)
