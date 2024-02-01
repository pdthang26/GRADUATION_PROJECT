import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
import serial.tools.list_ports
from PIL import ImageTk, Image
import serial
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import numpy as np
import pandas as pd
from math import *
import pandas as pd
import matlab.engine
import cv2
from numba import jit
import time

# Khởi động matlab engine
eng = matlab.engine.start_matlab()
print('MALAB ENGINE FINISHED BEGINNING !!!')

'''-----------------------------------------------------------------
--------------------------------------------------------------------
------------------KHAI BÁO NHỮNG ĐƯỜNG DẪN--------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

# Đường dẫn tương đối của file

''' ảnh của nút tiến'''
back_for_release_path = '1.png'
back_for_press_path = '2.png'

''' ảnh của nút lùi'''
back_rev_release_path = '3.png'
back_rev_press_path = '4.png'

'''ảnh của nút phanh'''
brake_release_path = 'b_1.png'
brake_press_path = 'b_2.png'

'''ảnh nút Emergency stop'''
emergency_stop_path = 'emergency stop.png'

'''ảnh tick xanh thông báo UART kết nối thành công'''
successful_path = 'successful.png'

# Xác định đường dẫn tuyệt đối
back_for_release = os.path.abspath(back_for_release_path)
back_for_press = os.path.abspath(back_for_press_path)

back_rev_release = os.path.abspath(back_rev_release_path)
back_rev_press = os.path.abspath(back_rev_press_path)

brake_release = os.path.abspath(brake_release_path)
brake_press = os.path.abspath(brake_press_path)

emergency_stop = os.path.abspath(emergency_stop_path)

successful = os.path.abspath(successful_path)


'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

'''-----------------------------------------------------------------
--------------------------------------------------------------------
-----------------------DECLARATION VARIABLE-------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

'''Tạo các biến cần thiết cho chương trình'''

# biến màu giao diện
GUI_color = '#C9F4AA'

# biến màu frame Manual
manu_color = '#F7C8E0'

# Bit bắt đầu cho gửi UART bánh sau
b_start_bit = b'B'

# Start bit for UART transmission on the front wheel
f_start_bit = b'F'

# Bit kết thúc
stop_bit = b'\x0A'

# Tạo cửa sổ giao diện chính
root = tk.Tk()
root.geometry("550x710")
root.configure(bg=GUI_color)
root.resizable(height=False, width=False)

# Tạo mảng bao gồm các thành phần trên giao diện
objects_1 = [] # mảng chứa các thành phần để active bằng nút manual
objects_2 = [] # mảng chứa các elements để active bằng nút Connect
objects_3 = [] # các element combobox về UART parameter
objects_4 = [] # mảng để chứa elements được active bằng nút auto

# Các biến dùng truyền UART
anglar_vel_uart = emer_uart= b_uart= f_uart= p_uart= ang_uart= vel_uart= dis_uart = None
ultra_uart=gps_port = None

'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

'''-----------------------------------------------------------------
--------------------------------------------------------------------
------------------------- MAIN GUI ----------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

''' Chức năng giao diện '''

# Hàm cho nút connect
def connect_uart():

    #Kích hoạt nút Disconnect
    disconnect_button['state']='normal'

    # kích hoạt nút Manual, Auto, Show Car Value
    for obj in objects_2:
        obj['state'] = 'normal'

    global b_uart,f_uart,p_uart,ultra_uart
    global ang_uart,vel_uart,dis_uart, anglar_vel_uart 
    global emer_uart
    global gps_uart

    # Các biến parameter cho UART
    selected_port = com_port.get()
    selected_gps = gps_port.get()
    selected_rate = int(rate_port.get())
    selected_stop = stop_port.get()
    select_data = int(data_port.get())
    select_parity = parity_port.get()
    time = 1
        
    # tạo một switch case check coi stop bit chọn bao nhiêu bit
    def switch_case_1(argument):
        if argument == '1':
                output = serial.STOPBITS_ONE
        elif argument == '1.5':
                output = serial.STOPBITS_ONE_POINT_FIVE
        elif argument == '2':
                output = serial.STOPBITS_TWO
        return output
    stop_bit_value = switch_case_1(selected_stop)

    # tạo một switch case để check coi data bit truyền là 7 hay 8 bits
    def switch_case_2(argument):
        if argument == 7:
            output = serial.SEVENBITS
        elif argument == 8:
            output = serial.EIGHTBITS
        return output
    data_bit_value = switch_case_2(select_data)

    #tạo một switch case để check coi parity bit được chọn là even, odd hay none
    def switch_case_3(argument):
        if argument == 'even':
            output = serial.PARITY_EVEN
        elif argument =='odd':
            output = serial.PARITY_ODD
        elif argument =='none':
            output =serial.PARITY_NONE
        return output
    parity_bit_value =switch_case_3(select_parity)

    # Kiểm tra nếu cổng UART không được cung cấp
    if (selected_port == '')or(selected_gps ==''):
        messagebox.showwarning('Warning', 'The COM/GPS port is empty.\nPlease select a COM/GPS port.')
    else:
        try:
            # Khởi tạo đối tượng Serial
            anglar_vel_uart = ultra_uart=emer_uart= ang_uart= vel_uart= dis_uart= b_uart= f_uart= p_uart =serial.Serial(
            port=selected_port,
            baudrate=selected_rate,
            stopbits=stop_bit_value,
            bytesize=data_bit_value,
            parity=parity_bit_value,
            timeout=time  # Timeout cho phép đọc từ giao diện UART
        )
             #khởi tạo đối tượng Serial
            gps_uart = serial.Serial(
            port=selected_gps,
            baudrate=9600,
            stopbits= stop_bit_value,
            bytesize= data_bit_value,
            parity= parity_bit_value,
            timeout=1
        )
            # Hiển thị thông báo kết nối UART thành công 
            if b_uart.is_open:

                # tạo popup thông báo kết nối thành công 
                popup = tk.Toplevel()
                popup.title('Success')
                popup.resizable(height=False,width=False)
                
                # Tính toán vị trí của popup
                root_width = root.winfo_width()
                root_height = root.winfo_height()
                popup_width = 320
                popup_height = 120
                x = root.winfo_rootx() + (root_width - popup_width) // 2
                y = root.winfo_rooty() + (root_height - popup_height) // 2
                popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
                
                canvas = tk.Canvas(popup, width=100, height=100,bd=0)
                canvas.place(x=1,y=1)

                image = Image.open(successful)
                image = image.resize((50, 50))
                photo = ImageTk.PhotoImage(image)

                canvas.create_image(40,60,image=photo)
                canvas.place(x= 1, y= 1)

                label = tk.Label(popup, text="UART connection successfull",font=('Arial',12,'bold'))
                label.place(x = 80, y= 30)

                ok_button = tk.Button(popup, text="OK",font=('Arial',11,'bold'), bg='white',command=popup.destroy)
                ok_button.place(x = 150, y = 70, width= 50, height= 30)

                popup.mainloop()

        except serial.SerialException as e:
            # Xử lý lỗi mở cổng UART
            messagebox.showerror('Error', f'Failed to open COM port: {str(e)}')

# hàm xử lý show angular velocity
actual_angular_vel=0            
def show_angular_vel():
    global actual_angular_vel
    while(True):
        # Đọc dữ liệu UART về góc
        angular_vel = anglar_vel_uart.readline().decode().strip()
        # Xử lý tín hiệu UART cho góc
        if angular_vel.startswith('A'):
            angular_vel_display['text'] = angular_vel[1:].replace('\x00','') 
            actual_angular_vel = float( angular_vel[1:].replace('\x00',''))  
            
            break
actual_angle = 0
# hàm xử lý show angle
def show_angle():
    global actual_angle
    while(True):
        # Đọc dữ liệu UART về góc
        angle = ang_uart.readline().decode().strip()
        # Xử lý tín hiệu UART cho góc
        if angle.startswith('Y'):
            angle_display['text'] = angle[1:].replace('\x00','') 
            actual_angle = float( angle[1:].replace('\x00',''))  
            
            break

#Hàm xử lý show Velocity
actual_vel = 0
def show_vel():
    global actual_vel
    while(True):
        # Đọc dữ liệu UART về tốc độ
        velocity = vel_uart.readline().decode().strip()
        # Xử lý tín hiệu UART cho tốc độ
        if velocity.startswith('V'):
            vel_display['text'] = velocity[1:].replace('\x00','') 
            actual_vel = float(velocity[1:].replace('\x00',''))
            
            break

# Hàm xử lý show Distance
actual_dis=0
def show_dis():
    global actual_dis
    while(True):
        # Đọc dữ liệu UART về quãng đường
        distance = dis_uart.readline().decode().strip()
        # Xử lý tín hiệu UART cho quãng đường
        if distance.startswith('D'):
            dis_display['text'] = distance[1:].replace('\x00','') 
            actual_dis = float(distance[1:].replace('\x00',''))
            
            break

# Hàm xử lý show GPS
def show_gps():
    try:
        # Đọc dữ liệu UART về GPS
        gps = gps_uart.readline().decode().strip()

        # Xử lý tín hiệu UART cho GPS
        if gps.startswith('$GPRMC'):
            data = gps.split(',')
            if data[2] == 'A':
                latitude_decimal = float(data[3])
                longitude_decimal = float(data[5])

                # Chuyển đổi độ phút giây
                latitude_degrees = int(latitude_decimal / 100)
                latitude_minutes = float((latitude_decimal % 100) / 60)
                latitude = latitude_degrees + latitude_minutes

                longitude_degrees = int(longitude_decimal / 100)
                longitude_minutes = float((longitude_decimal % 100) / 60)
                longitude = longitude_degrees + longitude_minutes

                # Cập nhật giá trị lên các ô label
                longitude_display['text'] = f"{longitude:.8f}"  # Hiển thị đến 6 chữ số thập phân
                latitude_display['text'] = f"{latitude:.8f}"  # Hiển thị đến 6 chữ số thập phân

    except:
        # Xử lý khi UART bị ngắt
        longitude_display['text'] = "0.0"
        latitude_display['text'] = "0.0"

ultra_values = []
new_values = []
# Hàm nhấn nút show value
def show():
    global ultra_values,new_values
    while True:
        show_angle()
        show_dis()
        show_vel()
        show_angular_vel()
        show_gps()
        ultrasonic = ultra_uart.readline().decode().strip()
        if ultrasonic.startswith('U'):
            ultra_data = ultrasonic[1:].replace('\x00','').split(',')
            if len(ultra_data) == 4:
                new_values = [float(value) for value in ultra_data]
                if new_values != ultra_values:
                    ultra_values = new_values

# phân luồng cho nút show 
def show_click():
    threading.Thread(target = show).start()

# Tạo nút Show value
show_button = tk.Button(root,text = 'Show Value',state= 'disabled',bg='white',command=show_click)
show_button.place(x= 460,y=460,height=30,width=80)
objects_2.append(show_button)


#Hàm cho nút Disconnect
def disconnect_uart():
    
    global update_flag

    update_flag = False

    for obj in objects_1+objects_2:
        obj['state'] = 'disabled'

    com_port['text']=''
    gps_port['text']=''

    #Ngắt UART
    if b_uart.is_open:
        b_uart.close()
        f_uart.close()
        p_uart.close()
        ang_uart.close()
        vel_uart.close()
        dis_uart.close()
        anglar_vel_uart.close()

    if not b_uart.is_open:
        messagebox.showerror('Warning','UART is disconnected !')

# Hàm cho nút Open
def open_click():
    #Kích hoạt nút Connect 
    connect_button['state']='normal'
    #Kích hoạt các Combobox 
    for obj in objects_3:
        obj['state']='normal'
    
# Hàm cho nút Close
def close_click():
    
    # Hiện ô thông báo lữa chọn muốn đóng cửa sổ giao diện
    result = messagebox.askyesno("Exit", "Do you want to exit?")
    if result:
        root.destroy()

'''Tạo các nhãn'''

# Tạo nhãn cho ô chọn cổng COM
com_label = tk.Label(root, text="COM Port:", bg=GUI_color)
com_label.place(x=180, y=5)

# Tạo nhãn cho ô chọn GPS COM
gps_label = tk.Label(root, text ='GPS Port', bg=GUI_color)
gps_label.place(x=300,y=5)

# Tạo nhãn cho ô chọn Baudrate
rate_label = tk.Label(root, text='Baud Rate', bg=GUI_color)
rate_label.place(x=420, y=5)

# Tạo nhãn cho Data Bits
data_label = tk.Label(root,text='Data Bit:',bg=GUI_color)
data_label.place(x=180,y=65)

#Tạo nhãn cho Stop Bit
stop_label = tk.Label(root,text='Stop Bit',bg=GUI_color)
stop_label.place(x=300,y=65)

#Tạo nhãn cho Parity bit
parity_label = tk.Label(root,text = 'Parity Bit',bg=GUI_color)
parity_label.place(x=420,y=65)


# Tạo nhãn cho hiển thị Angle
angle_label = tk.Label(root,text='Angle',bg=GUI_color)
angle_label.place(x=10,y= 460)

#Tạo ô hiển thị cho Angle
angle_display = tk.Label(root,relief=tk.SUNKEN,anchor=tk.W,padx=10,bg='white',font=('Arial',13,'bold'))
angle_display.place(x=10,y=485,height=30,width=100)

# Tạo nhãn hiển thị đơn vị góc quay
ang_unit = tk.Label(root,text = '\u00B0',bg=GUI_color,font=('Arial',15,'bold'))
ang_unit.place(x=110,y=485)

#Tạo nhãn cho Distance
dis_label = tk.Label(root,text='Distance',bg=GUI_color)
dis_label.place(x=150,y=460)

#Tạo ô hiển thị cho Distance 
dis_display = tk.Label(root,relief=tk.SUNKEN,anchor=tk.W,padx=10,bg='white',font=('Arial',13,'bold'))
dis_display.place(x=150,y=485,height=30,width=100)

#Tạo nhãn hiển thị đơn vị cho Distance
dis_unit =tk.Label(root,text='m',bg=GUI_color,font=('Arial',13))
dis_unit.place(x=250,y=485)

# Tạo nhãn cho Speed
vel_label = tk.Label(root,text='Speed',bg=GUI_color)
vel_label.place(x=300,y=460)

#Tạo ô hiển thị Speed
vel_display= tk.Label(root,relief=tk.SUNKEN,anchor=tk.W,padx=10,bg='white',font=('Arial',13,'bold'))
vel_display.place(x=300,y=485,width=100,height=30)

# Tạo nhãn đơn vị cho tốc độ
speed_unit = tk.Label(root,text='m/s',bg=GUI_color,font=('Arial',13))
speed_unit.place(x=400,y=485)

#Tạo nhãn hiển thị cho angular velocity
angular_vel_label = tk.Label(root,text='Angular velocity',bg=GUI_color)
angular_vel_label.place(x=10,y= 520)


#Tạo ô hiển thị angular velocity
angular_vel_display= tk.Label(root,relief=tk.SUNKEN,anchor=tk.W,padx=10,bg='white',font=('Arial',13,'bold'))
angular_vel_display.place(x=10,y=545,width=100,height=30)

# Tạo nhãn đơn vị cho angular velocity
angular_vel_unit = tk.Label(root,text='rad/s',bg=GUI_color,font=('Arial',13))
angular_vel_unit.place(x=120,y=545)

# Tạo nhãn cho Longitude
longitude_label= tk.Label(root,text='Longitude',bg = GUI_color)
longitude_label.place(x=10,y=580)

# Tạo ô hiển thị Longitude
longitude_display = tk.Label(root,relief=tk.SUNKEN,padx=5,bg='white',anchor=tk.W)
longitude_display.place(x=10,y=605,height=30,width=170)

# Tạo nhãn hiển thị Latitude
latitude_label = tk.Label(root,text='Latitude',bg=GUI_color)
latitude_label.place(x=10,y=640)

# Tạo ô hiển thị Latitude
latitude_display = tk.Label(root,relief=tk.SUNKEN,bg='white',anchor=tk.W,padx=5)
latitude_display.place(x=10,y=665,height=30,width=170)

# Lấy danh sách tất cả các cổng COM 
com_ports = [port.device for port in serial.tools.list_ports.comports()]

# Lấy danh sách Baud Rate
rate = [
    '1200', '1800', '2400', '4800', '9600', '19200', '28800', '38400',
    '57600', '76800', '115200', '230400', '460800', '576000', '921600']

''' Tạo các combobox cho các thông số liên quan tới UART'''
# Tạo ô chọn cổng COM cho điều khiển
com_port = ttk.Combobox(root, values=com_ports, state='disabled')
com_port.place(x=180, y=30, height=30, width=100)
objects_3.append(com_port)

# Tạo ô chọn cổng COM cho GPS
gps_port = ttk.Combobox(root,values = com_ports,state='disabled')
gps_port.place(x=300,y=30,height=30,width=100)
objects_3.append(gps_port)

# Tạo ô chọn Baud Rate
rate_port = ttk.Combobox(root, values=rate, state='disabled')
rate_port.place(x=420, y=30, height=30, width=100)
rate_port.set(rate[10])
objects_3.append(rate_port)

# Tạo ô chọn Data Bits
data_port  = ttk.Combobox(root,values = ['7','8'],state='disabled')
data_port.set('8')
data_port.place(x=180, y=90, height=30, width=100)
objects_3.append(data_port)

#Tạo ô chọn Stop Bit
stop_port = ttk.Combobox(root,values=['1','1.5','2'],state='disabled')
stop_port.set('1')
stop_port.place(x=300,y=90,height=30,width=100)
objects_3.append(stop_port)

#Tạo ô chọn Parity Bit 
parity_port =ttk.Combobox(root,values=['even','odd','none'],state='disabled')
parity_port.set('none')
parity_port.place(x=420,y=90,height=30,width=100)
objects_3.append(parity_port)

''' Tạo các nút '''
# Tạo nút Open
btn_open = tk.Button(root, text='Open', command=open_click,bg='white')
btn_open.place(x=10, y=30, height=30, width=70)

# Tạo nút Close cửa sổ chương trình
btn_close = tk.Button(root, text='Close', command=close_click,bg='white')
btn_close.place(x=90, y=30, height=30, width=70)

# Tạo nút kết nối UART
connect_button = tk.Button(root, text="Connect", state='disabled', command=connect_uart,bg='white',font=('Arial',12))
connect_button.place(x=300, y=140, height=30, width=100)

# Tạo nút ngắt UART 
disconnect_button = tk.Button(root,text='Disconnect',state ='disabled',bg='white',font=('Arial',12),command=disconnect_uart)
disconnect_button.place(x=420,y=140,height=30,width=100)

''' chức năng auto'''
# Hàm cho nút Auto
def auto_click():

    # Kích hoạt các elements của auto
    for obj in objects_4:
        obj['state'] = 'normal'
    
    # Disable elements of manual
    for obj in objects_1:
        obj['state']= 'disabled'

# Tạo nút Auto
btn_auto = tk.Button(root, text='Auto', state='disabled',bg='white',command=auto_click)
btn_auto.place(x=10, y=90, height=30, width=70)
objects_2.append(btn_auto)

# Tạo nhãn cho Auto 
auto_fr_label = tk.Label(root,text = 'Auto Control',bg = GUI_color, font=('Arial',16,'bold'))
auto_fr_label.place(x= 200, y =545)

# Tạo nhãn cho Auto Frame
auto_frame = tk.Frame(root,height=120,width=340,highlightthickness=2,highlightbackground='#241468',bg=manu_color)
auto_frame.place(x=200,y=580)


'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------FUNCTIONS---------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''


# Hàm tam xuất giá trị
def map(inValue,  inMax,  inMin, outMax,  outMin ):

	if inValue > inMax: 
	
		return outMax
	
	elif inValue < inMin:

		return outMin

	else:

		return (inValue-inMin)*(outMax-outMin)/(inMax-inMin) + outMin

    
# Hàm thuật toán đánh lái stanley control của bánh tước theo tọa độ điểm   
@jit(nopython = True)
def stanley_control_point(current_point,target_point,velocity,k_coef):
    psi = target_point[2]-current_point[2]
    front_wheel_point = [0,0] # tạo điểm lưu tọa độ của bánh trước
    front_wheel_point[0] = current_point[0] + sin(radians(-current_point[2])) * 2.1 # giá trị trục X
    front_wheel_point[1] = current_point[1] + cos(radians(-current_point[2])) * 2.1 # giá trị trục Y
    e = sqrt((target_point[0]-front_wheel_point[0])**2 + (target_point[1] - front_wheel_point[1])**2 )
    if velocity ==0:
        delta  = psi + atan(0)
    else:
        delta  = psi + atan((k_coef*e)/velocity)

    if delta == 0:
        return 20000
    elif delta > 0:
        pulse = int(map(delta, 38, 0, 39900, 20000))
        return pulse
    elif delta < 0:
        pulse = int(map(delta, 0, -38, 20000, 100))
        return pulse
    
# hàm tìm ra điểm tiếp theo
@jit(nopython = True)
def calulation_next_point(start_point_X,start_point_Y,distance_turn, angle_turn):
    x = start_point_X
    y = start_point_Y
    x_turn = sin(radians(-angle_turn)) * distance_turn 
    y_turn = cos(radians(-angle_turn)) * distance_turn 
    x_new =  (x_turn+x)
    y_new =  (y_turn+y)
    return x_new,y_new

# Hàm tính toán quỹ đạo sử dụng thuật toán Hybrid A*
def calculate_trajectory(map,startPoint,goalPoint):
    startPoint = matlab.double(startPoint) # [meters, meters, radians]
    goalPoint  = matlab.double(goalPoint)
    map  = matlab.double(map.tolist())
    result = eng.Hybrid_Astar(map,startPoint,goalPoint)
    result = np.array(result)
    point = result[:,:3]
    point[:,2] = (point[:,2]*180/pi)-90
    direction = result[:,3]
    return point, direction

@jit(nopython = True)
def check_collision_goal (loc_map, goal_x, goal_y):
    loc_x_check = int(goal_x)
    loc_y_check = int(200-goal_y)
    loc_y_up = max(min(loc_y_check-10, 200), 0)
    loc_y_down = max(min(loc_y_check+11, 200), 0)

    
    chk_right = 0
    chk_left = 0
    max_cnt_right = 200 - loc_x_check
    max_cnt_left  = loc_x_check 
    collision = False
    # dò sang bên phải 
    for cnt_right in range(0,max_cnt_right):
        column = loc_map[loc_y_up:loc_y_down,loc_x_check+cnt_right]
        if np.all(column == 0):
            chk_right = chk_right + 1 
            if chk_right == 10:
                break
        else: 
            chk_left = 0
    for cnt_left in range(0,max_cnt_left):
        column = loc_map[loc_y_up:loc_y_down,loc_x_check-cnt_left]
        if np.all( column == 0):
            chk_left = chk_left + 1 
            if chk_left == 10:
                break
        else: 
            chk_right = 0
    if cnt_right > max_cnt_right-5  and  cnt_left>max_cnt_left-5 :
        collision = True
        x_out = 0
        y_out = 0
    
    elif cnt_right > max_cnt_right-5 :
        x_out = goal_x - cnt_left + chk_left
        y_out = goal_y


    elif cnt_left > max_cnt_left-5:
        x_out = goal_x + cnt_right - chk_right
        y_out = goal_y

    else:
        if cnt_right <= cnt_left:
            x_out = goal_x + cnt_right - chk_right
            y_out = goal_y

        else:
            x_out = goal_x - cnt_left + chk_left
            y_out = goal_y


    map = np.copy(loc_map)
    return map,collision, x_out,y_out
    
@jit(nopython=True)
def create_local_map(glo_map,current_x, current_y):
    loc_map = np.ones((200,200))
    map = np.copy(glo_map)
    x=current_x
    y=map.shape[0]/10-current_y
    start_x = max(int(x*10) - 100, 0)
    start_y = max(int(y*10) - 100, 0)
    end_x = min(start_x + 200, map.shape[1])
    end_y = min(start_y + 200, map.shape[0])
    loc_map_temp = map[start_y:end_y, start_x:end_x]
    loc_map[:loc_map_temp.shape[0], :loc_map_temp.shape[1]] = loc_map_temp
    return loc_map,start_x,end_x,start_y,end_y

@jit(nopython=True)
def find_goalPose (glo_state,current_x, current_y, current_step ):
    state = np.copy(glo_state)
    current_x = current_x*10
    current_y = current_y*10
    for step in range(current_step,len(state)):
        dis = sqrt((state[step,0]-current_x)**2 + (state[step,1]-current_y)**2)
        if dis >= 90:
            break
    goal = state[step]
    goal[0]= max(min(100+(goal[0]-current_x), 199), 0)
    goal[1]= max(min(100+(goal[1]-current_y), 199), 0)
    goal[2]= (actual_angle+90)*pi/180
    return goal,step

# Tạo ma trận chuyển đổi từ hệ quy chiếu local image sang hệ quy chiếu global image

def transformation_matrix(center,arr_point_in):
    point_in = np.copy(arr_point_in)
    point_in = point_in.astype(np.int64)
    array_out = np.array([])
    rotation_matrix = np.array([[1, 0 ],
                                [0, -1],])
    translation_vector = np.array([center[0]*10-100, 750-(center[1]*10-100)])
    for i in range(0, len(point_in)):
        point = np.dot(rotation_matrix, point_in[i]) + translation_vector
        if array_out.size == 0:  # Kiểm tra xem array_out có rỗng không
            array_out = point
        else:
            array_out = np.vstack([array_out, point]) 
    return array_out

'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''


'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------POPUP ĐIỀU KHIỂN--------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

x_0 = 0 
y_0 = 0
goalPose = []
draw_signal = False
# cờ chạy
run = False
list_state=[]
# opencv để hiển thị vị trí và vẽ vật cản
globalMap = cv2.imread('globalMapParking.png', cv2.IMREAD_GRAYSCALE)





@jit(nopython=True)
def create_binary_map(map_array):
    binary_map = np.copy(map_array)
    # Avoid multi-dimensional indexing by iterating over each element
    for i in range(binary_map.shape[0]):
        for j in range(binary_map.shape[1]):
            if binary_map[i, j] > 100:
                binary_map[i, j] = 1
            else :
                binary_map[i, j] = 0
    return binary_map


# Hàm callback khi chuột được click
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(globalMap, (x, y), 1, 255, -1)

def draw_global_map ():
    global list_state,draw_signal
    global x_0,y_0
    global init_map, output_map,local_map
    # Tạo một cửa sổ OpenCV và thiết lập hàm callback
    cv2.namedWindow('Draw Circle',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Draw Circle', 150, 750)
    cv2.setMouseCallback('Draw Circle', draw_circle) 
    init_map = create_binary_map(globalMap)

    while True:
        output_map = create_binary_map(globalMap)
        local_map,start_x_loc_map,end_x_loc_map,start_y_loc_map,end_y_loc_map = create_local_map(output_map,current_x = x_0, current_y=y_0)
        
        if draw_signal:
            center = np.array([x_0,y_0])
            list_point = np.int32(transformation_matrix(center,list_state[:,:2]))
            cv2.polylines(globalMap, [list_point], isClosed=False, color=95, thickness=1)
            cv2.rectangle(globalMap, (start_x_loc_map, start_y_loc_map), (end_x_loc_map, end_y_loc_map), 95, 1)
        cv2.imshow('Draw Circle', globalMap)
        # Nhấn phím 'ESC' để thoát
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
    # np.savetxt('map_test.csv', local_map, delimiter=',')
    

# Hàm nút Start
def start_click():
    threading.Thread(target= draw_global_map).start()
    # phân luồng để vẽ đồ thị

       
# Nút start vẽ đồ thị 
start_btn = tk.Button(root, text = 'Open Map',bg='white',command = start_click,state='disabled')
start_btn.place(x=460,y = 500,height=30,width=80)
objects_2.append(start_btn)



# tạo nhập thông tin vị 
'''điểu khiển bánh trước sau chạy auto'''



def table_info():
    global x_init,y_init,x_0,y_0
    global goalPoseGlobal,list_state_global



    info_popup = tk.Toplevel(root)
    info_popup.title('Entry infomation table')
    info_popup.resizable(height=False,width=False)

    # Tính toán vị trí của info_popup
    root_width = root.winfo_width()
    root_height = root.winfo_height()
    info_popup_width = 360
    info_popup_height = 400
    x = root.winfo_rootx() + (root_width - info_popup_width) // 2
    y = root.winfo_rooty() + (root_height - info_popup_height) // 2
    info_popup.geometry(f"{info_popup_width}x{info_popup_height}+{x}+{y}")
    info_popup.configure(bg=GUI_color)
    
    Entry_frame = tk.Frame(info_popup,height=130,width=340,highlightthickness=2,highlightbackground='#8DF8FF',bg=manu_color)
    Entry_frame.place(x=10,y=10)

    # Tạo nhãn ô nhập X
    X_label_start = tk.Label(Entry_frame,text ='Entry X start',bg=manu_color)
    X_label_start.place(x= 10, y=10)

    #Tạo Entry X
    X_entry_start = tk.Entry(Entry_frame,relief=tk.SUNKEN,justify='center',font=('Arial',13,'bold'))
    X_entry_start.place(x=10,y=30,height=30,width=70)

    # Tạo nhãn cho ô Y
    Y_label_start = tk.Label(Entry_frame,text ='Entry Y start',bg=manu_color)
    Y_label_start.place(x=100, y= 10 )

    # Tạo ô ghi Y
    Y_entry_start = tk.Entry(Entry_frame,relief=tk.SUNKEN,justify='center',font=('Arial',13,'bold'))
    Y_entry_start.place(x=100, y= 30, height=30,width=70)

    # Tạo nhãn cho ô X muc tieu
    X_label_goal = tk.Label(Entry_frame,text ='Entry X goal',bg=manu_color)
    X_label_goal.place(x=10, y= 65 )

    # Tạo ô ghi X muc tieu
    X_entry_goal = tk.Entry(Entry_frame,relief=tk.SUNKEN,justify='center',font=('Arial',13,'bold'))
    X_entry_goal.place(x=10, y= 85, height=30,width=70)

    # Tạo nhãn cho ô Y muc tieu
    Y_label_goal = tk.Label(Entry_frame,text ='Entry Y goal',bg=manu_color)
    Y_label_goal.place(x=100, y= 65 )

    # Tạo ô ghi X muc tieu
    Y_entry_goal = tk.Entry(Entry_frame,relief=tk.SUNKEN,justify='center',font=('Arial',13,'bold'))
    Y_entry_goal.place(x=100, y= 85, height=30,width=70)

    # Tạo nhãn cho ô goc muc tieu
    angle_label_goal = tk.Label(Entry_frame,text ='Entry Angle goal',bg=manu_color)
    angle_label_goal.place(x=190, y= 65 )

    # Tạo ô ghi goc muc tieu
    angle_entry_goal = tk.Entry(Entry_frame,relief=tk.SUNKEN,justify='center',font=('Arial',13,'bold'))
    angle_entry_goal.place(x=190, y= 85, height=30,width=70)

  # hien thi ket qua khi an nut set
    result_text = "Coordinate of start point : \nCoordinate of goal point : "
    result_display = tk.Label(info_popup,text =result_text,justify='left',bg=GUI_color,font=('Arial',13,'bold'))
    result_display.place(x=10, y= 150 )
    

    # Hàm nút set
    def set_click():
        global x_init,y_init,x_0,y_0
        global goalPoseGlobal,list_state_global
        # Lấy giá trị từ các ô nhập liệu điểm bắt đầu
        x_init = X_entry_start.get()
        y_init = Y_entry_start.get()

        x_init=float(x_init)
        y_init=float(y_init)

        x_0 = x_init
        y_0 = y_init

        # Lấy giá trị từ các ô nhập liệu điểm mục tiêu

        x_goal = X_entry_goal.get()
        y_goal = Y_entry_goal.get()
        a_goal = angle_entry_goal.get()
        # Tọa độ điểm mục tiểu
        x_goal=float(x_goal)
        y_goal=float(y_goal)
        a_goal=float(a_goal)
        goalPoseGlobal = [x_goal*10,y_goal*10,(a_goal+90)*pi/180]

        #tạo quỹ đạo di chuyển trên global map
        start_state = [x_init*10, y_init*10, (actual_angle+90)*pi/180]
        list_state_global,_ = calculate_trajectory(map = init_map ,startPoint = start_state, goalPoint = goalPoseGlobal)
        # Hiển thị điểm vừa nhập
        result_display['text'] = "Coordinate of start point : {}, {}, {}\nCoordinate of goal point : {}, {}, {}\nCREATE GLOBAL LIST STATE SUCCESSFULLY !!!".format(x_init,x_init,actual_angle,x_goal,y_goal,a_goal)
        # Xóa dữ liệu vừa nhập trong ô
        X_entry_start.delete(0, tk.END)
        Y_entry_start.delete(0, tk.END)
        X_entry_goal.delete(0, tk.END)
        Y_entry_goal.delete(0, tk.END)
        angle_entry_goal.delete(0, tk.END)


    # Tạo nút SET điểm
    set_btn = tk.Button(info_popup,text='Set',bg='white',command=set_click)
    set_btn.place(x=300,y=250,height=30,width=50)
    info_popup.mainloop()
# Nút nhập vị trí
stop_btn = tk.Button(root,text ='Entry info',bg='white',command=table_info,state='disabled')
stop_btn.place(x=460,y=540,height=30,width=80 )
objects_2.append(stop_btn)

'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''




'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------ĐIỀU KHIỂN AUTOMATION---------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''



'''điểu khiển bánh trước sau chạy auto'''
def car_auto_control():
    global draw_signal
    global local_map,x_0,y_0,x_1,y_1
    global list_state,list_direction,goalPose,goalPoseGlobal,list_state_global
    index = 0
    p_index = 0
    while run:
        start = time.time()

        x_1,y_1 = calulation_next_point(start_point_X= x_0,start_point_Y=y_0,distance_turn = actual_dis,angle_turn=actual_angle)
        x_0,y_0 = x_1,y_1
        
        goalPose,index = find_goalPose (glo_state= list_state_global ,current_x = x_1, current_y = y_1, current_step = p_index)
        p_index = index
        goalPose = goalPose.tolist()
        chk_local_map, check_colision, checked_goal_x, checked_goal_y = check_collision_goal (loc_map = local_map, goal_x = goalPose[0], goal_y = goalPose[1])
        goalChecked = [int(checked_goal_x),int(checked_goal_y), goalPose[2]]

        print(goalChecked)
        # mặc định vị trí xe luôn nằm giữa local map
        s_state = [100, 100, (actual_angle+90)*pi/180]

        list_state, list_direction = calculate_trajectory(map = chk_local_map ,startPoint = s_state, goalPoint = goalChecked)
        draw_signal = True
        # Kiểm tra xe xe đã đến điểm cuối chưa
        # if len(list_state)>=2:
        #     next_moving_point = list_state[1]
        # else:
        #     next_moving_point = goalPose
        #     next_moving_point[2] = next_moving_point[2]*180/pi -90

        # current_state = [10, 10, actual_angle]
        # next_moving_point[:2] = next_moving_point[:2]/10

        # # xác định chiều xe 
        # if list_direction[1]==1:
        #     direction = b'T'
        #     front_pulse =str(int(stanley_control_point(current_point= current_state ,target_point = next_moving_point,velocity=actual_vel,k_coef =3)))
        #     maximum_speed = 60

        # elif list_direction[1]==-1:
        #     direction = b'L'
        #     front_pulse =str(int(39900-stanley_control_point(current_point= current_state ,target_point = next_moving_point,velocity=actual_vel,k_coef =3)))
        #     maximum_speed = 30
        
        # back_speed = 45
        # # UART cho bánh trước
        # front_speed = chr(75)
        # f_uart_data = f_start_bit + front_speed.encode('utf-8') + front_pulse.encode('utf-8') + stop_bit
        # f_uart.write(f_uart_data)

        # #UART cho bánh sau
        # b_speed_str = str(back_speed)
        # b_uart_data = b_start_bit + direction + b_speed_str.encode('utf-8') + stop_bit
        # b_uart.write(b_uart_data)

        end = time.time()
        print("Elapsed (with compilation) = %s" % (end - start))
'''------ooo------'''

def go_click():
    global run 
    run = True
    brake_adc_emer = b'S'
    brake_emer = p_start_bit + brake_adc_emer + stop_bit
    emer_uart.write(brake_emer)
    threading.Thread(target=car_auto_control).start() 

# Mở ảnh
go_img = PhotoImage(file = 'go.png')

# Tạo nút Go
go_btn = tk.Button(auto_frame,image=go_img,bg= manu_color,borderwidth=0,command=go_click,state='disabled')
go_btn.place(x=60, y=20 ,height=85,width=85)
objects_4.append(go_btn)

'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''



lines = []
# Hàm khi ấn nút Emergency Button 
def em_click():
    global run
    run = False
    brake_adc_emer = b'E'
    brake_emer = p_start_bit + brake_adc_emer + stop_bit
    lines.append(brake_emer)

    back_speed = str(0)
    direction = b'T'
    back_emer = b_start_bit + direction + back_speed.encode('utf-8') + stop_bit
    lines.append(back_emer)

    for line in lines:
        emer_uart.write(line)

# Mở ảnh
emer = PhotoImage(file = emergency_stop)

# Emergency Stop button for manual mode creation
emer_button = tk.Button(auto_frame,image = emer, bg= manu_color, borderwidth=0, state='disabled',command = em_click )
emer_button.place(x=190, y=10, width=100, height=100)
objects_4.append(emer_button)


'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''


'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------CHẾ ĐỘ MANUAL-----------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''
# Hàm chức năng cho nút Manual
def manual_click():
    for obj in objects_1:
        obj['state'] = 'normal'

    for obj in objects_4:
        obj['state'] = 'disabled'

# Tạo nút Manual
btn_manu = tk.Button(root, text='Manual', state='disabled',bg='white',command= manual_click)
btn_manu.place(x=90, y=90, height=30, width=70)
objects_2.append(btn_manu)

# Tạo nhãn cho manu frame
manu_fr_label = tk.Label(root,text='Manual Control', bg=GUI_color,font=('Arial',16,'bold'))
manu_fr_label.place(x=10,y=140)

'''Tạo frame cho Manual Control'''
manu_frame = tk.Frame(root,width =530,height=265, highlightbackground='#241468',highlightthickness=2,bg=manu_color )
manu_frame.place(x= 10, y= 180)

# Tạo nhãn cho di chuyển trước và sau
back_wheel_label = tk.Label(manu_frame,text='Back Wheel Control',bg=manu_color,font=('Arial',12,'bold'))
back_wheel_label.place(x=10,y=10)

# Tạo nhãn cho bánh trước
front_wheel_label = tk.Label(manu_frame,text='Front Wheel Control',bg=manu_color,font=('Arial',12,'bold'))
front_wheel_label.place(x=205, y=10)

# Tạo nhãn cho phanh
brake_label = tk.Label(manu_frame, text = 'Brake Control',bg=manu_color,font=('Arial',12,'bold'))
brake_label.place(x= 205, y= 135 )

''' Tạo frame cho bánh sau'''
back_frame = tk.Frame(manu_frame, width=185, height=215, highlightbackground='#645CAA', highlightthickness=2,bg=manu_color)
back_frame.place(x= 10, y= 40)

# Gía trị hàng đơn vị khi không nhấn nút
back_adc_unit = b'0'

# Gía trị hàng chục khi không nhấn nút 
back_adc_dozen = b'0'

# Gía trị hàng trăm khi không nhấn nút
back_adc_hundred = b'0'

# Hợp 3 số lại 
back_adc = back_adc_hundred + back_adc_dozen + back_adc_unit

'''Code cho nút forward'''

# Hàm cho nút forward để đổi hình khi ấn
def forward(event):

    global forward_btn
    if btn_forward['state'] != 'disabled':
        if event.type == '4':  # ButtonPress event
            # Mở ảnh khi nút được nhấn
            forward_btn = PhotoImage(file=back_for_press)
            #truyền UART cho bánh sau
            if b_uart.is_open:
                back_for_adc = str(int(linear_scale.get()))
                direction = b'T'
                uart_data = b_start_bit + direction + back_for_adc.encode('utf-8') + stop_bit
                b_uart.write(uart_data)
            else:
                messagebox.showwarning('Warning','Please connect the UART')
        else:
            # Mở ảnh mặc định khi nút được thả ra
            forward_btn = PhotoImage(file=back_for_release)
            # truyền UART cho bánh sau
            if b_uart.is_open:
                
                uart_data = b_start_bit + back_adc + stop_bit
                b_uart.write(uart_data)
            else:
                messagebox.showwarning('Warning','Please connect the UART')

    btn_forward['image'] = forward_btn

# Mở ảnh 
forward_btn = PhotoImage(file=back_for_release)

# Tạo nút forward
btn_forward = tk.Button(back_frame, image=forward_btn, borderwidth=0, state='disabled', bg='#A8DF8E')
btn_forward.place(x=10, y=10, height=90, width=90)
objects_1.append(btn_forward)

# Gắn sự kiện cho nút forward
btn_forward.bind('<ButtonPress-1>', forward)
btn_forward.bind('<ButtonRelease-1>', forward)

''' Code cho Reverse Button '''
# Hàm cho nút Reverse để dổi hình khi ấn
def reverse(event):

    global reverse_btn

    if btn_reverse['state']!='disabled': # check trạng thái của nút
        if event.type == '4':  # ButtonPress event
            # Mở ảnh khi nút được nhấn
            reverse_btn = PhotoImage(file=back_rev_press)
            #truyền UART giá trị tốc độ cho bánh sau
            if b_uart.is_open:
                back_rev_adc = str(int(linear_scale.get())) #gán thành chuỗi kí tự
                direction = b'L'
                uart_data = b_start_bit + direction + back_rev_adc.encode('utf-8') + stop_bit
                b_uart.write(uart_data)

            else:
                messagebox.showwarning('Warning','Please connect the UART')
        else:
            # Mở ảnh mặc định khi nút được thả ra
            reverse_btn = PhotoImage(file=back_rev_release)
            #truyền UART giá trị tốc độ cho bánh sau
            if b_uart.is_open:
                uart_data = b_start_bit + back_adc  + stop_bit
                b_uart.write(uart_data)
            else:
                messagebox.showwarning('Warning','Please connect the UART')

    btn_reverse['image']= reverse_btn

# Mở ảnh
reverse_btn = PhotoImage(file=back_rev_release)

# Tạo nút Reverse
btn_reverse = Button(back_frame, image=reverse_btn, bg='#A8DF8E', borderwidth=0, state='disabled')
btn_reverse.place(x=10, y=110, width=90, height=90)
objects_1.append(btn_reverse)

# Gắn sự kiện cho nút reverse
btn_reverse.bind('<ButtonPress-1>', reverse)
btn_reverse.bind('<ButtonRelease-1>', reverse)

''' Thanh điều tốc cho bánh sau '''
linear_scale = tk.Scale(back_frame, from_=0, to=100, orient=tk.VERTICAL,bg='white',
                        showvalue=True,state='disabled')
linear_scale.set(0)
linear_scale.place(x = 120, y= 10,height = 190)
objects_1.append(linear_scale)

'''Tạo frame cho bánh trước'''
front_frame = tk.Frame(manu_frame,height=85, width=315,bg=manu_color,highlightbackground='#645CAA',highlightthickness=2)
front_frame.place(x=205,y=40)

''' Code cho tạo thanh trượt rẽ trái phải'''
#truyền UART bánh trước
def turn_slide(value):
    valuess = int(value)
    if valuess <= 100:
        turn_adc = str(map(valuess,0, 100, 39900, 20000))
    elif valuess>100:
        turn_adc = str(map(valuess, 200, 100, 100, 20000))
    print(turn_adc)
    turn_speed = chr(70)
    uart_data = f_start_bit + turn_speed.encode('utf-8') + turn_adc.encode('utf-8') + stop_bit    
    if f_uart.is_open:
        f_uart.write(uart_data)  

# Tạo ô trượt điều khiển trái phải
turn_scale = tk.Scale(front_frame, from_=0, to=200, orient=tk.HORIZONTAL,bg='white',bd=1, 
    tickinterval=25, showvalue=True,troughcolor='white',state='disabled',command=turn_slide)
turn_scale.place(x = 5, y= 5, height= 70, width = 300)
turn_scale.set(100)
objects_1.append(turn_scale)

'''Tạo frame cho phanh'''
brake_frame = tk.Frame(manu_frame,height=90,width=315,bg=manu_color,highlightbackground='#645CAA',highlightthickness=2)
brake_frame.place(x=205,y=165)

# Start bit for UART transmission on brake
p_start_bit = b'P'

# Gía trị hàng đơn vị khi không nhấn nút 
brake_adc_unit = b'0'

#Gía trị hàng chục khi không nhấn nút 
brake_adc_dozen = b'0'

#Gía trị hàng trăm khi không nhấn nút 
brake_adc_hundred = b'0'

#Hợp giá trị 3 số 
brake_adc_1 = brake_adc_hundred + brake_adc_dozen + brake_adc_unit

''' Code cho nút phanh'''
# Hàm cho nút Brake để đổi hình khi ấn
def brake(event):

    global brake_btn

    if btn_brake['state']!='disabled':
        if event.type == '4':  # ButtonPress event
            # Mở ảnh khi nút được nhấn
            brake_btn = PhotoImage(file=brake_press)
            # truyền uart cho adc cho phanh
            if p_uart.is_open:
                brake_adc_0 = str(int(brake_slide.get()))
                uart_data = p_start_bit + brake_adc_0.encode('utf-8') + stop_bit
                p_uart.write(uart_data)
        else:
            # Mở ảnh mặc định khi nút được thả ra
            brake_btn = PhotoImage(file=brake_release)
            # truyền uart cho adc cho phanh
            if p_uart.is_open:
                uart_data = p_start_bit + brake_adc_1 + stop_bit
                p_uart.write(uart_data)
                
    btn_brake['image']= brake_btn

# Mở ảnh
brake_btn = PhotoImage(file=brake_release)

# Tạo nút Brake
btn_brake = Button(brake_frame, image=brake_btn, bg=manu_color, borderwidth=0, state='disabled')
btn_brake.place(x=10, y=10, width=70, height=70)
objects_1.append(btn_brake)

# Gắn sự kiện cho nút reverse
btn_brake.bind('<ButtonPress-1>', brake)
btn_brake.bind('<ButtonRelease-1>', brake)

'''Code cho thanh trượt độ lớn xung cấp cho phanh'''
brake_slide = tk.Scale(brake_frame,from_=0,to=100,orient=tk.HORIZONTAL,bg='white',bd=1,tickinterval=25,
                       showvalue=True,troughcolor='white',state='disabled')
brake_slide.set(0)
brake_slide.place(x=85,y=10,width=220,height=70)
objects_1.append(brake_slide)


'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''



# Chạy vòng lặp giao diện
root.mainloop()
eng.quit()
print("EXIT MATLAB ENGINE !!!")