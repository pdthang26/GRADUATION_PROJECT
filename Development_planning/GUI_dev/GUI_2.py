import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
import serial.tools.list_ports
from PIL import ImageTk, Image
import serial
import os
import threading
from math import *
import numpy as np
import matlab.engine
import cv2
from pyzbar import pyzbar
from numba import jit
import time
from ultralytics import YOLO
import socket
import pandas as pd

# # Khởi động matlab engine
# eng = matlab.engine.start_matlab()
# print('MALAB ENGINE FINISHED BEGINNING !!!')

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
root.geometry("635x675")
root.configure(bg=GUI_color)
root.resizable(height=False, width=False)

# Tạo mảng bao gồm các thành phần trên giao diện
objects_1 = [] # mảng chứa các thành phần để active bằng nút manual
objects_2 = [] # mảng chứa các elements để active bằng nút Connect
objects_3 = [] # các element combobox về UART parameter
objects_4 = [] # mảng để chứa elements được active bằng nút auto

# Các biến dùng truyền UART
anglar_vel_uart = emer_uart= b_uart= f_uart= p_uart= ang_uart= vel_uart= dis_uart = None

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

    global b_uart,f_uart,p_uart # các biến UART cho điều khiển bánh trước, sau, phanh
    global ang_uart,vel_uart,dis_uart, anglar_vel_uart #biến UART đọc các giá trị ghi nhận
    global emer_uart
  
    # Các biến parameter cho UART
    selected_port = com_port.get()
    time = 1
        
    # Kiểm tra nếu cổng UART không được cung cấp
    if (selected_port == ''):
        messagebox.showwarning('Warning', 'The COM/GPS port is empty.\nPlease select a COM/GPS port.')
    else:
        try:
            # Khởi tạo đối tượng Serial
            anglar_vel_uart = emer_uart= ang_uart= vel_uart= dis_uart= b_uart= f_uart= p_uart =serial.Serial(
            port=selected_port,
            baudrate=115200,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            timeout=time  # Timeout cho phép đọc từ giao diện UART
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
            actual_angle = float(angle[1:].replace('\x00','')) 
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
            print(actual_vel)
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

# Hàm nhấn nút show value
def show():
    while True:
        show_angle()
        show_dis()
        show_vel()
        show_angular_vel()
               
# phân luồng cho nút show 
def show_click():
    threading.Thread(target = show).start()
    
# Tạo nút Show value
show_button = tk.Button(root,text = 'Show Value',state= 'disabled',bg='white',command=show_click,font=('Arial',11))
show_button.place(x=170,y=90,height=30,width=100)
objects_2.append(show_button)

#Hàm cho nút Disconnect
def disconnect_uart():
    
    global update_flag

    update_flag = False

    for obj in objects_1+objects_2:
        obj['state'] = 'disabled'

    com_port['text']=''
   
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
    global socketClient
    # ban đầu gửi 'Test' để máy Thắng tìm thấy địa chỉ IP
    datatoSend = 'test'.encode('utf-8')
    socketClient.sendto(datatoSend,('192.168.1.1',2024))

    data, address = socketClient.recvfrom(1024)
    data = data.decode('utf-8')
    print(address)
    print('data from Server: ',data)

    i=0
    if data == 'success':
        #Kích hoạt nút Connect 
        connect_button['state']='normal'
            
    else:
        i+=1
        time.sleep(1)
        
    if i == 60 and data=='':
        print('Reset')

    print(data)

def open_act():
    connect_button['state']='normal'
    #Kích hoạt các Combobox 
    for obj in objects_3:
        obj['state']='normal'
    threading.Thread(target=open_click).start()
    
# Hàm cho nút Close
def close_click():

    # Hiện ô thông báo lữa chọn muốn đóng cửa sổ giao diện
    result = messagebox.askyesno("Exit", "Do you want to exit?")
    if result:
        root.destroy()

'''Tạo các nhãn'''
# Tạo nút Open
btn_open = tk.Button(root, text='Open', command=open_act,bg='white')
btn_open.place(x=10, y=30, height=30, width=70)

# Tạo nút Close cửa sổ chương trình
btn_close = tk.Button(root, text='Close', command=close_click,bg='white')
btn_close.place(x=90, y=30, height=30, width=70)

# Tạo nhãn cho hiển thị Angle
angle_label = tk.Label(root,text='Angle',bg=GUI_color)
angle_label.place(x=280,y= 5)

#Tạo ô hiển thị cho Angle
angle_display = tk.Label(root,relief=tk.SUNKEN,anchor=tk.W,padx=10,bg='white',font=('Arial',13,'bold'))
angle_display.place(x=280,y=30,height=30,width=100)

# Tạo nhãn hiển thị đơn vị góc quay
ang_unit = tk.Label(root,text = '\u00B0',bg=GUI_color,font=('Arial',15,'bold'))
ang_unit.place(x=380,y=30)

#Tạo nhãn cho Distance
dis_label = tk.Label(root,text='Distance',bg=GUI_color)
dis_label.place(x=420,y=5)

#Tạo ô hiển thị cho Distance 
dis_display = tk.Label(root,relief=tk.SUNKEN,anchor=tk.W,padx=10,bg='white',font=('Arial',13,'bold'))
dis_display.place(x=420,y=30,height=30,width=100)

#Tạo nhãn hiển thị đơn vị cho Distance
dis_unit =tk.Label(root,text='m',bg=GUI_color,font=('Arial',13))
dis_unit.place(x=520,y=30)

# Tạo nhãn cho Speed
vel_label = tk.Label(root,text='Speed',bg=GUI_color)
vel_label.place(x=280,y=65)

#Tạo ô hiển thị Speed
vel_display= tk.Label(root,relief=tk.SUNKEN,anchor=tk.W,padx=10,bg='white',font=('Arial',13,'bold'))
vel_display.place(x=280,y=90,width=100,height=30)

# Tạo nhãn đơn vị cho tốc độ
speed_unit = tk.Label(root,text='m/s',bg=GUI_color,font=('Arial',11))
speed_unit.place(x=380,y=90)

#Tạo nhãn hiển thị cho angular velocity
angular_vel_label = tk.Label(root,text='Angular velocity',bg=GUI_color)
angular_vel_label.place(x=420,y= 65)

#Tạo ô hiển thị angular velocity
angular_vel_display= tk.Label(root,relief=tk.SUNKEN,anchor=tk.W,padx=10,bg='white',font=('Arial',13,'bold'))
angular_vel_display.place(x=420,y=90,width=100,height=30)

# Tạo nhãn đơn vị cho angular velocity
angular_vel_unit = tk.Label(root,text='rad/s',bg=GUI_color,font=('Arial',11))
angular_vel_unit.place(x=520,y=90)

# Tạo nhãn cho ô chọn cổng COM
com_label = tk.Label(root, text="COM Port:", bg=GUI_color)
com_label.place(x=170, y=5)
# Lấy danh sách tất cả các cổng COM 
com_ports = [port.device for port in serial.tools.list_ports.comports()]

''' Tạo các combobox cho các thông số liên quan tới UART'''
# Tạo ô chọn cổng COM cho điều khiển
com_port = ttk.Combobox(root, values=com_ports, state='disabled')
com_port.place(x=170, y=30, height=30, width=100)
objects_3.append(com_port)

''' Tạo các nút '''
# Tạo nút kết nối UART
connect_button = tk.Button(root, text="Connect", state='disabled', command=connect_uart,bg='white')
connect_button.place(x=10, y=90, height=30, width=70)

# Tạo nút ngắt UART 
disconnect_button = tk.Button(root,text='Stop',state ='disabled',bg='white', command=disconnect_uart)
disconnect_button.place(x=90,y=90,height=30,width=70)

'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------CHẾ ĐỘ AUTO---------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''
# Hàm cho nút Auto
def auto_click():

    # Kích hoạt các elements của auto
    for obj in objects_4:
        obj['state'] = 'normal'
    
    # Disable elements of manual
    for obj in objects_1:
        obj['state']= 'disabled'


# Tạo nhãn cho Auto 
auto_fr_label = tk.Label(root,text = 'Auto Control',bg = GUI_color, font=('Arial',16,'bold'))
auto_fr_label.place(x= 10, y =140)

# Tạo nhãn cho Auto Frame
auto_frame = tk.Frame(root,height=170,width=530,highlightthickness=2,highlightbackground='#241468',bg=manu_color)
auto_frame.place(x=10,y=180)

# Tạo nút Auto
btn_auto = tk.Button(auto_frame, text='Auto', state='disabled',bg='white',command=auto_click)
btn_auto.place(x=450, y=5, height=30, width=70)
objects_2.append(btn_auto)

#Tạo nhãn cho vị trí hiện tại của xe
current_p_lbl = tk.Label(auto_frame,text='Current position',bg=manu_color)
current_p_lbl.place(x=5,y=1)

#Tạo ô hiển thị vị trí hiện tại
current_display = tk.Label(auto_frame,relief=tk.SUNKEN,anchor=tk.W,padx=5,bg='white')
current_display.place(x=5,y=25,height=30,width=200)

# Tạo nhãn Start Point
start_p_lbl = tk.Label(auto_frame,text='Start Point',bg=manu_color)
start_p_lbl.place(x=5,y=60)
# Tạo dropbox chọn start point
s_points = StringVar()
point_name = ['A','B','C','D','E','F']
start_menu = tk.OptionMenu(auto_frame,s_points,*point_name)
start_menu.place(x=5,y=84,height=30,width=80)

# Tạo nhãn End Point
end_p_lbl = tk.Label(auto_frame,text='End Point',bg=manu_color)
end_p_lbl.place(x=125,y=60)
# Tạo dropbox chọn End point
e_points = StringVar()
e_menu = tk.OptionMenu(auto_frame,e_points,*point_name)
e_menu.place(x=125,y=84,height=30,width=80)


# Tọa độ các điểm trạm
station_coors = {'A':(2,1,0),
                 'B':(11,30,0),
                 'C':(2.5,50,0),
                 'D':(10,9,90),
                 'E':(5,23,0),
                 'F':(10,53,0)}

#Hàm chức năng khi thực hiện nhấn nút set
def set_click():
    # Xem xét hướng đi của xe trên map
    map_dir = ''

    # Xem xét chiều đi của xe đang theo hướng nào trên map
    if s_points.get()=='A' or s_points.get()=='B':
        map_dir = 'T'
    elif s_points.get()=='D' or s_points.get()=='E':
        map_dir = 'N'

    data = f'{station_coors[s_points.get()][0]},{station_coors[s_points.get()][1]},{station_coors[s_points.get()][2]},{station_coors[e_points.get()][0]},{station_coors[e_points.get()][1]},{station_coors[e_points.get()][2]},{map_dir}'
    datatoSend = data.encode('utf-8')
    socketClient.sendto(datatoSend,('192.168.1.1',2024))

# Tạo nút Set
set_btn = tk.Button(auto_frame,text ='Set',state='disabled',bg='white',command=set_click)
set_btn.place(x=215,y=25,height=30,width=50)
objects_4.append(set_btn)
'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------FUNCTIONS---------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

#Hàm đọc QR code
def qr_read(img):
    qr_code_data = ''
    qr_codes = pyzbar.decode(img)
    for qr_code in qr_codes:
        qr_code_data = str(qr_code.data.decode('utf-8'))
    return qr_code_data

#Hàm nhận diện biển báo
model = YOLO('best_3.pt')
def sign_detect(img):
    global model
    resized_img = cv2.resize(img,(640,640))
    # resized_img_RGB = cv2.cvtColor(resized_img,cv2.COLOR_BGR2RGB)
    result = model.predict(source=resized_img)
    # Extract predictions
    predictions = result[0].boxes.data
    class_label = None
    for pred in predictions:
        if len(pred) >= 6:
            x1, y1, x2, y2, confidence, class_id = pred
            class_label = model.names[int(class_id)]
    return result,class_label

#Hàm quyết định thực hiện hành động của mỗi biến báo
def sign_decide(class_label, b_speed):
    # Sử dụng từ điển để ánh xạ class_label tới các hành động
    actions = {
        'right': (1, b_speed),
        'junction': (1, b_speed),
        'crossing': (0, 20),
        'parking': (0, b_speed),
        'stop': (0, 0),
        'forbid': (0, 0),
        'no detections':(0,b_speed)
    }
    
    # Giá trị mặc định
    flag = 0

    def delay_speed_restore():
        time.sleep(3)
        b_speed = b_speed
   
    # Kiểm tra và thực hiện hành động tương ứng
    if class_label in actions:
        action = actions[class_label]
        flag = action[0]
        b_speed = action[1]
        if class_label == 'stop':
            threading.Thread(target=delay_speed_restore).start()
    # Giá trị mặc định cho các nhãn không xác định
    return b_speed, flag
        
socketClient = socket.socket(family= socket.AF_INET, type=  socket.SOCK_DGRAM)
#hàm nhận qua Ethenet
def ethenet_receive():
    data,_ = socketClient.recvfrom(1024)
    data = data.decode('utf-8')
    return data

#hàm gửi qua Ethenet
def ethenet_send(angle,x,y,flag):
    data = f'{angle},{x},{y},{flag}'
    datatoSend = data.encode('utf-8')
    socketClient.sendto(datatoSend,('192.168.1.1',2024))

def read_file_csv():
    global angle,distance,velocity
    file_name = 'Car_data_run.csv'
    df = pd.read_csv(file_name)
    row_count = len(df)
    index = 0
    while True:  
        # Xuất giá trị 3 cột Angle, Distance, Velocity
        angle = df['Angle'].iloc[index]
        distance = df['Distance'].iloc[index]
        velocity = df['Velocity'].iloc[index]
        # Tăng chỉ số, quay lại từ đầu nếu đến cuối file
        index = (index + 1) % row_count
        time.sleep(0.1)
    
'''''''''''Các hàm tính toán'''
@jit(nopython = True)
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

#Hàm tính stanley theo curvature
@jit(nopython = True)
def stanley_control(desired_angle,current_angle,center_offset,velocity,k_coef):
    psi = desired_angle-current_angle
    err = center_offset
    if velocity ==0:
        delta  = psi + atan(0)
    else:
        delta  = psi + atan((k_coef*err)/velocity)

    if delta == 0:
        return 20000
    elif delta > 0:
        pulse = int(map(delta, 38, 0, 39900, 20000))
        return pulse
    elif delta < 0:
        pulse = int(map(delta, 0, -38, 20000, 100))
        return pulse    
    
#Hàm tính từ curvature ra góc
@jit(nopython = True)
def cvrt_cur(curv):
    k = (1/curv)
    L = 2.16
    alpha = atan(L*k)
    return alpha

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

@jit(nopython = True)
def length_calculate(x_car,y_car,x_st,y_st):
    arr = [(x_car,y_car),(x_st,y_st)]
    for i in range(len(arr) - 1):
        a1 = np.array(arr[i])
        a2 = np.array(arr[i+1])
        direction_A = a2 - a1
        length = np.linalg.norm(direction_A)
    return length

'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''


'''-----------------------------------------------------------------
--------------------------------------------------------------------
--------------------------ĐIỀU KHIỂN AUTOMATION---------------------
--------------------------------------------------------------------
--------------------------------------------------------------------'''

back_speed = 0
'''điểu khiển bánh trước sau chạy auto'''
def car_auto_control_send():
    global run, model, back_speed

    #Thiệt lập cam
    cap = cv2.VideoCapture(0)
    # Thiết lập độ phân giải
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Đặt giá trị exposure. Thông thường, giá trị này sẽ là âm, ví dụ: -4.0
    exposure_value = -4.0
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

    x = y = 0
    d_p = 0
    
    #Vòng lặp xử lý chính
    while run:
        # Đọc frame từ camera
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận frame (kết thúc chương trình)")
            break
        if ret:
            result,sign = sign_detect(frame)
            frame_ = result[0].plot()
            station = qr_read(frame)
            # visualize
            cv2.imshow('frame',frame_)

        # Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
        back_speed,flag= sign_decide(sign,30)
        # x_new,y_new = calulation_next_point(x,y,actual_dis-d_p,actual_angle)
        x_new,y_new = calulation_next_point(x,y,distance-d_p,angle)
        # ethenet_send(flag,x_new,y_new,actual_angle)
        ethenet_send(flag,x_new,y_new,angle)
        length = length_calculate(x_new,y_new,station_coors[e_points.get()][0],station_coors[e_points.get()][1])
        #Cập nhật các giá trị cũ
        x = x_new
        y = y_new
        #d_p = actual_dis
        d_p = distance

        #Nếu phát hiện end_point thì ngưng
        if station == e_points.get() or length <= 1 :
            run = False
        else:
            run = True

        print(back_speed,'/',flag)
        print(x_new,'/',y_new)

    # Giải phóng bộ nhớ
    cap.release()
    cv2.destroyAllWindows()


def car_auto_control_receive():
    global run,back_speed

    #Vòng lặp xử lý chính
    while run:
        data = ethenet_receive()
        if data[0] == 's':
            target_point = (data[2:].split(',')[0],data[2:].split(',')[1])
            desired_angle = data[2:].split(',')[2]
            dir = data[2:].split(',')[3]
            front_pulse =str(int(stanley_control_point(desired_angle,angle,target_point,distance,velocity,0.1)))
            # front_pulse =str(int(stanley_control_point(desired_angle,actual_angle,target_point,actual_dis,actual_vel,0.1)))
        elif data[0] == 'm':
            curv = data[2:].split(',')[0]
            center_oft = data[2:].split(',')[1]
            desired_angle = cvrt_cur(curv)
            front_pulse =str(int(stanley_control(desired_angle,angle,center_oft,distance,velocity,0.1)))
            # front_pulse =str(int(stanley_control(desired_angle,actual_angle,center_oft,actual_dis,actual_vel,0.1)))

        #Quyết định tiến hay lùi
        if int(dir) == 1:
            direction = b'T'
        elif int(dir) == -1:
            direction = b'L'
        
        # UART cho bánh trước
        front_speed = chr(75)
        f_uart_data = f_start_bit + front_speed.encode('utf-8') + front_pulse.encode('utf-8') + stop_bit
        f_uart.write(f_uart_data)

        # UART cho bánh sau
        b_speed_str = str(back_speed)
        b_uart_data = b_start_bit + direction + b_speed_str.encode('utf-8') + stop_bit
        b_uart.write(b_uart_data)

        print(front_pulse)

'''------ooo------'''
def go_click():
    global run 
    run = True
    if s_points.get() =='' or e_points.get() =='':
        messagebox.showwarning('Warning', 'Please select your Start point or End point')
        run = False
    else:
        run = True
    brake_adc_emer = b'S'
    brake_emer = p_start_bit + brake_adc_emer + stop_bit
    emer_uart.write(brake_emer)
    threading.Thread(target=car_auto_control_send).start() 
    threading.Thread(target=car_auto_control_receive).start()
    threading.Thread(target=read_file_csv).start()
    
# Mở ảnh
go_img = PhotoImage(file = 'go.png')

# Tạo nút Go
go_btn = tk.Button(auto_frame,image=go_img,bg= manu_color,borderwidth=0,command=go_click,state='disabled')
go_btn.place(x=330, y=60 ,height=80,width=80)
objects_4.append(go_btn)

'''nút emergency'''
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
emer_button.place(x=420, y=50, width=100, height=100)
objects_4.append(emer_button)
'''nút emergency'''

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

# Tạo nhãn cho manu frame
manu_fr_label = tk.Label(root,text='Manual Control', bg=GUI_color,font=('Arial',16,'bold'))
manu_fr_label.place(x=10,y=360)

'''Tạo frame cho Manual Control'''
manu_frame = tk.Frame(root,width =530,height=265, highlightbackground='#241468',highlightthickness=2,bg=manu_color )
manu_frame.place(x= 10, y= 400)

# Tạo nút Manual
btn_manu = tk.Button(manu_frame, text='Manual', state='disabled',bg='white',command= manual_click)
btn_manu.place(x=450, y=5, height=30, width=70)
objects_2.append(btn_manu)

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

# Gía trị hàng đơn vị, chục, trăm khi không nhấn nút
back_adc_unit = back_adc_dozen =back_adc_hundred=b'0'

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
    if value=='0':
        valuess = 1
    else:
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


# # Kết thúc MATLAB Engine khi không cần thiết nữa
# eng.quit()
# Chạy vòng lặp giao diện
root.mainloop()
print("EXIT MATLAB ENGINE !!!")