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

# Khởi động matlab engine
eng = matlab.engine.start_matlab()
print('MALAB ENGINE FINISHED BEGINNING !!!')


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
root.geometry("1320x710")
root.configure(bg=GUI_color)
root.resizable(height=False, width=False)

# Tạo mảng bao gồm các thành phần trên giao diện
objects_1 = [] # mảng chứa các thành phần để active bằng nút manual
objects_2 = [] # mảng chứa các elements để active bằng nút Connect
objects_3 = [] # các element combobox về UART parameter
objects_4 = [] # mảng để chứa elements được active bằng nút auto

# Các biến dùng truyền UART
emer_uart= b_uart= f_uart= p_uart= ang_uart= vel_uart= dis_uart = None
ultra_uart=gps_port = None

''' Chức năng giao diện '''

# Hàm cho nút connect
def connect_uart():

    #Kích hoạt nút Disconnect
    disconnect_button['state']='normal'

    # kích hoạt nút Manual, Auto, Show Car Value
    for obj in objects_2:
        obj['state'] = 'normal'

    global b_uart,f_uart,p_uart,ultra_uart
    global ang_uart,vel_uart,dis_uart
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
            ultra_uart=emer_uart= ang_uart= vel_uart= dis_uart= b_uart= f_uart= p_uart =serial.Serial(
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

                latitude_semisphere = data[4]
                longtitude_semisphere = data[6]

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
    show_angle()
    show_dis()
    show_vel()
    show_gps()
    ultrasonic = ultra_uart.readline().decode().strip()
    if ultrasonic.startswith('U'):
        ultra_data = ultrasonic[1:].replace('\x00','').split(',')
        if len(ultra_data) == 4:
            new_values = [float(value) for value in ultra_data]
            if new_values != ultra_values:
                ultra_values = new_values
    root.after(100, show)    

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

# Tạo nhãn cho Longitude
longitude_label= tk.Label(root,text='Longitude',bg = GUI_color)
longitude_label.place(x=10,y=520)

# Tạo ô hiển thị Longitude
longitude_display = tk.Label(root,relief=tk.SUNKEN,padx=5,bg='white',anchor=tk.W)
longitude_display.place(x=10,y=545,height=30,width=170)

# Tạo nhãn hiển thị Latitude
latitude_label = tk.Label(root,text='Latitude',bg=GUI_color)
latitude_label.place(x=10,y=580)

# Tạo ô hiển thị Latitude
latitude_display = tk.Label(root,relief=tk.SUNKEN,bg='white',anchor=tk.W,padx=5)
latitude_display.place(x=10,y=605,height=30,width=170)

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

# biến trạng thái 
state = 0

# Điểm đầu của xe
init =(0.0,0.0)

# Mảng để lưu các điểm
points = [init]

def type_click():
    global init,points,state
    state = 0
    # tạo popup chế độ nhập tọa độ điểm
    popup = tk.Toplevel()
    popup.title('Type mode')
    popup.resizable(height=False,width=False)
    popup.configure(bg=manu_color)
                
    # Tính toán vị trí của popup
    root_width = root.winfo_width()
    root_height = root.winfo_height()
    popup_width = 450
    popup_height = 265
    x = root.winfo_rootx() + (root_width - popup_width) // 2
    y = root.winfo_rooty() + (root_height - popup_height) // 2
    popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

    ''' Phần ô ghi tọa độ'''
    # Tạo nhãn ô nhập X
    X_label = tk.Label(popup,text ='X',bg=manu_color)
    X_label.place(x= 10, y=10)

    #Tạo Entry X
    X_entry = tk.Entry(popup,relief=tk.SUNKEN,justify='center',font=('Arial',13,'bold'))
    X_entry.place(x=10,y=40,height=30,width=173)

    # Tạo nhãn cho ô Y
    Y_label = tk.Label(popup,text ='Y',bg=manu_color)
    Y_label.place(x=200, y= 10 )

    # Tạo ô ghi Y
    Y_entry = tk.Entry(popup,relief=tk.SUNKEN,justify='center',font=('Arial',13,'bold'))
    Y_entry.place(x=200, y= 40, height=30,width=173)

    '''-----------ooo----------'''

    ''' tạo bảng hiển thị danh sách các điểm'''
    #create the frame
    style=ttk.Style()
    style.theme_use('default')
    style.configure('Treeview',
                    background='white',
                    foreground='black',
                    rowheight=30)
    style.configure("Treeview.Heading", font=('Arial', 10, 'bold'),background='#A6FF96',foreground='black')
    style.map('Treeview',background=[('selected', 'grey')])

    # Tạo bảng để hiển thị các giá trị
    tree = ttk.Treeview(popup, columns=("Index", "X", "Y"), show="headings")
    tree.heading("Index", text="Index")
    tree.heading("X", text="X")
    tree.heading("Y", text="Y")

    # Thiết lập độ rộng cho từng cột
    tree.column("Index", width=120, anchor="center")
    tree.column("X", width=120, anchor="center")
    tree.column("Y", width=120, anchor="center")
    tree.place(x=10, y=85, height=170, width=363)
    '''------ooo----'''
    # Hàm nút set
    def set_click():

        global points,run 
        # Set no go
        run = False

        # Lấy giá trị từ các ô nhập liệu
        x = X_entry.get()
        y = Y_entry.get()

        # Tọa độ điểm 
        entry_point=(float(x),float(y))

        # Thêm các điểm vào mảng
        points.append(entry_point)

        # Hiển thị các giá trị trong bảng
        tree.delete(*tree.get_children())
        for i, point in enumerate(points):
            index = i
            x = point[0]
            y = point[1]
            # Thêm dữ liệu vào cây và gắn đường kẻ cho từng cột
            if index % 2 == 0:
                tree.insert("", "end", values=(index, x, y), tags=('evenrow',))
            else:
                tree.insert("", "end", values=(index, x, y), tags=('oddrow',))

        # Cấu hình lại màu nền cho các hàng
        tree.tag_configure("evenrow", background="white")
        tree.tag_configure("oddrow", background="lightblue")
        print(points)

    # Tạo nút SET điểm
    set_btn = tk.Button(popup,text='Set',bg='white',command=set_click)
    set_btn.place(x=390,y=40,height=30,width=50)

    # Hàm cho nút Delete
    def delete_click():
        global points
        if points:
            points.pop()

        # Hiển thị các giá trị trong bảng
        tree.delete(*tree.get_children())
        for i, point in enumerate(points):
            index = i 
            x = point[0]
            y = point[1]
        # Thêm dữ liệu vào cây và gắn đường kẻ cho từng cột
            if index % 2 == 0:
                tree.insert("", "end", values=(index, x, y), tags=('evenrow',))
            else:
                tree.insert("", "end", values=(index, x, y), tags=('oddrow',))

        # Cấu hình lại màu nền cho các hàng
        tree.tag_configure("evenrow", background="white")
        tree.tag_configure("oddrow", background="lightblue")
    

    # Tạo nút Delete
    delete_btn = tk.Button(popup,text='Delete',bg='white',command=delete_click)
    delete_btn.place(x=390,y=80,height=30,width=50)

    # Hàm nút Clear
    def clear_click():
        global points
        points=[init]
        X_entry.delete(0, 'end')
        Y_entry.delete(0,'end')

        # Xóa tất cả các mục trong cây
        tree.delete(*tree.get_children())
    
    # Tạo nút Clear 
    clear_btn = tk.Button(popup,text= 'Clear',bg='white',command = clear_click)
    clear_btn.place(x=390,y=120,height=30,width=50)          
    
    popup.mainloop() 

# Tạo nút chế độ nhập tọa độ
type_mode = tk.Button(auto_frame,text ='Type mode',bg='white',state='disabled',command =type_click)
type_mode.place(x=10,y=20,height=30,width=100)
objects_4.append(type_mode)

arr_point = []
direction = b'T'
count = 1
def sketch_click():
    global arr_point,state,direction,count
    state = 1
    # tạo popup chế độ vẽ quỹ đạo đi mong muốn
    popup = tk.Toplevel()
    popup.title('Sketch mode')
    popup.resizable(height=False,width=False)
    popup.configure(bg=manu_color)
                
    # Tính toán vị trí của popup
    root_width = root.winfo_width()
    root_height = root.winfo_height()
    popup_width = 350
    popup_height = 320
    x = root.winfo_rootx() + (root_width - popup_width) // 2
    y = root.winfo_rooty() + (root_height - popup_height) // 2
    popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

    # Tạo canvas
    canvas = tk.Canvas(popup, width=150, height=300,bg='white',relief=tk.SUNKEN)
    canvas.place(x=10,y=10)

    # Biến lưu trữ tọa độ điểm cuối cùng
    prev_x = None
    prev_y = None

    def start_paint(event):
        global is_drawing, prev_x, prev_y
        is_drawing = True
        prev_x = event.x
        prev_y = event.y

    def draw_line(event):
        global prev_x, prev_y
        # Lấy tọa độ hiện tại của chuột
        if is_drawing:
            x = event.x
            y = event.y
            # Tìm điểm gần nhất cách chuột 5 pixel
            nearest_x = round(x / 1) * 1
            nearest_y = round(y / 1) * 1
            if prev_x != nearest_x or prev_y != nearest_y:
                # Vẽ đường thẳng từ điểm cuối cùng đến điểm gần nhất
                canvas.create_line(prev_x, prev_y, nearest_x, nearest_y, fill='green',width=4)
                arr_point.append((nearest_x,nearest_y))
            # Cập nhật tọa độ điểm cuối cùng
            prev_x = nearest_x
            prev_y = nearest_y

    def stop_paint(event):
        global is_drawing, prev_x, prev_y
        is_drawing = False
        prev_x = None
        prev_y = None

    # Vẽ các điểm cách nhau 50 pixel
    for x in range(0, 150, 10):
        for y in range(0, 300, 10):
            canvas.create_rectangle(x, y, x, y, width=3)
    
    # Function to clear the canvas
    def clear_canvas():
        global arr_point
        canvas.delete("all")
        # Vẽ các điểm cách nhau 50 pixel
        for x in range(0, 150, 10):
            for y in range(0, 300, 10):
                canvas.create_rectangle(x, y, x, y, width=3)
        arr_point.clear()

    # Create a Clear button
    clear_button = tk.Button(popup, text="Clear", bg='white',command=clear_canvas)
    clear_button.place(x=215,y=190,width=70,height=30)

    # Hàm cho nút reverse trong chế độ sketch
    
    def switch_click():
        global count,direction
        count += 1
        if (count%2) == 0:
            direction = b'L'
            t_status.configure(bg='red')
            l_status.configure(bg='green')
        else:
            direction = b'T'
            t_status.configure(bg='green')
            l_status.configure(bg='red')

    # Create a Reverse button
    switch_button = tk.Button(popup,text = 'Switch',bg = 'white',command = switch_click)
    switch_button.place(x=215,y=230,width=70,height=30)

    # Status notification
    t_status_text = tk.Label(popup,text='T',bg=manu_color)
    t_status_text.place(x=195,y=265)
    t_status = tk.Label(popup,relief=tk.SUNKEN,bg='green')
    t_status.place(x=215,y=270,width=32,height=15)

    l_status_text = tk.Label(popup,text='L',bg=manu_color)
    l_status_text.place(x=290,y=265)
    l_status = tk.Label(popup,relief=tk.SUNKEN,bg='red')
    l_status.place(x=253,y=270,width=32,height=15)
    
    # Thiết lập sự kiện khi kéo chuột
    canvas.bind("<Button-1>", start_paint)
    canvas.bind("<B1-Motion>", draw_line)
    canvas.bind("<ButtonRelease-1>", stop_paint)

    compass_draw = tk.Canvas(popup, width=170, height=170, bg=manu_color)
    compass_draw.place(x=170, y=10)

    image = Image.open('compass.png')
    # Resize the image to fit the canvas
    image = image.resize((150, 150), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    # Calculate the position to center the image on the canvas
    x_pos = (compass_draw.winfo_reqwidth() - image.width) // 2
    y_pos = (compass_draw.winfo_reqheight() - image.height) // 2

    compass_draw.create_image(x_pos, y_pos, image=photo, anchor=tk.NW)
    compass_draw.place(x=170, y=10)

    popup.mainloop() 

#Tạo nút cho chế độ vẽ quỹ đạo mong muốn
sketch_mode = tk.Button(auto_frame,text ='Sketch mode',bg='white',state='disabled',command=sketch_click)
sketch_mode.place(x=10,y=70,height=30,width=100)
objects_4.append(sketch_mode)

''' tính toán quãng đường từ các tọa độ điểm chế độ nhập'''
def type_calculate_total_length(arr):
    total_length = 0
    length_list = []
    for i in range(len(arr) - 1):
        a1 = np.array(arr[i])
        a2 = np.array(arr[i+1])
        direction_A = a2 - a1
        length = np.linalg.norm(direction_A)
        total_length += length
        length_list.append(total_length)
      
    return total_length, length_list
'''-----ooo-----'''
def sketch_calculate_total_length(arr):
    total_length = 0
    length_list = []
    for i in range(len(arr) - 1):
        a1 = np.array(arr[i])
        a2 = np.array(arr[i+1])
        direction_A = a2 - a1
        length = np.linalg.norm(direction_A)
        total_length += length*0.1
        length_list.append(total_length)

    return total_length, length_list

''' tính toán góc quay giữa các tọa độ điểm'''
def type_calculate_total_angle(arr):
    angle = 0
    angle_array = []
    desired_array = []
    if len(arr) < 2:
        return angle_array

    a1 = np.array(arr[0])
    a2 = np.array(arr[1])

    # Tính góc ban đầu của xe 
    direction_A = [arr[0][0],1]
    direction_B = a2 - a1
    normalized_direction_A = direction_A / np.linalg.norm(direction_A)
    normalized_direction_B = direction_B / np.linalg.norm(direction_B)
    angle_first = np.arccos(np.dot(normalized_direction_A, normalized_direction_B))
    angle_first_deg = np.degrees(angle_first)
    if direction_A[0] * direction_B[1] - direction_A[1] * direction_B[0] >= 0:
        angle_array.append(angle_first_deg) 
    else:
        angle_array.append(-angle_first_deg) 

    for i in range(len(arr) - 2):

        # Chuyển đổi các vector thành dạng numpy array
        a1 = np.array(arr[i])
        a2 = np.array(arr[i+1])
        a3 = np.array(arr[i+2])

        # Tính toán vector hướng của đường thẳng A và B
        direction_A = a2 - a1
        direction_B = a3 - a2

        # Chuẩn hóa vector hướng
        normalized_direction_A = direction_A / np.linalg.norm(direction_A)
        normalized_direction_B = direction_B / np.linalg.norm(direction_B)

        # Tính toán góc giữa hai vector
        angle_rad = np.arccos(np.dot(normalized_direction_A, normalized_direction_B))
        angle_deg = np.degrees(angle_rad)
        if direction_A[0] * direction_B[1] - direction_A[1] * direction_B[0] >= 0:
            angle_array.append(angle_deg) 
        else:
            angle_array.append(-angle_deg) 

    for j in range(len(angle_array)):
        angle += angle_array[j]
        desired_array.append(angle)

    return angle_array,desired_array

# Tính góc khi chạy tiến
def T_sketch_calculate_total_angle(arr):
    angle = 0
    angle_array = []
    desired_array = []
    if len(arr) < 2:
        return angle_array

    a1 = np.array(arr[0])
    a2 = np.array(arr[1])

    # Tính góc ban đầu của xe 
    direction_A = [1,arr[0][0]]
    direction_B = a1 - a2
    normalized_direction_A = direction_A / np.linalg.norm(direction_A)
    normalized_direction_B = direction_B / np.linalg.norm(direction_B)
    angle_first = np.arccos(np.dot(normalized_direction_A, normalized_direction_B))
    angle_first_deg = np.degrees(angle_first)
    if direction_A[1] * direction_B[0] - direction_A[0] * direction_B[1] >= 0:
        angle_array.append(angle_first_deg) 
    else:
        angle_array.append(-angle_first_deg) 

    for i in range(len(arr) - 2):

        # Chuyển đổi các vector thành dạng numpy array
        a1 = np.array(arr[i])
        a2 = np.array(arr[i+1])
        a3 = np.array(arr[i+2])

        # Tính toán vector hướng của đường thẳng A và B
        direction_A = a1 - a2
        direction_B = a2 - a3

        # Chuẩn hóa vector hướng
        normalized_direction_A = direction_A / np.linalg.norm(direction_A)
        normalized_direction_B = direction_B / np.linalg.norm(direction_B)

        # Tính toán góc giữa hai vector
        angle_rad = np.arccos(np.dot(normalized_direction_A, normalized_direction_B))
        angle_deg = np.degrees(angle_rad)
        if direction_A[1] * direction_B[0] - direction_A[0] * direction_B[1] >= 0:
            angle_array.append(angle_deg) 
        else:
            angle_array.append(-angle_deg) 

    for j in range(len(angle_array)):
        angle += angle_array[j]
        desired_array.append(angle)

    return angle_array,desired_array

# Tính góc khi chạy lùi
def L_sketch_calculate_total_angle(arr):
    angle = 0
    angle_array = []
    desired_array = []
    if len(arr) < 2:
        return angle_array

    a1 = np.array(arr[0])
    a2 = np.array(arr[1])

    # Tính góc ban đầu của xe 
    direction_A = [1,arr[0][0]]
    direction_B = a2 - a1
    normalized_direction_A = direction_A / np.linalg.norm(direction_A)
    normalized_direction_B = direction_B / np.linalg.norm(direction_B)
    angle_first = np.arccos(np.dot(normalized_direction_A, normalized_direction_B))
    angle_first_deg = np.degrees(angle_first)
    if direction_A[0] * direction_B[1] - direction_A[1] * direction_B[0] >= 0:
        angle_array.append(angle_first_deg) 
    else:
        angle_array.append(-angle_first_deg) 

    for i in range(len(arr) - 2):

        # Chuyển đổi các vector thành dạng numpy array
        a1 = np.array(arr[i])
        a2 = np.array(arr[i+1])
        a3 = np.array(arr[i+2])

        # Tính toán vector hướng của đường thẳng A và B
        direction_A = a2 - a1
        direction_B = a3 - a2

        # Chuẩn hóa vector hướng
        normalized_direction_A = direction_A / np.linalg.norm(direction_A)
        normalized_direction_B = direction_B / np.linalg.norm(direction_B)

        # Tính toán góc giữa hai vector
        angle_rad = np.arccos(np.dot(normalized_direction_A, normalized_direction_B))
        angle_deg = np.degrees(angle_rad)
        if direction_A[0] * direction_B[1] - direction_A[1] * direction_B[0] >= 0:
            angle_array.append(angle_deg) 
        else:
            angle_array.append(-angle_deg) 

    for j in range(len(angle_array)):
        angle += angle_array[j]
        desired_array.append(angle)

    return angle_array,desired_array
'''-----ooo-----'''

# Hàm tam xuất giá trị
def map(inValue,  inMax,  inMin, outMax,  outMin ):

	if inValue > inMax: 
	
		return outMax
	
	elif inValue < inMin:

		return outMin

	else:

		return (inValue-inMin)*(outMax-outMin)/(inMax-inMin) + outMin

# Hàm điều khiển góc bánh trước dùng thuật Stanley Control
def stanley_control(desired_angle,current_angle,desired_length,actual_length,velocity,k_coef):
    psi = desired_angle-current_angle
    err = desired_length - actual_length
    e = abs(sqrt(err**2 + (2.1**2)-(2*err*2.1*cos(current_angle))))
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
    
# Hàm thuật toán đánh lái stanley control của bánh tước theo tọa độ điểm   
def stanley_control_point(current_point,current_angle,target_point,target_angle,velocity,k_coef):
    psi = target_angle-current_angle
    front_wheel_point = [0,0] # tạo điểm lưu tọa độ của bánh trước
    front_wheel_point[0] = current_point[0] + sin(radians(-current_angle)) * 2.1 # giá trị trục X
    front_wheel_point[1] = current_point[1] + cos(radians(-current_angle)) * 2.1 # giá trị trục Y
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
    point = result[:,:2]/2
    angle = (result[:,2]*180/pi)-90
    direction = result[:,3]
    return point, angle, direction

# cờ chạy
run = False
'''điểu khiển bánh trước sau chạy auto'''
achieved_length = 0 # biển để lưu total length sau khi kết thúc
actual_dis_p = 0 # lưu giá trị thực tế sau khi kết thúc 
back_speed = 0 # giá trị tốc độ của bánh sau
displaced_angle = 0 # giá trị góc mong muốn
desired_angle = 0 # góc mong muốn của xe
target_point = 0 # khoảng cách điểm mong muốn
def car_auto_control():
    global achieved_length,actual_dis_p,init
    global run,back_speed,counter,state
    global displaced_angle,desired_angle
    global points,arr_point,direction
    global target_point

    if state ==0:
        required_length, length_list = type_calculate_total_length(points)
        change_angle_list,desired_angle_list = type_calculate_total_angle(points)
    elif state==1:
        required_length, length_list = sketch_calculate_total_length(arr_point)
        if direction ==b'T':
            change_angle_list,desired_angle_list = T_sketch_calculate_total_angle(arr_point)
        elif direction == b'L':
            change_angle_list,desired_angle_list = L_sketch_calculate_total_angle(arr_point)

    total_length = achieved_length + required_length

    if run:
        if abs(actual_dis)<total_length:
            step = 0
            for i in range(len(length_list)):
                if actual_dis< (actual_dis_p + length_list[i])-2.1:
                    step = i
                    break
            desired_angle = desired_angle_list[step]
            displaced_angle = change_angle_list[step] 
            target_point = actual_dis_p + length_list[step]
            
            if (actual_dis>= actual_dis_p) and (actual_dis<= actual_dis_p+2):
                back_speed = 50 # chạy tốc 50 với 2m đầu
            elif (actual_dis>actual_dis_p+2) and (actual_dis<=total_length-1):
                if (actual_angle-desired_angle>=-1) and (actual_angle-desired_angle<=1):
                    back_speed = 40 # nếu  xe đi thẳng chạy tốc 40
                else:
                    back_speed = 35 #  nếu xe rẽ chạy tốc 35
            else: 
                back_speed = 0

        else:
            achieved_length = total_length # lưu lại tổng độ dài quãng đường đi được
            actual_dis_p = actual_dis # lưu lại quãng đường thực tế
            init = points[len(points)-1] # lưu tọa độ kết thúc của 1 chu trình
            points =[init] 
            run = False

        # UART cho bánh trước
        front_pulse =str(int(stanley_control(desired_angle,actual_angle,target_point,actual_dis,actual_vel,0.1)))
        front_speed = chr(75)
        f_uart_data = f_start_bit + front_speed.encode('utf-8') + front_pulse.encode('utf-8') + stop_bit
        f_uart.write(f_uart_data)

        #UART cho bánh sau
        b_speed_str = str(back_speed)
        b_uart_data = b_start_bit + direction + b_speed_str.encode('utf-8') + stop_bit
        b_uart.write(b_uart_data)

        print(length_list)

    root.after(50, car_auto_control)
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
go_btn.place(x=130, y=20 ,height=85,width=85)
objects_4.append(go_btn)





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
emer_button.place(x=230, y=10, width=100, height=100)
objects_4.append(emer_button)

''' Chức năng manual'''
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
    print(valuess)
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

''' Code để vẽ đồ thị '''
# Khởi tạo đồ thị
fig = Figure()

# Vẽ theo góc quay xe
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Car Trajectory')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_xlim(-10,10)
ax1.set_ylim(-30,60)
ax1.grid(True)      
line1, = ax1.plot([], [], 'g')

# Khởi tạo canvas để hiển thị đồ thị
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().place(x=560, y=10, height=690, width=750)

# Điều chỉnh vị trí của đồ thị
fig.tight_layout()

# Biến cờ cho đồ thị
graph_run = False

# Biến x,y zero
x_0= y_0 = 0

# Mảng x,y
x_s=[]
y_s=[]

# giá trị dis trước
a_p = 0



def update_plot():
    global graph_run,state
    global x_0,y_0
    global a_p

    # tính toán các khoản độ dài
    if state ==0:
        required_length, length_list = type_calculate_total_length(points)
    elif state==1:
        required_length, length_list = sketch_calculate_total_length(arr_point)

    if graph_run:
        
        # Biến x, y 
        x_1,y_1 = calulation_next_point(x_0,y_0,actual_dis-a_p,actual_angle)

        # Lưu các giá trị tính vô mảng
        x_s.append(x_1)
        y_s.append(y_1)

        # Lưu lại giá trị đã tính
        x_0,y_0 = x_1,y_1
        a_p = actual_dis

        # Xóa dữ liệu cũ trên đồ thị
        ax1.clear()

        # Vẽ đồ thị
        ax1.scatter(x_s, y_s, color='g')
        ax1.set_title('Car Trajectory')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_xlim(-10,10)
        ax1.set_ylim(-30,60)
        ax1.grid(True)
            
        # Cập nhật đồ thị
        canvas.draw()

    # Gọi lại hàm update_plot mỗi 50ms
    root.after(50, update_plot)
    
# Hàm nút Start
def start_click():
    global graph_run
    graph_run = True
    # phân luồng để vẽ đồ thị
    threading.Thread(target = update_plot).start()
       
# Nút start vẽ đồ thị 
start_btn = tk.Button(root, text = 'Start',bg='white',command = start_click,state='disabled')
start_btn.place(x=460,y = 500,height=30,width=80)
objects_2.append(start_btn)

def stop_click():
    global run
    run = False

# Nút Stop vẽ đồ thị
stop_btn = tk.Button(root,text ='Stop',bg='white',command=stop_click,state='disabled')
stop_btn.place(x=460,y=540,height=30,width=80 )
objects_2.append(stop_btn)

# Chạy vòng lặp giao diện
root.mainloop()