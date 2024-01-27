import tkinter as tk
import numpy as np

# Tạo cửa sổ giao diện
root = tk.Tk()
root.title("Vẽ đường thẳng")
root.geometry("600x600")

# Tạo canvas
canvas = tk.Canvas(root, width=400, height=400,bg='white',relief=tk.SUNKEN)
canvas.pack()

# Biến lưu trữ tọa độ điểm cuối cùng
prev_x = None
prev_y = None

def start_paint(event):
    global is_drawing, prev_x, prev_y
    is_drawing = True
    prev_x = event.x
    prev_y = event.y

arr_point = []
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
            canvas.create_line(prev_x, prev_y, nearest_x, nearest_y, fill='red',width =5)
            arr_point.append((nearest_x,nearest_y))
        # Cập nhật tọa độ điểm cuối cùng
        prev_x = nearest_x
        prev_y = nearest_y

def stop_paint(event):
    global is_drawing, prev_x, prev_y
    is_drawing = False
    prev_x = None
    prev_y = None

# Vẽ các điểm cách nhau 10 pixel
for x in range(0, 400, 10):
    for y in range(0, 400, 10):
        canvas.create_rectangle(x, y, x, y, width=4)

# Function to clear the canvas
def clear_canvas():
    global arr_point
    canvas.delete("all")
    # Vẽ các điểm cách nhau 10 pixel
    for x in range(0, 400, 10):
        for y in range(0, 400, 10):
            canvas.create_rectangle(x, y, x, y, width=4)
    arr_point.clear()

# Create a Clear button
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

def calculate_total_length(arr):
    total_length = 0
    length_list=[]
    length_incremental = []
    for i in range(len(arr) - 1):
        a1 = np.array(arr[i])
        a2 = np.array(arr[i+1])
        direction_A = a2 - a1
        length = np.linalg.norm(direction_A)
        total_length += length*0.1
        length_incremental.append(total_length) 
        length_list.append(length)
    return total_length, length_list, length_incremental

def calculate_total_angle(arr):
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

# Thiết lập sự kiện khi kéo chuột
canvas.bind("<Button-1>", start_paint)
canvas.bind("<B1-Motion>", draw_line)
canvas.bind("<ButtonRelease-1>", stop_paint)

# Chạy chương trình
root.mainloop()
total,_,increa = calculate_total_length(arr_point)
print(total)