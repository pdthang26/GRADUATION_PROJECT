import tkinter as tk
import numpy as np
arr_point = []


root = tk.Tk()
root.title("Paint App")

canvas = tk.Canvas(root, bg="white", width=400, height=400)
canvas.pack()

pen_color = "black"
pen_width = 2
is_drawing = False
prev_x = None
prev_y = None

def start_paint(event):
    global is_drawing, prev_x, prev_y
    is_drawing = True
    prev_x = event.x
    prev_y = event.y



def paint(event):
    global is_drawing, prev_x, prev_y
    if is_drawing:
        x, y = event.x, event.y
        if prev_x and prev_y:
            canvas.create_line(prev_x, prev_y, x, y, fill=pen_color, width=pen_width)
            arr_point.append((x,y))
        prev_x = x
        prev_y = y
    
    
    
    

def stop_paint(event):
    global is_drawing, prev_x, prev_y
    is_drawing = False
    prev_x = None
    prev_y = None


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

canvas.bind("<Button-1>", start_paint)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", stop_paint)

root.mainloop()


print(arr_point)
total,_,increa = calculate_total_length(arr_point)
angle_array,desired = calculate_total_angle(arr_point)
print (total, increa)
print (angle_array)
print (desired)


