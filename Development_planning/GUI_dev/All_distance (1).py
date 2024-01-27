import numpy as np
import matplotlib.pyplot as plt

def plot_lines_and_angle(a1, a2, a3, ax):
    # Vẽ đường thẳng A
    ax.plot([a1[0], a2[0]], [a1[1], a2[1]], 'b-', label='A')

    # Vẽ đường thẳng B
    ax.plot([a2[0], a3[0]], [a2[1], a3[1]], 'r-', label='B')

    # Tính toán và vẽ góc giữa hai đường thẳng
        # Chuyển đổi các vector thành dạng numpy array
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)

    # Tính toán vector hướng của đường thẳng A và B
    direction_A = a2 - a1
    direction_B = a3 - a2

    # Chuẩn hóa vector hướng
    normalized_direction_A = direction_A / np.linalg.norm(direction_A)
    normalized_direction_B = direction_B / np.linalg.norm(direction_B)

    # Tính toán góc giữa hai vector
    angle_rad = np.arccos(np.dot(normalized_direction_A, normalized_direction_B))
    angle_deg = np.degrees(angle_rad)
    if direction_A[0] * direction_B[1] - direction_A[1] * direction_B[0] > 0:
        angle_deg = angle_deg
    else:
        angle_deg = -angle_deg
    angle_text = f'Angle: {angle_deg:.2f} degrees'
    ax.text(a2[0], a2[1], angle_text, fontsize=10, ha='center')

def calculate_total_length(arr):
    total_length = 0
    length_list=[]
    length_incremental = []
    for i in range(len(arr) - 1):
        a1 = np.array(arr[i])
        a2 = np.array(arr[i+1])
        direction_A = a2 - a1
        length = np.linalg.norm(direction_A)
        total_length += length
        length_incremental.append(total_length) 
        length_list.append(length)
    return total_length,length_list,length_incremental

def calculate_total_angle(arr):
    angle_array = []
    angle = 0
    desired = []
    
    a1 = np.array(arr[0])
    a2 = np.array(arr[1])

    direction_A = [arr[0][0],1]
    direction_B = a2 - a1
    normalized_direction_A = direction_A / np.linalg.norm(direction_A)
    normalized_direction_B = direction_B / np.linalg.norm(direction_B)
    angle_first = np.arccos(np.dot(normalized_direction_A, normalized_direction_B))
    angle_first_deg = np.degrees(angle_first)
    if direction_A[0] * direction_B[1] - direction_A[1] * direction_B[0] > 0:
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
        if direction_A[0] * direction_B[1] - direction_A[1] * direction_B[0] > 0:
            angle_array.append(angle_deg) 
        else:
            angle_array.append(-angle_deg) 

    for j in range(len(angle_array)):
        angle += angle_array[j]
        desired.append(angle)
        
    return angle_array,desired


    
# Ví dụ sử dụng:
a0 = (0, 0)
a1 = (0, 10)
a2 = (5.4, 15.4)
a3 = (5.4, 25.4)
a4 = (0,25.4)
a5 = (0,0)



arr = [a0,a1,a2,a3,a4,a5]

fig, ax = plt.subplots()

a,b,e = calculate_total_length(arr)
c,d = calculate_total_angle(arr)

for i in range(len(arr) - 2):
    plot_lines_and_angle(arr[i], arr[i+1], arr[i+2], ax)
print(arr)
print(f'Total length: {a:.2f}')
print(f'array of length: {b}')
print(f'array of change angle:{c} ')
print(f'array of dersired angle:{d}')
print(f'incremental length:{e}')


# Đặt tiêu đề và chú thích cho biểu đồ
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Lines and Angle')
ax.legend()

plt.grid(True)
plt.axis('equal')
plt.show()

