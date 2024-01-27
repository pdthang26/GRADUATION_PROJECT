import matplotlib.pyplot as plt
import numpy as np

# Tạo dữ liệu cho đường tròn
radius = 2.67
center = (1, 1)  # Tọa độ tâm đường tròn
center_2 = (1,center[1]+2*radius)

# Tạo một mảng góc từ 0 đến 2*pi với số điểm là 100
theta = np.linspace(np.pi/2,np.pi, 100)

# Tính toán tọa độ x và y của các điểm trên đường tròn
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)


# Vẽ đường tròn
plt.plot(x, y)
theta = np.linspace(np.pi*3/2,2*np.pi, 100)
x = center_2[0] + radius * np.cos(theta)
y = center_2[1] + radius * np.sin(theta)
plt.plot(x, y)

# Đặt giới hạn trục x và y để đảm bảo đường tròn được hiển thị đầy đủ
plt.xlim(center[0] - radius - 1, center[0] + radius + 1)
plt.ylim(center[1] - radius - 1, center[1] + radius + 1)

# Đặt tên cho trục x và y
plt.xlabel('X')
plt.ylabel('Y')

# Đặt tiêu đề cho đồ thị
plt.title('Đường tròn')
plt.grid(True)
plt.axis('equal')
# Hiển thị đồ thị
plt.show()