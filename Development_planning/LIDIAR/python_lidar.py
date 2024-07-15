import serial
import matplotlib.pyplot as plt
import numpy as np
import math

class LidarReader:
    def __init__(self, port, baudrate=115200):
        ser = serial.Serial(port, baudrate)
        angle_increment = math.pi / 180  # Chuyển đổi góc sang radian

    def read_data(self):
        """Đọc dữ liệu từ LiDAR và trả về các điểm."""
        try:
            data = ser.readline().decode().strip()
            distance_data = [int(x) for x in data.split(',')]
            return distance_data
        except Exception as e:
            print(f"Error reading data: {e}")
            return []

    def generate_points(self, distance_data):
        """Chuyển đổi dữ liệu khoảng cách thành tọa độ điểm x, y."""
        points = []
        for i, distance in enumerate(distance_data):
            angle = angle_increment * i
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            points.append((x, y))
        return points

    def plot_data(self, points):
        """Vẽ bản đồ điểm dựa trên danh sách các điểm."""
        if not points:
            return
        x_vals, y_vals = zip(*points)
        plt.figure()
        plt.scatter(x_vals, y_vals, s=1)
        plt.axis('equal')
        plt.show()

    def run(self):
        """Chạy đọc và vẽ điểm."""
        while True:
            try:
                distance_data = read_data()
                if distance_data:
                    points = generate_points(distance_data)
                    plot_data(points)
            except KeyboardInterrupt:
                print("Dừng đọc và vẽ điểm.")
                break
            except Exception as e:
                print(f"Error: {e}")

# Khởi tạo và chạy đọc LiDAR
if __name__ == "__main__":
    lidar = LidarReader(port='COM4')  # Thay đổi tên cổng COM phù hợp
    lidar.run()
