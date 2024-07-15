import socket

def get_ip_address():
    hostname = socket.gethostname()  # Lấy tên máy tính
    ip_address = socket.gethostbyname(hostname)  # Lấy địa chỉ IP từ tên máy tính
    return ip_address

# Ví dụ sử dụng
ip_address = get_ip_address()
print(f"Địa chỉ IP của máy tính hiện tại: {ip_address}")