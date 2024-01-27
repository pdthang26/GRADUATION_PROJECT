import cv2
import numpy as np

# Tạo một hình ảnh đen trắng
image = np.zeros((500, 500), dtype=np.uint8)

# Tọa độ của hình vẽ gốc
original_coordinates = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])

# Tạo một hình vẽ mới bằng cách thêm offset về phía bên trái
offset_left = 50
new_coordinates_left = original_coordinates + np.array([-offset_left, 0])
cv2.polylines(image, [new_coordinates_left], isClosed=True, color=255, thickness=2)

# Tạo một hình vẽ mới bằng cách thêm offset về phía bên phải
offset_right = 50
new_coordinates_right = original_coordinates + np.array([offset_right, 0])
cv2.polylines(image, [new_coordinates_right], isClosed=True, color=255, thickness=2)

# Hiển thị hình ảnh
cv2.imshow('Image with Offset Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()