
import cv2
import numpy as np
from numba import jit
import time

map = cv2.imread('globalMap.png', cv2.IMREAD_GRAYSCALE)
print(map.shape)
# @jit(nopython=True)
def create_binary_map(map_array):
    binary_map = np.copy(map_array)
    # Avoid multi-dimensional indexing by iterating over each element
    for i in range(binary_map.shape[0]):
        for j in range(binary_map.shape[1]):
            if binary_map[i, j] > 1:
                binary_map[i, j] = 1
            else :
                binary_map[i, j] = 0
    return binary_map
    


# Hàm callback khi chuột được click
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(map, (x, y), 1, 255, -1)



# Tạo một cửa sổ OpenCV và thiết lập hàm callback
cv2.namedWindow('Draw Circle',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Draw Circle', 300, 751)
cv2.setMouseCallback('Draw Circle', draw_circle)

while True:
    
    start = time.time()
    out = create_binary_map(map)
    print(type(out))
    end = time.time()
    print("Elapsed (with compilation) = %f" % (end - start))
    cv2.imshow('Draw Circle', map)
    # Nhấn phím 'ESC' để thoát
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
np.savetxt('map22.csv', out, delimiter=',')
print(out)
