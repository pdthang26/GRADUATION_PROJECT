
import cv2
import numpy as np

map = cv2.imread('map.png', cv2.IMREAD_GRAYSCALE)

# Hàm callback khi chuột được click
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(map, (x, y), 0, 255, -1)


# Tạo một cửa sổ OpenCV và thiết lập hàm callback
cv2.namedWindow('Draw Circle',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Draw Circle', 500, 500)
cv2.setMouseCallback('Draw Circle', draw_circle)

while True:
    cv2.imshow('Draw Circle', map)

    # Nhấn phím 'ESC' để thoát
    if cv2.waitKey(1) == 27:
        break
map_reshaped = map.reshape(40, 40)
map_reshaped[map_reshaped == 255] = 1

print(map_reshaped)

np.savetxt('map2.csv', map_reshaped, delimiter=',')
cv2.destroyAllWindows()