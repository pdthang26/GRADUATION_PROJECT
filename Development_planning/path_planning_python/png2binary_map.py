import cv2
import numpy as np

map = cv2.imread('globalMap.png', cv2.IMREAD_GRAYSCALE)
map[map==255] = int(1)
np.savetxt('globalMap.csv', map, delimiter=',')
