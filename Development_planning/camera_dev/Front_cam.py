import cv2
import numpy as np

cv2.namedWindow("Camera")

global angle 
angle = 38
def mirror_draw (width, arr):
    arr_mirror_point = []
    for i in range(len(arr)):
        point_temp =(width - arr[i][0],arr[i][1]) 
        arr_mirror_point.append(point_temp)
    return arr_mirror_point

def lagrange_interpolation(data, x):
    n = len(data)
    result = 0
    term  = 0
    for i in range(n):
        xi, yi = data[i]

        # Tính giá trị của hàm Lagrange cho điểm x
        term = yi
        for j in range(n):
            if j != i:
                xj, _ = data[j]
                term *= (x - xj) / (xi - xj)
        result += term

    return int(result)
def four_point_to_curve(arr):
    curve = []
    for t in np.arange(0, 1, 0.01):
        x = int((1 - t) ** 3 * arr[0][0] + 3 * (1 - t) ** 2 * t * arr[1][0] + 3 * (1 - t) * t ** 2 * arr[2][0] + t ** 3 * arr[3][0])
        y = int((1 - t) ** 3 * arr[0][1] + 3 * (1 - t) ** 2 * t * arr[1][1] + 3 * (1 - t) * t ** 2 * arr[2][1] + t ** 3 * arr[3][1])
        curve.append([x, y])
    return curve

def calculating_four_point_from_angle(angle,arr_1, arr_2 , arr_3 ,arr_4):
    max_arr = [arr_1[3][0]-arr_1[0][0] , arr_2[3][0]-arr_2[0][0] , arr_3[3][0]-arr_3[0][0] ,arr_4[3][0]-arr_4[0][0]]
    arr_in = [arr_1,arr_2,arr_3,arr_4]
    arr_out = []
    for i in range(len(max_arr)):
        x = int((1-angle/38)* max_arr[i]+ arr_in[i][0][0])
        y = lagrange_interpolation(arr_in[i], x)
        arr_out.append([x,y])
    return arr_out
    
        


    


def draw_grid(image, grid_size, color=(110, 0, 0), thickness=1):
    height, width = image.shape[:2]
    cell_width = width // grid_size[1]
    cell_height = height // grid_size[0]

    # Vẽ đường dọc
    for i in range(1, grid_size[1]):
        x = i * cell_width
        cv2.line(image, (x, 0), (x, height), color, thickness)

    # Vẽ đường ngang
    for i in range(1, grid_size[0]):
        y = i * cell_height
        cv2.line(image, (0, y), (width, y), color, thickness)


camera = cv2.VideoCapture(0)
ret, frame = camera.read()
height, width, _ = frame.shape

print(frame.shape)

bottom_left = (int(width/2)-210, height)  # Điểm bottom-left nằm ở góc dưới bên trái

convergent_center = (int(width / 2), int(height / 2))  # Điểm top-center nằm ở giữa phía trên
up_left = (int(width/2)-20, bottom_left[1]+int((convergent_center[1]-bottom_left[1])*((int(width/2)-20)-bottom_left[0])
                                                                                /(convergent_center[0]-bottom_left[0])))


arr_point_1_left=[[up_left[0]-250, up_left[1]+40],[up_left[0]-200, up_left[1]+30], [up_left[0]-100, up_left[1]+10],[up_left[0],up_left[1]]]
arr_point_2_left=[[up_left[0]-200, up_left[1]+55],[up_left[0]-160, up_left[1]+50], [up_left[0]-85, up_left[1]+45], [up_left[0]-30,up_left[1]+38] ]
arr_point_3_left=[[up_left[0]-170, up_left[1]+120], [up_left[0]-145, up_left[1]+100], [up_left[0]-90, up_left[1]+80],[up_left[0]-58,up_left[1]+70]]
arr_point_4_left=[[bottom_left[0]-20,bottom_left[1]], [bottom_left[0]-15,bottom_left[1]], [bottom_left[0]-10,bottom_left[1]],[bottom_left[0],bottom_left[1]]]

arr1 = mirror_draw(width,arr_point_1_left )
arr2 = mirror_draw(width,arr_point_2_left )
arr3 = mirror_draw(width,arr_point_3_left )
arr4 = mirror_draw(width,arr_point_4_left )

arr_point_5_left=[[up_left[0]+260,up_left[1]+10],     [up_left[0]+190,up_left[1]-10],     [up_left[0]+90,up_left[1]-40],      [up_left[0],up_left[1]]]
arr_point_6_left=[[up_left[0]+210,up_left[1]+15],     [up_left[0]+165,up_left[1]],        [up_left[0]+80,up_left[1]-35],      [up_left[0]-30,up_left[1]+38] ]
arr_point_7_left=[[up_left[0]+160,up_left[1]+30],     [up_left[0]+135,up_left[1]+10],     [up_left[0]+70,up_left[1]-30],      [up_left[0]-58,up_left[1]+70]]
arr_point_8_left=[[bottom_left[0]+20,bottom_left[1]], [bottom_left[0]+15,bottom_left[1]], [bottom_left[0]+10,bottom_left[1]], [bottom_left[0],bottom_left[1]]]



while True:


    # Đọc khung hình từ camera
    ret, frame = camera.read()
    height, width, _ = frame.shape

    # Vẽ vùng an toàn
    # Lấy kích thước của khung hình
    
    

    # Tạo mảng các điểm tạo nên tam giác
    points_left = np.array([bottom_left,  up_left])
    cv2.polylines(frame, [points_left], 0, (0, 255, 0), 2)
    point_right = np.array(mirror_draw(width,points_left))
    cv2.polylines(frame, [point_right], 0, (0, 255, 0), 2)

    # cv2.circle(frame,arr1[0],1,(0, 255, 0),2)

    # cv2.circle(frame,arr1[0],1,(0, 255, 0),2)
    # cv2.circle(frame,arr2[0],1,(0, 255, 0),2)
    # cv2.circle(frame,arr3[0],1,(0, 255, 0),2)
    # cv2.circle(frame,arr4[0],1,(0, 255, 0),2)

    # cv2.circle(frame,arr1[1],1,(255, 0, 0),2)
    # cv2.circle(frame,arr2[1],1,(255, 0, 0),2)
    # cv2.circle(frame,arr3[1],1,(255, 0, 0),2)
    # cv2.circle(frame,arr4[1],1,(255, 0, 0),2)

    # cv2.circle(frame,arr1[2],1,(255, 255, 0),2)
    # cv2.circle(frame,arr2[2],1,(255, 255, 0),2)
    # cv2.circle(frame,arr3[2],1,(255, 255, 0),2)
    # cv2.circle(frame,arr4[2],1,(255, 255, 0),2)

    # cv2.circle(frame,arr1[3],1,(0, 255, 255),2)
    # cv2.circle(frame,arr2[3],1,(0, 255, 255),2)
    # cv2.circle(frame,arr3[3],1,(0, 255, 255),2)
    # cv2.circle(frame,arr4[3],1,(0, 255, 255),2)

    # cv2.circle(frame,arr_point_5_left[0],1,(0, 255, 0),2)
    # cv2.circle(frame,arr_point_6_left[0],1,(0, 255, 0),2)
    # cv2.circle(frame,arr_point_7_left[0],1,(0, 255, 0),2)
    # cv2.circle(frame,arr_point_8_left[0],1,(0, 255, 0),2)

    # cv2.circle(frame,arr_point_5_left[1],1,(255, 0, 0),2)
    # cv2.circle(frame,arr_point_6_left[1],1,(255, 0, 0),2)
    # cv2.circle(frame,arr_point_7_left[1],1,(255, 0, 0),2)
    # cv2.circle(frame,arr_point_8_left[1],1,(255, 0, 0),2)

    # cv2.circle(frame,arr_point_5_left[2],1,(255, 255, 0),2)
    # cv2.circle(frame,arr_point_6_left[2],1,(255, 255, 0),2)
    # cv2.circle(frame,arr_point_7_left[2],1,(255, 255, 0),2)
    # cv2.circle(frame,arr_point_8_left[2],1,(255, 255, 0),2)

    # cv2.circle(frame,arr_point_5_left[3],1,(0, 255, 255),2)
    # cv2.circle(frame,arr_point_6_left[3],1,(0, 255, 255),2)
    # cv2.circle(frame,arr_point_7_left[3],1,(0, 255, 255),2)
    # cv2.circle(frame,arr_point_8_left[3],1,(0, 255, 255),2)
    
    if angle > 0:
        curve_left = calculating_four_point_from_angle(angle,arr_point_1_left,arr_point_2_left,arr_point_3_left,arr_point_4_left)
        curve_left = np.array(four_point_to_curve(curve_left))
        curve_right = calculating_four_point_from_angle(angle,arr_point_5_left,arr_point_6_left,arr_point_7_left,arr_point_8_left)
        curve_right = np.array(mirror_draw(width,four_point_to_curve(curve_right)))
    elif angle <0:
        curve_left = calculating_four_point_from_angle(-angle,arr_point_1_left,arr_point_2_left,arr_point_3_left,arr_point_4_left)
        curve_left = np.array(mirror_draw(width,four_point_to_curve(curve_left))) 
        curve_right = calculating_four_point_from_angle(-angle,arr_point_5_left,arr_point_6_left,arr_point_7_left,arr_point_8_left)
        curve_right = np.array(four_point_to_curve(curve_right))
    
    cv2.polylines(frame, [curve_right], 0, (255, 255, 0), 2)
    cv2.polylines(frame, [curve_left], 0, (255, 255, 0), 2)

    # Vẽ tam giác lên khung hình


    # Hiển thị khung hình
    cv2.imshow("Camera", frame)
    # if angle <=-38:
    #     flag = 1
    # elif angle >=38:
    #     flag = 0
    # if flag == 1:
    #     angle +=1
    # elif flag ==0:
    #     angle -=1
    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ hiển thị
camera.release()
cv2.destroyAllWindows()