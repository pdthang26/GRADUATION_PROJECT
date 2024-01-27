import numpy as np 
import cv2
from math import *
from ultralytics import YOLO

#---------------------------------------------------------------
# Setup variable for camera
camera = cv2.VideoCapture(0)

ret, f = camera.read()
height, width, _ = f.shape

width_resize = 1920
height_resize = 1200

pts1 = np.float32([[0,0],[width,0],[0,height],[width,height]])
pts2 = np.float32([[0,0],[width_resize,0],[880,height_resize],[1100,height_resize]])
pts3 = np.float32([[0,0],[width_resize,0],[0,height_resize],[width_resize,height_resize]])

M_forward = cv2.getPerspectiveTransform(pts3,pts2)
M_inverse = cv2.getPerspectiveTransform(pts2,pts3)
M_resize  = cv2.getPerspectiveTransform(pts1,pts3)

object_detector = cv2.createBackgroundSubtractorMOG2(history = 150, varThreshold = 150)

# dữ liệu test quỹ đạo 
array_movement = [10.0, 17.91959594928933, 27.91959594928933, 33.51959594928933, 59.119595949289334]
array_length = [10.0, 7.919595949289332, 10.000000000000002, 5.6, 25.6]
change_angle =[-0.0, -45.0, 45.0, 90.0, 90.0]
array_angle = [0.0, -45.0, 0.0, 90.0, 180.0]
actual_distance = 0.0
actual_dis_p = 0
actual_angle = 0.0


# thêm model segmentation của YOLO
model_seg = YOLO('YOLO_pretrain_models\yolov8n-seg.pt')

# ------------------------------------------------------------------
width = 1920
height = 1200
fps = 24
output_video_path = "output_test"
        # Tạo đối tượng VideoWriter để lưu video (thay đổi tên và định dạng video nếu cần)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Chọn codec
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))   

def calulation_turning_point(start_point,distance_turn, angle_turn):
    pixel_per_miles = 30
    x = start_point[0]
    y = start_point[1]
    x_turn = sin(radians(angle_turn)) * distance_turn 
    y_turn = cos(radians(angle_turn)) * distance_turn 
    turning_point = [int (x-x_turn*pixel_per_miles), int(y- y_turn*pixel_per_miles)]
    
    return turning_point



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


def draw_trajectory (frame, array_movement_incremental, array_length , array_angle_dersired,change_angle_arr, actual_distance, actual_dis_p, actual_angle):
    step = 0
    start_point_left = [900,1200]
    start_point_right =[1090,1200]
    start_point_center = [995,1200]

    array_turning_point_left =[start_point_left] # mảng lưu giá trị pixel thời điểm xe quay 
    array_turning_point_right =[start_point_right] # mảng lưu giá trị pixel thời điểm xe quay 
    for i in range(len(array_movement_incremental)):
        if actual_distance< actual_dis_p + array_movement_incremental[i]:
            step = i
            break
    

    for count in range(step,len(array_movement_incremental)) :
        next_center_turning_point = calulation_turning_point(start_point_center, array_length[count], array_angle_dersired[count]-actual_angle )
        # next_turning_point_left = calulation_turning_point(start_point_left, array_length[count], array_angle[count]-actual_angle )
        # next_turning_point_right = calulation_turning_point(start_point_right, array_length[count], array_angle[count]-actual_angle )
        # alpha = change_angle_arr[count]
        # beta = array_angle_dersired[count]
        # if alpha>0:
        #      A= beta+(180-alpha)/2
        #      B= A-180 
        # elif alpha<0:
        #      B= -(abs(beta)+(180-abs(alpha))/2)
        #      A= B180 
        # else:
        #      A=90
        #      B=-A

        # print(A,B)
        # next_turning_point_left = calulation_turning_point(next_center_turning_point, 0.5, A )
        # next_turning_point_right = calulation_turning_point(next_center_turning_point, 0.5, B )
        # print(count,next_turning_point_left,next_turning_point_right)

        # cv2.line(frame,start_point_left,next_turning_point_left,22, 2 )
        # cv2.line(frame,start_point_right,next_turning_point_right,200, 2 )
        
        cv2.line(frame,start_point_center,next_center_turning_point,250, 20 )

        
        # start_point_left = next_turning_point_left
        # start_point_right = next_turning_point_right
        start_point_center = next_center_turning_point 


        # array_turning_point_left.append(next_turning_point_left)
        # array_turning_point_right.append(next_turning_point_right)

        
    return array_turning_point_left,array_turning_point_right


def Perspective_point(convesion_matrix,arr_point):
    arr_new_point = []
    for i in range(len(arr_point)):
        original_x = arr_point[i][0]
        original_y = arr_point[i][1]
        transformed_coords = np.dot(convesion_matrix, np.array([original_x, original_y, 1]))

        # Chia tọa độ x và y cho tọa độ z để chuẩn hóa
        transformed_x = transformed_coords[0] / transformed_coords[2]
        transformed_y = transformed_coords[1] / transformed_coords[2]

        new_point = [transformed_x,transformed_y]
        arr_new_point.append(new_point)
    return arr_new_point



    
def detect_use_MOG (frame_of_interest, object_detector):
    
    obstructing_objects = []
    
    mask = object_detector.apply(frame_of_interest)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        print(cnt)
        if area > 300:
            # cv2.drawContours(frame_of_interest, [cnt], -1, (0, 255, 0), 2)
            cv2.fillPoly(frame_of_interest, [cnt], (0, 255, 0), lineType=cv2.LINE_8)

            # x, y, w, h = cv2.boundingRect(cnt)
            # obstructing_objects.append([x, y, w, h])
            # cv2.rectangle(frame_of_interest, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('mask', 640, 480)
    cv2.imshow('mask',mask)
    return obstructing_objects

def detect_by_YOLO (image):
    #prediction processing
    predict = model_seg(image, device = '0',stream= True)
    for r in predict:
        masks  = r.masks.cpu()
    if masks is not None and masks.data is not None:  #nếu có predict được vật
        result_array = (masks.data.numpy() *255).astype("uint8")
        # image processing
        H, W = result_array.shape[1],result_array.shape[2]
        pts_of_predict = np.float32([[0,0],[W,0],[0,H],[W,H]])
        height_img, width_img = image.shape[0],image.shape[1]
        pts_of_img = np.float32([[0,0],[width_img,0],[0,height_img],[width_img,height_img]])
        Matrix  = cv2.getPerspectiveTransform(pts_of_predict,pts_of_img)

        #combining all picture of each segment into one frame
        combined_image = np.zeros((H , W), dtype=np.uint8)
        for i in range(len(masks)):
            combined_image += (masks[i].data.numpy() *255).astype("uint8").squeeze(axis=0) 
            combined_image[combined_image>=255]=255
        frame_result = cv2.warpPerspective(combined_image,Matrix,(width_img,height_img))
        return frame_result
    else:
        frame_result= np.zeros((1200 , 1980), dtype=np.uint8)
        return frame_result 

def map(inValue,  inMax,  inMin, outMax,  outMin ):

	if inValue > inMax: 
	
		return outMax
	
	elif inValue < inMin:

		return outMin

	else:

		return (inValue-inMin)*(outMax-outMin)/(inMax-inMin) + outMin



        

        

    




while(camera.isOpened()):

    ret, frame = camera.read()
    if ret==True:
        frame_resize = cv2.warpPerspective(frame,M_resize,(width_resize,height_resize))

        frame_detect = detect_by_YOLO(frame_resize)
        # detect_use_MOG(frame_resize,object_detector)
        bird_eye_frame = cv2.warpPerspective(frame_detect,M_forward,(width_resize,height_resize))
        # bird_eye_frame = cv2.warpPerspective(frame_resize,M_forward,(width_resize,height_resize))

        # draw_trajectory (bird_eye_frame,array_movement,array_length, array_angle,change_angle, actual_distance, actual_dis_p, actual_angle)
        # frame_resize = cv2.warpPerspective(bird_eye_frame,M_inverse,(width_resize,height_resize))
        

        # grid_size = (108, 192)  # Kích thước của đường lưới (số hàng, số cột)
        
        # draw_grid(bird_eye_frame, grid_size)

        # show ảnh ra màn hình
        cv2.imshow('Input',frame)

        # cv2.namedWindow("Input_1", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Input_1", 640, 480)
        # cv2.imshow('Input_1',frame_resize)

        cv2.namedWindow("Bird's eye view", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bird's eye view", 640, 480)
        cv2.imshow("Bird's eye view",bird_eye_frame)
        
        # cv2.namedWindow("Detection by YOLO", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Detection by YOLO", 640, 480)
        # cv2.imshow("Detection by YOLO",frame_detect)

        # Lấy thông tin về video (width, height, fps)
        # video_writer.write(frame)


            # Exit by pressing 'q'
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
                
        # Breaking when video is finished
    else:
        break

cv2.destroyAllWindows()
video_writer.release()
camera.release()

