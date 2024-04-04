from math import *

# Hàm tam xuất giá trị
def map(inValue,  inMax,  inMin, outMax,  outMin ):

	if inValue > inMax: 
	
		return outMax
	
	elif inValue < inMin:

		return outMin

	else:

		return (inValue-inMin)*(outMax-outMin)/(inMax-inMin) + outMin

def stanley_control_point(current_point,current_angle,target_point,target_angle,velocity,k_coef):
    psi = target_angle-current_angle
    front_wheel_point = [0,0] # tạo điểm lưu tọa độ của bánh trước
    front_wheel_point[0] = current_point[0] + sin(radians(-current_angle)) * 2.1 # giá trị trục X
    front_wheel_point[1] = current_point[1] + cos(radians(-current_angle)) * 2.1 # giá trị trục Y
    e = sqrt((target_point[0]-front_wheel_point[0])**2 + (target_point[1] - front_wheel_point[1])**2 )
    if velocity ==0:
        delta  = psi + atan(0)
    else:
        delta  = psi + atan((k_coef*e)/velocity)

    if delta == 0:
        return 20000
    elif delta > 0:
        pulse = int(map(delta, 38, 0, 39900, 20000))
        return pulse
    elif delta < 0:
        pulse = int(map(delta, 0, -38, 20000, 100))
        return pulse

print(stanley_control_point((2,4),35,(1,2),45,0.8,10))


    