from math import *

# Hàm tam xuất giá trị
def map(inValue,  inMax,  inMin, outMax,  outMin ):

	if inValue > inMax: 
	
		return outMax
	
	elif inValue < inMin:

		return outMin

	else:

		return (inValue-inMin)*(outMax-outMin)/(inMax-inMin) + outMin


def stanley_control(desired_angle,current_angle,desired_length,actual_length,velocity,k):
    psi = desired_angle-current_angle
    e = desired_length - actual_length
    delta  = psi + atan((k*e)/velocity)
    if delta == 0:
        return 20000
    elif delta > 0:
        pulse = int(map(delta, 38, 0, 39900, 20000))
        return pulse
    elif delta < 0:
        pulse = int(map(delta, 0, -38, 20000, 100))
        return pulse


