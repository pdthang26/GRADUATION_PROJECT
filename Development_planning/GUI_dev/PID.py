
from math import *
def map(inValue,  inMax,  inMin, outMax,  outMin ):

	if inValue > inMax: 
	
		return outMax
	
	elif inValue < inMin:

		return outMin

	else:

		return (inValue-inMin)*(outMax-outMin)/(inMax-inMin) + outMin




last_d_term_f = 0
def PID_control_front_wheel(angle_desire, angle_actual,sample_time):
    global last_d_term_f 
    P_gain =  2
    D_gain = 0.08
    alpha = 0.1

    error = angle_desire - angle_actual
    d_term = error/sample_time
    d_term_f = alpha*d_term + (1-alpha)*last_d_term_f
    last_d_term_f = d_term_f
    output = int(P_gain*error + D_gain*d_term_f)

    if output == 0:
        return 10000
    elif output > 0:
        pulse = int(map(output, 100, 0, 20000, 10000))
        return pulse
    elif output < 0:
        pulse = int(map(output, 0, -100, 10000, 0))
        return pulse

while(True):
      angle_ac = float(input('nhap goc that: '))
      angle_d = float(input('nhap goc yeu cau: '))
      print(last_d_term_f)
      print(int(PID_control_front_wheel(angle_d,angle_ac,0.15)))

      

		



