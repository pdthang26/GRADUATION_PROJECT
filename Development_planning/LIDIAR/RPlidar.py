import serial
import struct
from time import sleep

port = "COM1"
baudrate = 115200
ser = serial.Serial(port, baudrate)

def send_request( request):
    """Send request to the RPLidar."""
    ser.write(request)

def read_response(size=1):
    """Read a given size of response from RPLidar."""
    return ser.read(size)

def stop():
    """Send a Stop command to RPLidar."""
    stop_command = b'\xA5\x25'
    send_request(stop_command)

def reset():
    """Send a Reset command to RPLidar."""
    reset_command = b'\xA5\x40'
    send_request(reset_command)

def get_health():
    """Get the health status of the RPLidar."""
    health_command = b'\xA5\x52'
    send_request(health_command)
    # Read descriptor and health response
    descriptor = read_response(7)
    health_response = read_response(3)
    status, error_code = struct.unpack('<Bh', health_response)
    return {'status': status, 'error_code': error_code}

def get_info():
    """Get the info from the RPLidar."""
    info_command = b'\xA5\x50'
    send_request(info_command)
    # Read descriptor and info response
    descriptor = read_response(7)
    info_response = read_response(20)
    unpacked_data = struct.unpack('<BBBHB16s', info_response)
    return {
        'model': unpacked_data[0],
        'firmware_version': (unpacked_data[1], unpacked_data[2]),
        'hardware_version': unpacked_data[3],
        'serial_number': unpacked_data[5]
    }

def start_scan():
    """Initiate scanning mode."""
    scan_command = b'\xA5\x20'
    send_request(scan_command)
    # Read descriptor to determine response size
    # descriptor = read_response(7)
    # print(descriptor)
    sleep(2)
    full_scan_data = []
    collecting = False
    
    while True:
        response = read_response(5)  # Read one measurement
        if len(response) == 5:
            
            # break
            quality, angle, distance = struct.unpack('<BHH', response)
            current_angle = (angle >> 1) / 64.0
            start_flag = quality & 0b1  # Check the lowest bit for start flag

            if start_flag:
                if collecting:  # We've completed a 360 degree scan
                    break
                collecting = True  # Start collecting data for a new scan

            if collecting:
                # Store the measurement in the list
                full_scan_data.append({
                    'quality': quality >> 2,
                    'angle': current_angle,
                    'distance': distance / 4.0
                })
        else:
            print('not enough data')
    return full_scan_data
    


def close():
    """Close the serial port."""
    ser.close()

start_scan()
    
        
