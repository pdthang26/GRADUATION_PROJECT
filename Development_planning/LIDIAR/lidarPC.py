import socket
msgPC = 'hello rasp, this is a message from PC'
run_signal = 'run'
run_signal = run_signal.encode('utf-8')
bytesToSend = msgPC.encode('utf-8')
raspAddress = ('192.168.2.72',2222)
bufferSize = 1024
socketPC = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
socketPC.sendto(bytesToSend,raspAddress)
while True:
    socketPC.sendto(run_signal,raspAddress)
    data, address = socketPC.recvfrom(bufferSize)
    data =data.decode('utf-8')
    print('data from raspberrypi: ',data)
    