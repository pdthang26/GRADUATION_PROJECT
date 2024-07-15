import socket
import threading
bufferSize = 1024
serverAdress = "192.168.1.1"
serverPort = 2024


socketClient = socket.socket(family= socket.AF_INET, type=  socket.SOCK_DGRAM)
# gui loi chào đến server

msgPC = 'success'
bytesToSend = msgPC.encode('utf-8')
socketClient.sendto(bytesToSend,(serverAdress,serverPort))

print(" Client is Up and Listenning . . . ")

def recei_data():
    while True:
        data, address = socketClient.recvfrom(bufferSize)
        data = data.decode('utf-8')
        print(address)
        print('data from Server: ',data)

thread = threading.Thread(target= recei_data,args= ()).start()

while True:
    msg = input()
    bytesToSend = msg.encode('utf-8')
    socketClient.sendto(bytesToSend,(serverAdress,serverPort))

    