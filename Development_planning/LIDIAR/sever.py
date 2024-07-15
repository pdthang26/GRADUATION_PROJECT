import socket
import time
import threading

#declaration varible-----------------------------------------
bufferSize = 1024 # maximum size of transfer data
msgFromServer = "success" # message from raspberry pi to PC
serverAdress = "192.168.1.1"
serverPort = 2024

#------------------------------------------------------------
# transfer and receive data ---------------------------------
bytesToSend = msgFromServer.encode('utf-8') # encoding data 
socketServer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # initial the socket in raspberry pi
socketServer.bind((serverAdress,serverPort)) # conecting to adress
print(" server is Up and Listenning . . . ")

def recei_data():
    while True:
        data, address = socketServer.recvfrom(bufferSize)
        data = data.decode('utf-8')
        print(address)
        print('data from client: ',data)

        socketServer.sendto(bytesToSend,address)

thread = threading.Thread(target= recei_data,args= ()).start()
# while True:
#     time.sleep(1)
#     msg = input()
#     bytesToSend = str.encode(msg)
#     socketServer.sendto(bytesToSend,(serverAdress,serverPort))
    
    
#------------------------------------------------------------
