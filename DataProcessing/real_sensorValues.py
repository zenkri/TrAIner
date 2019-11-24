import serial
import json
import time

ser = serial.Serial('COM13', 9600, timeout=10)
n=0
Acc = 1
millis = int(round(time.time() * 1000))

while 1:
    while (n < 10):
       data = ser.readline()
       n = n+1

    data = ser.readline()

    data = str(data, 'utf-8')
   # print(data)
   # print(data[:-16])
    str_data = data[:-16] + "}"
    #print(str_data)
    j = json.loads(str_data)
    #print(j["acc"]['x'])
    #add acc variables
    xAcc = j["acc"]['x']
    yAcc = j["acc"]['x']
    zAcc = j["acc"]['x']
    prevAcc= Acc
    prevTime = millis
    millis =int(round(time.time() * 1000))

    Acc = (xAcc*xAcc + yAcc*yAcc + zAcc*zAcc)/7900

    #print(Acc)
    derAcc = (Acc - prevAcc)/(int(round(time.time() * 1000)) - prevTime)
    print (derAcc)