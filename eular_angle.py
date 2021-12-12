import json
import math
import time as ti
Q = []
timeStamp = []

sensorFile = open('26.dat')
lines = sensorFile.readlines()
for line in lines:
    data = line.split()
    QData = json.loads(data[2])
    time = data[1]
    time = int(time)
    Q.append(QData["vals"])
    timeStamp.append(time)

print(len(timeStamp)/200)
while True and False:
    aaa = 1
for i in range(len(timeStamp)-1):
    x = Q[i][0]
    y = Q[i][1]
    z = Q[i][2]
    time = timeStamp[i]
    print(time,Q[i][0],Q[i][1],Q[i][2])
    ti.sleep(0.003)


