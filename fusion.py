import json
import time as ti
import numpy as np 
from matplotlib import pyplot as plt 
import wifi_particle_origin as particle_filter
import math
import random
from shapely.geometry import Polygon


class sensors:
    def __init__(self,speed,heading):
        self.speed = speed
        self.heading = heading

class maps:
    def __init__(self,fileNames):
        inFiles = []
        outFiles = []
        for name in fileNames:
            if 'in' in name:
                inFiles.append(name)
            elif 'out' in name:
                outFiles.append(name)
        print(inFiles)
        print(outFiles)
        self.inPolygons = []
        for inFile in inFiles:
            iFile = open(inFile)
            positionInfos = iFile.readlines()
            vertexs = []
            for positionInfo in positionInfos:
                pos = positionInfo.split(' ') 
                posX = float(pos[0])
                posY = float(pos[1])
                vertexs.append([posX,posY]) 
            poly = Polygon(vertexs) 
            self.inPolygons.append(poly)

        self.outPolygons = []
        for outFile in outFiles:
            oFile = open(outFile)
            positionInfos = oFile.readlines()
            vertexs = []
            for positionInfo in positionInfos:
                pos = positionInfo.split(' ') 
                posX = float(pos[0])
                posY = float(pos[1])
                vertexs.append([posX,posY])  
            poly = Polygon(vertexs) 
            self.outPolygons.append(poly)
      
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def getSpeed(eular, timePeriod, acc, previousVelocity):
    x = eular[0]
    y = eular[1]
    z = eular[2]
    Y = x*math.pi/180
    P = y*math.pi/180
    R = z*math.pi/180
    theta = (Y,P,R)
    matrix = eulerAnglesToRotationMatrix(theta)
    inverseRotation = np.linalg.inv(matrix) 
    accWorld = np.dot(acc, inverseRotation)
    velocity = [0,0,0]
    if abs(acc[0]) >= 0.3:
        velocity[0] = previousVelocity[0] + timePeriod*acc[0]/1000
    else:
        velocity[0] = previousVelocity[0]

    if abs(acc[1]) >= 0.3:
        velocity[1] = previousVelocity[1] + timePeriod*acc[1]/1000
    else:
        velocity[1] = previousVelocity[1]
    
    if abs(acc[2]) >= 0.3:
        velocity[2] = previousVelocity[2] + timePeriod*acc[1]/1000
    else:
        velocity[2] = previousVelocity[2]
    
 
    speed = math.sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2])
    return velocity[0],velocity[1],velocity[2],speed



pppx = []
pppy = []
#pppx.append(0)
#pppy.append(0)
# Load WiFi localization
sensorFile = open('wifiLocateResult.json')
data = sensorFile.read()
WiFilocations = json.loads(data)
floorMap = maps(['in1.txt','in2.txt','in3.txt','out1.txt'])

for poly in floorMap.inPolygons:
    x, y = poly.exterior.coords.xy
    plt.plot(x,y)

for poly in floorMap.outPolygons:
    x, y = poly.exterior.coords.xy
    plt.plot(x,y)



acc = []
timeStamp = []
# Load acceleration data
sensorFile = open('19.dat')
lines = sensorFile.readlines()
for line in lines:
    data = line.split()
    accData = json.loads(data[2])
    time = data[0]
    time = int(time)
    acc.append(accData["vals"])
    timeStamp.append(time)
    #print(time,accData["vals"])

# load orientation data
Q = []
timeStamp = []
sensorFile = open('26.dat')
lines = sensorFile.readlines()
for line in lines:
    data = line.split()
    QData = json.loads(data[2])
    time = data[0]
    time = int(time)
    Q.append(QData["vals"])
    timeStamp.append(time)

def createPos():
    x = random.uniform(2500,3500)
    y = random.uniform(2500,3500)
    return x,y

ppppx =[]
ppppy = []
# init speed info: 
velocity = [0,0,0]
currentSpeed = 0
particles = []

    
for i in range(len(timeStamp)-1):
    timeTag = timeStamp[i]
     
    # load INS data
    eular = Q[i]
    accData = acc[i]
    velocity = getSpeed(eular,5,accData, velocity)
    currentSpeed = velocity[3]
    sensor = sensors(currentSpeed, eular[0]+30)
    # Check if there is WiFi data available
    if  WiFilocations:   
        latestWiFiInformation = WiFilocations[0]
    else:
        latestWiFiInformation = []
    hasWiFiLocations = False
    if latestWiFiInformation: 
        WiFiTime = latestWiFiInformation['ts']
        if abs(WiFiTime - timeTag) <= 100:
            hasWiFiLocations = True
    
    # Add WiFi particles
    WiFiParticles = []
    if hasWiFiLocations: 
        latestWiFiInfo = WiFilocations.pop(0) # a dictionary
        latestWiFiLoc = latestWiFiInfo['candidates']  # a list 
        for loc in latestWiFiLoc:
            x = loc['x']
            y = loc['y']
            pos = (x,y)
            w = loc['likelihood']
            aParticle = particle_filter.particle(pos,w) 
            WiFiParticles.append(aParticle)  
    if i % 200 == 0 or hasWiFiLocations:
        particles = particle_filter.particleFilter(particles,sensor, WiFiParticles,floorMap)
        if particles:    
            pos, confidence = particle_filter.getMean(particles)
            if i % 200 == 0:
                pppx.append(pos[0])
                pppy.append(pos[1])
            if hasWiFiLocations: 
                ppppx.append(pos[0])
                ppppy.append(pos[1])
            
            print(pos,confidence,sensor.heading,sensor.speed)
        #ti.sleep(1)

sensorFile = open('wifiLocateResult.json')
data = sensorFile.read()
locations = json.loads(data)
x = []
y = []
#x.append(0)
#y.append(0)
for loc in locations:
    x.append(loc['candidates'][0]['x'])
    y.append(loc['candidates'][0]['y'])
shapeFile = open('shape_description.shape')
xx = []
yy = []
#xx.append(0)
#yy.append(0)
lines = shapeFile.readlines()
for line in lines:
    data = line.split()
    xx.append(float(data[0]))
    yy.append(float(data[1]))

plt.plot(x,y,color='blue')
#plt.scatter(xx,yy,color='green')
plt.scatter(x,y,color='blue')
plt.plot(ppppx,ppppy,color='green')
plt.plot(pppx,pppy,color = 'red')
plt.scatter(pppx,pppy,color = 'red')
plt.show()    
