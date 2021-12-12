import matplotlib.pyplot as plt     
import time
def getN(posA, posB,samplePeriod):
    tsA = posA['ts']
    tsB = posB['ts']
    N =int((tsB - tsA) / (samplePeriod*1000))
    return N 

def generateGT(rawPosTable,samplePeriod):
    gtPos = []
    for  i in range(len(rawPosTable)-1):
        N = getN(rawPosTable[i], rawPosTable[i+1],samplePeriod)
        initTime = rawPosTable[i]['ts']
        initPosX =  rawPosTable[i]['x']
        initPosY = rawPosTable[i]['y']
        endTime = rawPosTable[i+1]['ts']
        endPosX = rawPosTable[i+1]['x']
        endPosY = rawPosTable[i+1]['y']
        realPeriod = (endTime - initTime) / N
        #print(N)
        for j in range(N):
            pos = {}
            pos['ts'] = initTime + j * realPeriod
            pos['x'] = (initPosX * (N-j) + endPosX * j )/ N
            pos['y'] = (initPosY * (N-j) + endPosY * j )/ N
            gtPos.append(pos)
    return gtPos
'''
start1 = time.time()
samplePeriod = 100
gtFile = open("movements.dat")
lines = gtFile.readlines()
rawPosTable = [] 
tsPrevious = 0
for line in lines:
    pos = {}
    raw = line.split(',')
    #print((raw))
    pos['ts'] = int(raw[1])
    pos['x'] = float(raw[3])
    pos['y'] = float(raw[4])
    rawPosTable.append(pos)

#print(rawPosTable)
gt = generateGT(rawPosTable)
#print(gt)  
for pos in gt:
    plt.xlim(600,1200)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-2332,-1635)
    locx = pos['x']
    locy = pos['y']
    ts = pos['ts']
    if tsPrevious !=0 :
        period = (ts - tsPrevious) / 1000
    else:
        period = 0.2
    tsPrevious = ts
    #print(period)

    plt.scatter(locx, -locy)
    plt.pause(period )
    plt.clf()

end1 = time.time()
print(end1-start1)
'''