import numpy as np
import time as ti
import random
import json
import math
import grid_generator as gg
from matplotlib import pyplot as plt 
import beaconLoc as bleLoc
import beaconTable
import particle_filter  as pf 
exception = 0
distanceBound = 17
minDis  = 1
wifi_sigma2 = 90000
ble_sigma = 80
orientation_sigma2 = 0.25

def huber_gauss(error, sigma):
    if abs(error) < abs(sigma):
        Loss =  1/2 * error ** 2 
    else:
        Loss =   sigma*(abs(error)- 1/2*sigma)
    g = 100 * math.e ** (-Loss/(sigma*sigma))  
    return g

def generateCandidatePoints(map):
    return "pending implement"

def  turning(insBuffer):
    # get accumulate INS readings
    gyro_accumulate = 0
    for ins in insBuffer:
        ins_data = ins['INSDict']
        gyro = ins_data['gyro']
        if gyro:
            gyro_accumulate += gyro[1]

    if gyro_accumulate >=  4:
        return -1
    elif gyro_accumulate <= -4:
        return 1
    else:
        return 0

def euclideanDistance(posA, posB):
    vector1 = np.array(posA)
    vector2 = np.array(posB)
    dis=np.linalg.norm(vector1-vector2)
    return dis


def findNeighborPoints(candidatePoint, points):
    neighbor = []
    for pos in points: 
        dis = euclideanDistance(candidatePoint, pos) 
        if dis <= distanceBound :
            neighbor.append(pos)
    return neighbor

def generateGraph(points): 
    graph = {}
    for point in points:
        neighborPoints = findNeighborPoints(point, points)
        graph[str(point)] = neighborPoints
    return graph

def weight_initialize(points):
    if not points: 
        print("no candidate points!")
        return exception
    weight_dict = {}
    totalPoints  = len(points)
    for point in points:
        weight_dict[str(point)] = 1 / totalPoints
    return weight_dict
# transition takes such way: a point can only walks into his neighbors with equal probability. we can learn the transition in the future. 
# Need to be modified
def transition2(weight_dict, neighbor_dict,yaw):
    yaw += 170
    if yaw > 180:
        yaw -= 360
    new_weight_dict = {}
    for pos in weight_dict:
        new_weight_dict[pos] = 0
    for pos in weight_dict:
        candidateNeighbors = neighbor_dict[pos]  # find the list of pos that the points may come to 
        numOfCandidates = len(candidateNeighbors)
        error_min = 100
        for neighbor in candidateNeighbors:
            theta = 0
            candidatePos = json.loads(pos)
            dx = neighbor[0] - candidatePos [0]
            dy = neighbor[1] - candidatePos[1]
            if dx < -3: 
                theta = -180
            elif dx > 3:
                theta = 0
            elif dy < -3:
                theta = -90
            elif dy > 3: 
                theta = 90
            else:
                theta = yaw

            error = abs(yaw - theta)
            error = min([error, 360-error]) * math.pi /180 
            if error < error_min:
                error_min = error
            #print(error,w_gauss(error,orientation_sigma2 ) )
            if error > 0:
                new_weight_dict[str(neighbor)] += weight_dict[pos]*w_gauss(error,orientation_sigma2 )   # belief * transition
            else:  
                new_weight_dict[str(neighbor)] += weight_dict[pos]*w_gauss(error_min,orientation_sigma2 ) *0.1
            #print(new_weight_dict)
            #ti.sleep(0.1)
    return new_weight_dict
def transition(weight_dict, neighbor_dict,yaw):
    yaw += 170
    if yaw > 180:
        yaw -= 360
    new_weight_dict = {}
    for pos in weight_dict:
        new_weight_dict[pos] = 0
    for pos in weight_dict:
        candidateNeighbors = neighbor_dict[pos]  # find the list of pos that the points may come to 
        numOfCandidates = len(candidateNeighbors)
        for neighbor in candidateNeighbors:
            theta = 0
            candidatePos = json.loads(pos)
            dx = neighbor[0] - candidatePos [0]
            dy = neighbor[1] - candidatePos[1]
            if dx < -3: 
                theta = -180
            elif dx > 3:
                theta = 0
            elif dy < -3:
                theta = -90
            elif dy > 3: 
                theta = 90
            else:
                theta = yaw

            error = abs(yaw - theta)
            error = min([error, 360-error]) * math.pi /180 
            #print(error,w_gauss(error,orientation_sigma2 ) )
            if error > 0:
                new_weight_dict[str(neighbor)] += weight_dict[pos]*w_gauss(error,orientation_sigma2 )   # belief * transition
            else:  
                new_weight_dict[str(neighbor)] += weight_dict[pos]*0.2
            #print(new_weight_dict)
            #ti.sleep(0.1)
    return new_weight_dict

def isTurnPoint(candidatePoint, neighborPoints):
#####   Possible shapes: U L ---- T + 
    if len(neighborPoints) != 3:
        #print(len(neighborPoints))
        return True
    meanX = np.mean([pos[0] for pos in neighborPoints])
    meanY = np.mean([pos[1] for pos in neighborPoints])
    #print(meanX, meanY)
    if abs(meanX - candidatePoint[0]) > 3 or abs(meanY - candidatePoint[1]) > 3:
        return True
    return False 
 

def update_gyroscope(turn, weight_dict, neighbor_dict):
    if  turn == 0:
        return weight_dict

    for pos in weight_dict:
        candidatePos = json.loads(pos)
        if isTurnPoint(candidatePos, neighbor_dict[pos]):
            weight_dict[pos] *= 5

    return weight_dict

def w_gauss(error, sigma2):
    g = 10000 * math.e ** -(error ** 2 / (2 * sigma2))  
    return g

def ble_localization(bleTable):
    pos = [0,0]
    likelyhoodSum = 0
    for bleSignal in bleTable:
        ble_dict = {}
        blePos = [bleSignal['x'], bleSignal['y']]
        rssi = bleSignal['rssi']
        Loss = (abs(rssi)+40)/20
        likelyhood = 1000*(10**(-Loss))
        pos[0] += bleSignal['x'] * likelyhood
        pos[1] += bleSignal['y'] * likelyhood
        likelyhoodSum+= likelyhood
    pos[0]/= likelyhoodSum
    pos[1] /= likelyhoodSum
    return pos

#  bleTable:[ {'x': x, 'y':y, 'rssi':rssi}
def updateBle(bleTable, weight_dict):
    ble_weight_dict = {}
    for pos in weight_dict:
        for bleSignal in bleTable:
            blePos = [bleSignal['x'], bleSignal['y']]
            rssi = bleSignal['rssi']
            Loss = (abs(rssi)+40)/20
            likelyhood = 1000*(10**(-Loss))
            candidatePos = json.loads(pos)
            dis = euclideanDistance(candidatePos,blePos)
            ble_weight = huber_gauss(dis, ble_sigma)
            if not(pos in ble_weight_dict.keys()):
                ble_weight_dict[pos] = ble_weight* likelyhood
            else:    
                ble_weight_dict[pos]+=  ble_weight* likelyhood
    for pos in ble_weight_dict:
        weight_dict[pos]*=ble_weight_dict[pos]
    return weight_dict

def updateMultiWiFi(wifi, weight_dict):
    wifi_weight_dict = {}
    for pos in weight_dict:
        for wifiSignal in wifi:
            wifi_pos = [wifiSignal['x'], wifiSignal['y']]
            candidatePos = json.loads(pos)
            dis = euclideanDistance(candidatePos, wifi_pos)
            wifi_weight = w_gauss(dis, wifi_sigma2)
            if not (pos in wifi_weight_dict.keys()):
                wifi_weight_dict[pos] = wifi_weight * wifiSignal['likelihood']
            else:
                wifi_weight_dict[pos]+= wifi_weight * wifiSignal['likelihood']
    for pos in weight_dict:
        weight_dict[pos]*=wifi_weight_dict[pos]
    return weight_dict

def update_wifi(wifiPos,weight_dict):
    for pos in weight_dict: 
        candidatePos = json.loads(pos)
        dis = euclideanDistance(candidatePos, wifiPos)
        #print(candidatePos, wifiPos,dis)
        wifi_weight = w_gauss(dis, wifi_sigma2)
        #print(wifi_weight)
        weight_dict[pos] *= wifi_weight
    return weight_dict

def update_acclerometer():
    return "pending implement"
 
def update_orientation(orientation):
    return "pending implement"        

def normalization(weight_dict):
    if not weight_dict:
        print("no weight dict available")
        return exception
    for pos in weight_dict: 
        if weight_dict[pos] < 1e-6:
            weight_dict[pos] = 0
    weightSum = 0   
    for pos in weight_dict: 
        weightSum += weight_dict[pos]
    #print(weightSum)
    testSum = 0
    for pos in weight_dict: 
        weight_dict[pos] /= weightSum
        testSum+= weight_dict[pos] 
    #print(testSum)
    return weight_dict

def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c
     
def rms(list):
    sum = 0
    for term in list:
        sum+= term*term
    rms = math.sqrt(sum / len(list))
    return rms
def main():
    import groundtruth as  GT
    errorP = []
    orientation = []
    errorList = []
    iBeaconErrorList = []
    samplePeriod = 200
    gtFile = open("movements.dat")
    lines = gtFile.readlines()
    rawPosTable = [] 
    for line in lines:
        pos = {}
        raw = line.split(',')
    #print((raw))
        pos['ts'] = int(raw[1])
        pos['x'] = float(raw[3])
        pos['y'] = float(raw[4])
        rawPosTable.append(pos)
    gt = GT.generateGT(rawPosTable)
    #print(gt)
#print(rawPosTable)

    resultX = []
    resultY = []
    #--------------------load all the sensor readings--------------------------#
    # WiFi readings first
    beaconInfos = beaconTable.getBeaconInfo()
    beaconData =bleLoc. logToList2("ibeaconScanner.dat",beaconInfos)
    # INS data
    INSData = []
    sensorFile = open('data.dat')
    lines = sensorFile.readlines()
    for line in lines:
        result = line.split(';')
        readings = json.loads(result[1])
        time = int(result[0])
        if  readings['orientation'] and readings['gyro'] and readings['acc']: 
            bufferDict = {'ts':time, 'INSDict':readings}
            INSData.append(bufferDict)
    #print(INSData)
    sensorFile.close()

    # --------------------firstly, generate the grids -----------------------------#
    grids = gg.generateGridPoints(13.33, ["shape_description.shape","shape_description1.shape"])
    points = gg.quickDelete(grids, 5)
    
    #--------------then generate the graph, and initialization---------------------#
    neighbor_dict = generateGraph(points)
    weight_dict = weight_initialize(points)
    #print(neighbor_dict)
    #print(weight_dict)
    i = 0
    beaconTimeCursor  = 0
    count = 0
    start2 = ti.time()
    while INSData and len(INSData) >= 20:
        #---------------fetch data for 1 second--------------#
        insBuffer = [INSData.pop(0) for i in range(21)]
        beaconBuffer = {}
        #print(insBuffer)
        # see if WiFi data is available
        startTime = insBuffer[0]['ts']
        endTime = insBuffer[20]['ts']
        groundTruth = None
        while gt[0]['ts'] < startTime:
            groundTruth = gt.pop(0)
       # print(endTime-startTime)
        if beaconData:
            beaconTimeCursor = beaconData[0]['ts']
        #print(turning(insBuffer))
        if  abs(beaconTimeCursor - startTime) <= 1000 and  beaconData:
            beaconBuffer = beaconData.pop(0)
           # print(beaconTimeCursor,beaconBuffer)
        # ------------ Transition -----------#
        yaw = insBuffer[0]['INSDict']['orientation'][0]
        orientation.append(yaw)
        #print(yaw)
        weight_dict =  transition2(weight_dict, neighbor_dict,yaw)
        weight_dict = normalization(weight_dict)

        # ------------- Turning  detection and update-----------#
        turn = turning(insBuffer)
        
        #if turn:
            #print("turn")
        #else:
            #print("not turn")
        
        weight_dict = update_gyroscope(turn, weight_dict, neighbor_dict)
        weight_dict = normalization(weight_dict)
        # -----------WiFi update----------------#
        ibeacon_pos = None
        if beaconBuffer:
            #wifiMostLikely = WiFiBuffer['candidates'][0]
            #loc = [wifiMostLikely['x'], wifiMostLikely['y']]
            #print(loc)
            weight_dict = updateBle(beaconBuffer['table'], weight_dict)
            weight_dict = normalization(weight_dict)
            ibeacon_pos = ble_localization(beaconBuffer['table'])
            #print(ibeacon_pos)
        x = []
        y = []
        w = []
        w2 = []
        xMax = 0
        yMax = 0
        wMax = 0
        count += 1

        for pos  in weight_dict:
            loc = json.loads(pos)
            x.append(loc[0])
            y.append(-loc[1])
            weight = weight_dict[pos]
            if weight > wMax:
                wMax = weight
                xMax  = loc[0]
                yMax = loc[1]
            w.append(weight * 500)
            w2.append(10)
        #plt.xlim(600,1200)
      #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
        #plt.ylim(-2332,-1735)
        #plt.scatter(x, y,s=w,color='blue')
        gtx = groundTruth['x']
        gty = groundTruth['y']
        error = euclideanDistance([gtx,gty], [xMax,yMax]) / 13.33
        if ibeacon_pos:
            iBeaconPosError = euclideanDistance([gtx,gty],ibeacon_pos) /13.33
            #print(iBeaconPosError)
            iBeaconErrorList.append(iBeaconPosError)
        #print(error/13.33)
        errorList.append(error)
        errorP.append(error+abs(random.gauss(0.5,4.5)))
        #if groundTruth:
            #plt.scatter(groundTruth['x'],-groundTruth['y'],s = 50,color = 'green')
        #plt.scatter(xMax,-yMax,s=wMax*500,color='red')
        #plt.pause((endTime-startTime)/3000)
        #plt. clf()
        #print(count)
        #print()
        #result = json.loads(keys)
        #resultX.append(result[0])
        #resultY.append(result[1])
    #gridsX = [pos[0] for pos  in points]
    #gridsY = [pos[1] for pos  in points]
    #plt.scatter(gridsX, gridsY, color='r')
    #plt.scatter(resultX, resultY)
    #plt.plot(resultX, resultY)
    #plt.show()
    end2 = ti.time()
    #print(end2-start2)
    errorListSorted = sorted(errorList)
    errorPSorted = sorted(errorP)
    X = []
    Y  = [i*0.2 for i in range(len(errorList))]
    for i in range(len(errorList)):
        X.append(i / len(errorList))
    iBeaconY = [i*1 for i in range(len(iBeaconErrorList))]        
    iBeaconX = []
    for i in range(len(iBeaconErrorList)):
        iBeaconX.append(i/len(iBeaconErrorList))
    iBeaconErrorListSorted = sorted(iBeaconErrorList)

    period_hmm = end2-start2
    rmsBeacon = rms(iBeaconErrorList)
    rmsHMM = rms(errorList)
    return orientation,errorList,iBeaconErrorList
    #plt.clf()
    #plt.plot([0,3],[rmsBeacon,rmsBeacon],linestyle='--',color='green')
    #plt.scatter(3,rmsBeacon,s=50,color='green',marker='D')
    #plt.annotate("iBeacon WCL", xy = (3, rmsBeacon), xytext = (2.8, rmsBeacon+0.5)) 
    #plt.scatter(2,rmsPF,s=50,color='red')
    #plt.plot([0,1],[rmsHMM,rmsHMM],linestyle='--',color='blue')
    #plt.scatter(1,rmsHMM,s=50,color='blue')
    #plt.annotate("HMM", xy = (1, rmsHMM), xytext = (0.9, rmsHMM+0.5)) 
    #plt.plot(Y,errorList,color = 'blue')
    #plt.plot(Y,errorP,color = 'red')
    #plt.plot(iBeaconY, iBeaconErrorList, color = 'green')
    #plt.scatter(1,period_hmm/len(errorList),s=50,color='blue')
    #plt.annotate("HMM", xy=(1, period_hmm/len(errorList)), xytext = (0.8, period_hmm/len(errorList)+0.002))
    #plt.plot([0,1],[period_hmm/len(errorList),period_hmm/len(errorList)],linestyle='--',color='blue')
    #plt.show()
     
if __name__ == '__main__':
    particleErrorList = pf.pf_process()
    orientation,hmmErrorList,iBeaconErrorList = main()
    errorListSorted = sorted(hmmErrorList)
    errorParticleSorted = sorted(particleErrorList)
    X = []
    Y  = [i*0.2 for i in range(len(hmmErrorList))]
    for i in range(len(hmmErrorList)):
        X.append(i / len(hmmErrorList))
    iBeaconY = [i*1 for i in range(len(iBeaconErrorList))]        
    iBeaconX = []
    for i in range(len(iBeaconErrorList)):
        iBeaconX.append(i/len(iBeaconErrorList))
    iBeaconErrorListSorted = sorted(iBeaconErrorList)
    plt.xlim(0,30)
    plt.ylim(0,1)
    plt.plot(errorListSorted,X,color = 'blue',linestyle='-',linewidth='2')
    plt.plot(errorParticleSorted,X,color = 'red',linestyle='--',linewidth='1.5')
    plt.plot(iBeaconErrorListSorted,iBeaconX,color='green',linestyle=':',linewidth='1')
    plt.show()
    plt.xlim(0,35)
    plt.ylim(0,35)
    plt.plot(iBeaconY, iBeaconErrorList, color = 'green')
    plt.plot(Y,particleErrorList,color = 'red')
    plt.plot(Y,hmmErrorList,color = 'blue')
    plt.show()
    plt.xlim(0,30)
    plt.ylim(0,1)
    plt.plot(iBeaconErrorListSorted,iBeaconX,color='green',linestyle=':',linewidth='1')
    plt.show()
    
    '''
    grids = gg.generateGridPoints(13.33, ["shape_description.shape","shape_description1.shape"])
    gridsNew = gg.quickDelete(grids, 5)
    neighbor_dict = generateGraph(gridsNew)
    for pos in neighbor_dict:
        candidatePos = json.loads(pos)
        #print(neighbor_dict[pos])
        turn = isTurnPoint(candidatePos,neighbor_dict[pos])
        if turn: 
            print(pos + " is a turn point")
    '''



def test():
    # this is to test the functionality. I think I can have better way to generate the gragh
    points = []
    for i in range(1000):
        points.append( [i*30, 0] )

    neighbor_dict = generateGraph(points)
    #for pos in neighbor_dict:
        #print(pos + "has following neighbor: ")
        #print(neighbor_dict[pos])
    
    weight_dict = weight_initialize(points)
    if not weight_dict:
        print(" try to debug the none of weight_dict")
    for i in range(100):
        weight_dict =  transition(weight_dict, neighbor_dict)
        turn = False
        # random turn to test
        if random.uniform(0,1) > 0.99:
            turn = True 
        weight_dict = update_gyroscope(turn, weight_dict, neighbor_dict)
        weight_dict = normalization(weight_dict)
        # test wifi updates
        wifiPos = [i * 30 + random.gauss(0,200), 0]
        weight_dict = update_wifi(wifiPos, weight_dict)
        weight_dict = normalization(weight_dict)
        for key,value in weight_dict.items():
            if(value == max(weight_dict.values())):
                print(key,value)
        #print(weight_dict)
        #new_weight_dict[pos] =  weight_dict[pos]
        #print("round" + str(i))
        #time.sleep(1)
    #print(weight_dict)

    for pos in neighbor_dict:
        candidatePos = json.loads(pos)
        #print(neighbor_dict[pos])
        turn = isTurnPoint(candidatePos,neighbor_dict[pos])
        if turn: 
            print(pos + " is a turn point")
 
 #Things I have done:
# 1. Turn detection  
# 2. WiFi Update 
#3. Turning Update
#4. Transition Update based on orientation

#Still need improving
#1) the speed information does matter
#2) Sequence smoothing Viterbi 
#3) Auto generate the grids 
#4) Transition Training
#we need to consider into that whether orientation should be treated as a state, so next time please think of this issue.
#Posterier = Likelyhood * Prior 
#








#t =1 A1 A2 .. A10 
#t = 2  B1 B2 .... B10
#t =10

    
#nextly, we need to do the sensor update. But the lieklyhood and vitebi algorithm remains to be an issue         
