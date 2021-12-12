import numpy as np
import time as ti
import random
import json
import math
import grid_generator as gg
from matplotlib import pyplot as plt 
import beaconLoc as bleLoc
import beaconTable
#import particle_filter  as pf 
minDis  = 1
wifi_sigma2 = 90000
ble_sigma = 30.9*21
orientation_sigma = 1
def huber_gauss(error, sigma):
    if abs(error) < abs(sigma):
        Loss =  1/2 * error ** 2 
    else:
        Loss =   sigma*(abs(error)- 1/2*sigma)
    g = 1000 * math.e ** (-Loss/(sigma*sigma))  
    return g

def w_gauss(error, sigma2):
    g = 10000 * math.e ** -(error ** 2 / (2 * sigma2))  
    return g

def euclideanDistance(posA, posB):
    vector1 = np.array(posA[0:2])
    vector2 = np.array(posB[0:2])
    dis=np.linalg.norm(vector1-vector2)
    return dis

def  isturning(insBuffer, threshold):
    # get accumulate INS readings
    gyro_accumulate = 0
    for ins in insBuffer:
        ins_data = ins['INSDict']
        gyro = ins_data['gyro']
        if gyro:
            gyro_accumulate += gyro[1]
    if gyro_accumulate >=  abs(threshold):
        return -1
    elif gyro_accumulate <= -abs(threshold):
        return 1
    else:
        return 0


def findNeighborPoints(candidatePoint, points,distanceBound):
    neighbor = []
    for pos in points: 
        dis = euclideanDistance(candidatePoint, pos) 
        if dis <= distanceBound :
            neighbor.append(pos)
    return neighbor

def generateGraph(points,distanceBound): 
    graph = {}
    for point in points:
        neighborPoints = findNeighborPoints(point, points,distanceBound)
        graph[str(point)] = neighborPoints
    return graph

def weight_initialize(points):
    if not points: 
        print("no candidate points!")
        return "no"
    weight_dict = {}
    totalPoints  = len(points)

    for point in points:
        point = list(point)
        weight_dict[str(point)] = 1 / totalPoints
    return weight_dict
# transition takes such way: a point can only walks into his neighbors with equal probability. we can learn the transition in the future. 
# Need to be modified

def transition_prediction(weight_dict, transition_probability, yaw, n):
    #print(yaw)
    new_weight_dict = {}
    for pos in weight_dict:
        new_weight_dict[pos] = 0
    for pos in weight_dict:
        pos_value = json.loads(pos)
        all_transitions = transition_probability[pos]
        for i in range(n):
            prob_list  = all_transitions[i]
            #print(prob_list)
            # The probability of stay
            if i == 0 :
                pos_prob = prob_list[0]
                buff_pos = pos_prob[0:3]
                prob = pos_prob[3]
                buff_pos = str(buff_pos)
                new_weight_dict[buff_pos] += (prob) * weight_dict[pos]
            else:
                for point_prob in prob_list:
                    buff_pos = point_prob[0:3]
                    prob = point_prob[3]
                    #print(prob)
                    dir_pos = direction(0, buff_pos[0:2],pos_value[0:2], 1)
                    #print(dir_pos)
                    ori_error = abs(dir_pos - yaw)
                    ori_error = ori_error* math.pi /180 
                    
                    buff_pos = str(buff_pos)
                    new_weight_dict[buff_pos] +=  prob * weight_dict[pos]*  huber_gauss(ori_error,orientation_sigma )
    return new_weight_dict
def L1_distance(pointx, pointy):
    return abs(pointx[0]-pointy[0]) + abs(pointx[1]-pointy[1])

def find_nhop_neighbor(candidatePoint,points, distanceBound, n):
    nhop_neighbor = {}
    for pos in points:
        dis =  L1_distance(candidatePoint, pos)
        for i in range(n):
            if dis > (i-1) * distanceBound and dis <=  (i) * distanceBound+0.01:
                if not i in nhop_neighbor:
                    neighbor = []
                    neighbor.append(pos)
                    nhop_neighbor[i] = neighbor
                else:
                    nhop_neighbor[i].append(pos)
    return nhop_neighbor


def getNhopGraph(points, distanceBound, n):
    graph = {}
    for point in points:
        nhop = find_nhop_neighbor(point,points,distanceBound,n)
        point = list(point)
        graph[str(point)] = nhop
    return graph

def visualize_nhop(graph):
    for pos_str in graph:
        plt.xlim(0,3000)
        plt.ylim(-2000,-600)
        n_hop = graph[pos_str]
        for n in n_hop:
            points = n_hop[n]
            points = np.array(points)
            x = points[:,0]
            y = -points[:,1]
            plt.scatter(x,y)
        plt.pause(0.2)
        plt. clf()
def loadGraph(fileName):
    import pickle
    file = open(fileName, 'rb')
    graph = pickle.load(file)
    return graph

def get_tag_route_pair():
    shapeFile = open("shape_description.shape")
    routes_raw = shapeFile.readlines()
    tag_route_pair = {}
    tag = 0
    positions = []
    for route in routes_raw:
        pos = route.split('  ')
        #print(pos)
        x = float(pos[0])
        y = float(pos[1].replace('\n',''))
        positions.append([x,y])
    for i in range(len(positions)-1):
        tag_route_pair[i] = [positions[i],positions[i+1]]
    return tag_route_pair

def getTransitionProbability(speed_data, n_hop_graph, d, delta_t):
    transitionProbability = {}
    for point in n_hop_graph:
        point = json.loads(point)
        tag = point[2]
        speed_distribution = speed_data[int(tag)]
        n_hop_neighbors = n_hop_graph[str(point)]
        point_transition = {}
        for n in n_hop_neighbors:
            n_hop_list = n_hop_neighbors[n]
            count = 0
            speed_l = (n-0.5)*d / delta_t
            speed_h = (n+0.5)*d / delta_t
            for speed in speed_distribution:
                if speed >= speed_l and speed <= speed_h:
                    count += 1
            transition_probability = count / len(speed_distribution)/len(n_hop_list)
            p_list  = []
            for p in n_hop_list:
                p_buff = [p[0],p[1],p[2],transition_probability]
                p_list.append(p_buff)
            point_transition[n] = p_list
        transitionProbability[str(point)] = point_transition
    return transitionProbability
                
def transition(weight_dict, neighbor_dict,yaw,samplePeriod,speed,grid_dis, yaw_bias):
    if yaw == None:
        yaw  = 10000
    yaw += yaw_bias
    if yaw > 180:
        yaw -= 360
    new_weight_dict = {}
    for pos in weight_dict:
        new_weight_dict[pos] = 0
    for pos in weight_dict:
        candidateNeighbors = neighbor_dict[pos]  # find the list of pos that the points may come to 
        numOfCandidates = len(candidateNeighbors)
        error_min = 100
        new_neighbor_dict_buff = {}
        speedCurrent = speed
        if isTurnPoint(json.loads(pos),candidateNeighbors) == 2:
            #print("hello")
            speedCurrent*=0.2
        for neighbor in candidateNeighbors:
            theta = 0
            
            candidatePos = json.loads(pos)
            dx = neighbor[0] - candidatePos [0]
            dy = neighbor[1] - candidatePos[1]
            if dx < -10*grid_dis: 
                theta = -180
            elif dx > 10*grid_dis:
                theta = 0
            elif dy < -10*grid_dis:
                theta = 90
            elif dy > 10*grid_dis: 
                theta = -90
            else:
                theta = yaw

            error = abs(yaw - theta)
            #print(yaw,error)
            error = min([error, 360-error]) * math.pi /180 
            if error < error_min and error !=0:
                error_min = error

            if yaw > 1000:
                new_neighbor_dict_buff[str(neighbor)] = 1
            elif error != 0:
                new_neighbor_dict_buff[str(neighbor)]  = huber_gauss(error,orientation_sigma )  # belief * transition

            else: 
                new_neighbor_dict_buff[str(neighbor)] =  0
        #print(new_neighbor_dict_buff)
        new_neighbor_dict_buff = normalization(new_neighbor_dict_buff)
        if error_min > 45*math.pi/180:
            for pos in new_neighbor_dict_buff:
                new_neighbor_dict_buff[pos] *= 0.01

        for  neighbor in new_neighbor_dict_buff:
            if not str(neighbor) in pos:
                new_weight_dict[neighbor] += weight_dict[pos]*new_neighbor_dict_buff[neighbor]
            else:
                new_weight_dict[neighbor] += weight_dict[pos]  * 0.5
            #print(new_weight_dict)
            #ti.sleep(0.1)
    return new_weight_dict


def isTurnPoint(candidatePoint, neighborPoints):
#####   Possible shapes: U L ---- T + 
    if len(neighborPoints) >= 3:
        #print(len(neighborPoints))
        return 1
    meanX = np.mean([pos[0] for pos in neighborPoints])
    meanY = np.mean([pos[1] for pos in neighborPoints])
    #print(meanX, meanY)
    if abs(meanX - candidatePoint[0]) > 3 or abs(meanY - candidatePoint[1]) > 3:
        return 2
    return 0 
 

def update_gyroscope(turn, weight_dict, neighbor_dict):
    if  turn == 0:
        return weight_dict

    for pos in weight_dict:
        candidatePos = json.loads(pos)
        if isTurnPoint(candidatePos, neighbor_dict[pos]):
            weight_dict[pos] *= 5
        return weight_dict



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
    #print(weight_dict)
    for pos in weight_dict:
        for bleSignal in bleTable:
            blePos = [bleSignal['x'], bleSignal['y']]
            rssi = bleSignal['rssi']
            Loss = (abs(rssi)+40)/20
            likelyhood = 1000*(10**(-Loss))
            candidatePos = json.loads(pos)
            dis = euclideanDistance(candidatePos[0:2],blePos)
            #ble_weight = huber_gauss(dis, ble_sigma)
            ble_weight = huber_gauss(dis, ble_sigma)
            if not(pos in ble_weight_dict.keys()):
                ble_weight_dict[pos] = ble_weight* likelyhood
            else:    
                ble_weight_dict[pos]+=  ble_weight* likelyhood
    for pos in ble_weight_dict:
        weight_dict[pos]*=ble_weight_dict[pos]
    return weight_dict

def normalization(weight_dict):
    if not weight_dict:
        print("no weight dict available")
        return exception
    weightSum = 0   
    for pos in weight_dict: 
        weightSum += weight_dict[pos]
    #print(weightSum)
    for pos in weight_dict: 
        weight_dict[pos] /= weightSum
    #print(testSum)
    return weight_dict


def rms(list):
    sum = 0
    for term in list:
        sum+= term*term
    rms = math.sqrt(sum / len(list))
    return rms

def expandPoints(start, end,samplePeriod,tag):
    expandedPoints = []
    ts_start = start['ts']
    ts_end = end['ts']
    points_num = int((ts_end-ts_start) / (samplePeriod * 1000))
    if points_num == 0:
        points_num += 1
    realPeriod = (ts_end - ts_start) / points_num 
    px_start = start['x']
    px_end = end['x']
    py_start = start['y']
    py_end = end['y']
    distance = math.sqrt(math.pow(px_start-px_end,2) + 
    math.pow(py_start-py_end,2) )
    N = points_num
    for j in range(points_num):
        pos  = {}
        pos['ts'] = ts_start + j * realPeriod
        pos['x'] = (px_start * (N-j) + px_end * j )/ N
        pos['y'] = (py_start* (N-j) + py_end * j )/ N
        pos['v'] = abs(distance / ((ts_start-ts_end) / 1000))
        pos['tag'] = tag
        expandedPoints.append(pos)        
    return expandedPoints,  abs(distance / ((ts_start-ts_end) / 1000))


# should return a list: x,y,t,v.
def processGT(filename):
    ground_truth = []
    ##### Generate Driving routes #######
    routes = []
    shapeFile = open("shape_description.shape")
    routes_raw = shapeFile.readlines()
    for route in routes_raw: 
        pos = route.split('  ')
        #print(pos)
        x = float(pos[0])
        y = float(pos[1].replace('\n',''))
        routes.append([x,y])
    #print(routes)
    gtFile = open(filename)
    gtLines = gtFile.readlines()
    gt_text = []

    # delete all "PAUSE" and following "RESTART"
    pauseID = []
    for i in range(len(gtLines)):
        if  gtLines[i].find('\"PAUSE\"') != -1:
            pauseID.append(i)
    for i in  range(int(len(gtLines) / 2)):
        if not i in pauseID:
            gt_text.append(gtLines[2*i])
            gt_text.append(gtLines[2*i+1])
    
    path_pairs = []
    route_cursor = 0
    routes_tag  = {}
    for i in range(len(gt_text)-1):
        current_text = gt_text[i]
        next_text = gt_text[i+1]
        ts_start = int(current_text.split('\t')[1])
        ts_end = int(next_text.split('\t')[1])
        px_start = routes[route_cursor][0]
        py_start = routes[route_cursor][1]
        # if it is start, then can generate it
        move = False
        if current_text.find("START") != -1:
            move = True
            px_end = routes[route_cursor+1][0]
            py_end = routes[route_cursor+1][1]  
            routes_tag[route_cursor] = {'start':[px_start, py_start], 'end':[px_end,py_end]}
            route_cursor += 1
        else: 
            px_end = routes[route_cursor][0]
            py_end = routes[route_cursor][1]
        
        start_node = {'ts': ts_start, 'x':px_start, 'y':py_start}
        end_node = {'ts': ts_end, 'x':px_end, 'y':py_end}
        batch_node, speed =  expandPoints(start_node,end_node,0.2,route_cursor)
        if move: 
            routes_tag[route_cursor-1]['speed'] = speed / 30.6
            if speed/30.6 > 10: 
                print("abnormal")
        for node in batch_node:
            #print(node)
            ground_truth.append(node)
    assert(route_cursor == 5)
    return ground_truth, routes_tag
    # Merge it as road segments 

def direction(yaw, a,b,grid_dis):
    dx = a[0] - b [0]
    dy =a[1] -b[1]
    if dx < -3*grid_dis: 
        theta = -181
    elif dx > 3*grid_dis:
        theta = 1
    elif dy < -3*grid_dis:
        theta = 91
    elif dy > 3*grid_dis: 
        theta = -91
    else:
        theta = yaw
    return theta

def modified_main(folder, INS_samples = 20, point_dis=0.5, n_hop=4):
    import groundtruth as  GT
    from matplotlib import pyplot as plt 
    import json
    
    #folder = '2021-01-31-10-51-15-481-trial11.sess'
    errorP = []
    orientation = []
    errorList = []
    iBeaconErrorList = []
    INS_samples = 20
    samplePeriod = INS_samples * 0.01
    gt,routes_tag = processGT(folder+'/movements.dat')

    scatter = []
    resultX = []
    resultY = []
    #--------------------load all the sensor readings--------------------------#
    # WiFi readings first
    beaconInfos = beaconTable.getBeaconInfo()
    beaconData =bleLoc. logToList2(folder+"/ibeaconScanner.dat",beaconInfos)
    
    INSData = []
    sensorFile = open(folder+'/IMUdata.dat')
    lines = sensorFile.readlines()
    for line in lines:
        result = line.split(';')
        readings = json.loads(result[1])
        time = int(result[0])
        bufferDict = {'ts':time, 'INSDict':readings}
        INSData.append(bufferDict)
    #print(INSData)
    sensorFile.close()
    import grid_generator as gg
    from matplotlib import pyplot as plt
    import json
    import numpy as np
    import pickle
    speed_file = open('speed.txt', 'rb')
    speed_data = pickle.load(speed_file)
    #print(sp)
    delta_t = 0.2
    grids = gg.generateGridPoints(point_dis*30.9, ["shape_description.shape"])
    points = gg.quickDelete(grids, point_dis*10)
    #print(points)
    #points = np.array(points)

    graph = getNhopGraph(points,point_dis*30.9*1.2,10)
    file = open('graph.txt','wb')
    pickle.dump(graph,file,0)
    file.close()
    n_hop_graph = loadGraph('graph.txt')
    #print(n_hop_graph)
    transition_probability = getTransitionProbability(speed_data,n_hop_graph,point_dis, delta_t)
    #print(transition_probability)
    #neighbor_dict = generateGraph(points,point_dis*30.9*1.4)
    weight_dict = weight_initialize(points)
    
    for pos in weight_dict:
            pos_gt = [gt[0]['x'],gt[0]['y']]
            #print(pos)
            candidatePos = json.loads(pos)
            dis = euclideanDistance(candidatePos, pos_gt)
            #ble_weight = huber_gauss(dis, ble_sigma)
            weight_dict[pos] = huber_gauss(dis, 300)
    weight_dict = normalization(weight_dict)

    i = 0
    beaconTimeCursor  = 0
    count = 0
    start2 = ti.time()
    yaw_old = 0
    yaw = 0
    while INSData and len(INSData) >= 2*INS_samples:
        #print("hello")
        #---------------fetch data for 1 second--------------#
        insBuffer = [INSData.pop(0) for i in range(INS_samples+1)]
        beaconBuffer = {}
        #print(insBuffer)
        # see if WiFi data is available
        startTime = insBuffer[0]['ts']
        endTime = insBuffer[INS_samples]['ts']
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
        #ori = insBuffer[0]['INSDict']['orientation']
        ori = [0,0,0]
        yaw = direction( yaw , [ gt[1]['x'], gt[1]['y']],  [gt[0]['x'], gt[0]['y']],point_dis)
        #yaw_old = yaw
        orientation.append(yaw)
        
        if ori != [] and len(ori) > 0:
            orientation_error = yaw-ori[0]
        #print(orientation_error)
        #print(yaw)
        weight_dict = transition_prediction(weight_dict, transition_probability, yaw, n_hop)
        weight_dict = normalization(weight_dict)
        # ------------- Turning  detection and update-----------#
        #turn = isturning(insBuffer,10)
        #if turn != 0:
            #print("turn")
        #weight_dict = update_gyroscope(turn, weight_dict, neighbor_dict)
        #weight_dict = normalization(weight_dict)
        # -----------WiFi update----------------#
        ibeacon_pos = None
        if beaconBuffer:
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

        #print(weight_dict)
        
        for pos  in weight_dict:
            loc = json.loads(pos)
            x.append(loc[0])
            y.append(-loc[1])
            weight = weight_dict[pos]
            w.append(weight *500)
            w2.append(10)
            if weight > wMax:
                wMax = weight
                xMax  = loc[0]
                yMax = loc[1]
        debug = True
        
        #plt.xlim(0,3000)
        #plt.ylim(-2332,0)

        #plt.scatter(x, y,s=w,color='blue')
        #if groundTruth:
            #plt.scatter(groundTruth['x'],-groundTruth['y'],s = 50,color = 'green')
        
        #plt.pause((endTime-startTime)/3000)
        #plt. clf()

    
        if groundTruth:
            gtx = groundTruth['x']
            gty = groundTruth['y']
            error = euclideanDistance([gtx,gty], [xMax,yMax]) / 30.9
            errorList.append(error)
        if ibeacon_pos:
            iBeaconPosError = euclideanDistance([gtx,gty],ibeacon_pos) /30.9
            #print(iBeaconPosError)
            iBeaconErrorList.append(iBeaconPosError)
        if groundTruth and ibeacon_pos:
            scatter.append([iBeaconPosError,error/1.3])

        #print(error/13.33)
        if False:
            plt.scatter(x, y,s=w,color='blue')
            plt.pause(0.2)
            plt. clf()
        
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

    period_hmm = (end2-start2)/len(errorList)
    rmsBeacon = rms(iBeaconErrorList)
    rmsHMM = rms(errorList)
    return  orientation,errorList,iBeaconErrorList,scatter,period_hmm
def main(folder,INS_samples = 20,speed = 2.5,point_dis = 1):
    import groundtruth as  GT
    from matplotlib import pyplot as plt 
    errorP = []
    orientation = []
    errorList = []
    iBeaconErrorList = []
    samplePeriod = INS_samples * 0.01
    gt,routes_tag = processGT(folder+'/movements.dat')
    #print(gt)
#print(rawPosTable)
    scatter = []
    resultX = []
    resultY = []
    #--------------------load all the sensor readings--------------------------#
    # WiFi readings first
    beaconInfos = beaconTable.getBeaconInfo()
    beaconData =bleLoc. logToList2(folder+"/ibeaconScanner.dat",beaconInfos)
    # INS data
    INSData = []
    sensorFile = open(folder+'/IMUdata.dat')
    lines = sensorFile.readlines()
    for line in lines:
        result = line.split(';')
        readings = json.loads(result[1])
        time = int(result[0])
        bufferDict = {'ts':time, 'INSDict':readings}
        INSData.append(bufferDict)
    #print(INSData)
    sensorFile.close()

    # --------------------firstly, generate the grids -----------------------------#
    grids = gg.generateGridPoints(point_dis*30.9, ["shape_description.shape"])
    points = gg.quickDelete(grids, point_dis*10)

    debug = False
    if debug:
        import matplotlib.pyplot as plt
        for grid in points:
            plt.scatter(grid[0], grid[1])
        plt.show()
    #--------------then generate the graph, and initialization---------------------#
    neighbor_dict = generateGraph(points,point_dis*30.9*1.4)
    weight_dict = weight_initialize(points)
    for pos in weight_dict:
            pos_gt = [gt[0]['x'],gt[0]['y']]
            candidatePos = json.loads(pos)
            dis = euclideanDistance(candidatePos, pos_gt)
            #ble_weight = huber_gauss(dis, ble_sigma)
            weight_dict[pos] = huber_gauss(dis, 300)
    weight_dict = normalization(weight_dict)
    #beaconBuffer = beaconData[0]
    #weight_dict = updateBle(beaconBuffer['table'], weight_dict)
    #weight_dict = normalization(weight_dict)
    #weight_dict = updateBle(beaconBuffer['table'], weight_dict)
    #weight_dict = normalization(weight_dict)
    #print(neighbor_dict)
    #print(weight_dict)
    i = 0
    beaconTimeCursor  = 0
    count = 0
    start2 = ti.time()
    yaw_old = 0
    yaw = 0
    while INSData and len(INSData) >= 2*INS_samples:
        #print("hello")
        #---------------fetch data for 1 second--------------#
        insBuffer = [INSData.pop(0) for i in range(INS_samples+1)]
        beaconBuffer = {}
        #print(insBuffer)
        # see if WiFi data is available
        startTime = insBuffer[0]['ts']
        endTime = insBuffer[INS_samples]['ts']
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
        #ori = insBuffer[0]['INSDict']['orientation']
        ori = [0,0,0]
        yaw = direction( yaw , [ gt[1]['x'], gt[1]['y']],  [gt[0]['x'], gt[0]['y']],point_dis)
        #yaw_old = yaw
        orientation.append(yaw)
        
        if ori != [] and len(ori) > 0:
            orientation_error = yaw-ori[0]
        #print(orientation_error)
        #print(yaw)
        weight_dict =  transition(weight_dict, neighbor_dict,yaw,samplePeriod,speed,point_dis,0)
        weight_dict = normalization(weight_dict)
        # ------------- Turning  detection and update-----------#
        #turn = isturning(insBuffer,10)
        #if turn != 0:
            #print("turn")
        #weight_dict = update_gyroscope(turn, weight_dict, neighbor_dict)
        #weight_dict = normalization(weight_dict)
        # -----------WiFi update----------------#
        ibeacon_pos = None
        if beaconBuffer:
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

        #print(weight_dict)
        
        for pos  in weight_dict:
            loc = json.loads(pos)
            x.append(loc[0])
            y.append(-loc[1])
            weight = weight_dict[pos]
            w.append(weight *500)
            w2.append(10)
            if weight > wMax:
                wMax = weight
                xMax  = loc[0]
                yMax = loc[1]
        debug = True
        
       # plt.xlim(600,1200)
       # plt.ylim(-2332,-1735)
        '''
        plt.scatter(x, y,s=w,color='blue')
        if groundTruth:
            plt.scatter(groundTruth['x'],-groundTruth['y'],s = 50,color = 'green')
        
        plt.pause((endTime-startTime)/3000)
        plt. clf()
        '''
        #plt.xlim(0,3000)
        #plt.ylim(-3000,0)
        if groundTruth:
            gtx = groundTruth['x']
            gty = groundTruth['y']
            error = euclideanDistance([gtx,gty], [xMax,yMax]) / 30.9
            errorList.append(error)
        if ibeacon_pos:
            iBeaconPosError = euclideanDistance([gtx,gty],ibeacon_pos) /30.9
            #print(iBeaconPosError)
            iBeaconErrorList.append(iBeaconPosError)
        if groundTruth and ibeacon_pos:
            scatter.append([iBeaconPosError,error/1.3])

        #print(error/13.33)
        if False:
            plt.scatter(x, y,s=w,color='blue')
            plt.pause(0.2)
            plt. clf()
        
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

    period_hmm = (end2-start2)/len(errorList)
    rmsBeacon = rms(iBeaconErrorList)
    rmsHMM = rms(errorList)
    return orientation,errorList,iBeaconErrorList,scatter


def get_nhop_performance(folderName,n_hop):
    global ble_sigma,orientation_sigma
    #import particle_filter  as pf 
    #print("begin particle filter")
    #particleErrorList,period_pf = pf.pf_process(500,folderName)
    print("begin RICH")
    orientation,hmmErrorList,iBeaconErrorList,error_scatter,period_hmm= modified_main(folderName,INS_samples=20,point_dis=0.5,n_hop=n_hop)
    #print(period_hmm)
    return hmmErrorList, period_hmm
def cdf_draw(folderName):
    global ble_sigma,orientation_sigma
    import particle_filter  as pf 
    print("begin particle filter")
    particleErrorList,period_pf = pf.pf_process(300,folderName)
    print("begin RICH")
    orientation,hmmErrorList,iBeaconErrorList,error_scatter,period_hmm= modified_main(folderName,INS_samples=20,point_dis=0.5)
    #print(period_pf,period_hmm)
    hmm_time_stamp  = [i*0.2 for i in range(len(hmmErrorList))]
    hmm_error_sorted = sorted(hmmErrorList)
    hmm_error_cdf = []
    for i in range(len(hmm_error_sorted)):
        hmm_error_cdf.append(i / len(hmmErrorList))

    pf_time_stamp  = [i*0.2 for i in range(len(particleErrorList))]
    pf_error_sorted = sorted(particleErrorList)
    pf_error_cdf = []
    for i in range(len(pf_error_sorted)):
        pf_error_cdf.append(i / len(pf_error_sorted))

    iBeacon_time_stamp = [i*1 for i in range(len(iBeaconErrorList))]        
    iBeacon_error_cdf = []
    for i in range(len(iBeaconErrorList)):
        iBeacon_error_cdf.append(i/len(iBeaconErrorList))
    iBeacon_error_sorted = sorted(iBeaconErrorList)
    plt.xlabel("Error(m)")
    plt.ylabel("CDF")
    hmm = plt.plot(hmm_error_sorted,hmm_error_cdf,color = 'blue',linestyle='-',linewidth='2',label='RICH')
    pf = plt.plot(pf_error_sorted,pf_error_cdf,color = 'red',linestyle='--',linewidth='1.5',label='Particle filter')
    wcl = plt.plot(iBeacon_error_sorted,iBeacon_error_cdf,color='green',linestyle=':',linewidth='1',label='iBeacon WCL')
    #plt.legend([hmm, pf, wcl], ['RICH', 'Particle Filter', 'iBeacon WCL'])
    plt.xlim(0,)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig(folderName+"/cdf.png")
    plt.show()
    return hmmErrorList,particleErrorList,iBeaconErrorList,error_scatter

def error_with_time_draw():
    global ble_sigma,orientation_sigma
    import particle_filter  as pf 
    particleErrorList,time = pf.pf_process(2000)
    orientation,hmmErrorList,iBeaconErrorList = main(speed=4,INS_samples=20,point_dis=1.2)
    hmm_time_stamp  = [i*0.2 for i in range(len(hmmErrorList))]
    pf_time_stamp  = [i*0.2 for i in range(len(particleErrorList))]
    iBeacon_time_stamp = [i*1 for i in range(len(iBeaconErrorList))]
    plt.xlim((0,35))
    plt.ylim((0,35))
    wcl = plt.plot(iBeacon_time_stamp,iBeaconErrorList,color='green',linestyle=':',linewidth='1',label='iBeaconWCL')
    pf = plt.plot(pf_time_stamp,particleErrorList,color = 'red',linestyle='--',linewidth='1.5',label='Particle filter')
    hmm = plt.plot(hmm_time_stamp,hmmErrorList,color = 'blue',linestyle='-',linewidth='2',label='FRICH')
    plt.xlabel("time stamp(s)")
    plt.ylabel("error(m)")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    import groundtruth as  GT
    from matplotlib import pyplot as plt 
    folder = '2021-01-31-10-51-15-481-trial11.sess'
    errorP = []
    orientation = []
    errorList = []
    iBeaconErrorList = []
    INS_samples = 20
    samplePeriod = INS_samples * 0.01
    gt,routes_tag = processGT(folder+'/movements.dat')

    scatter = []
    resultX = []
    resultY = []
    #--------------------load all the sensor readings--------------------------#
    # WiFi readings first
    beaconInfos = beaconTable.getBeaconInfo()
    beaconData =bleLoc. logToList2(folder+"/ibeaconScanner.dat",beaconInfos)
    
    INSData = []
    sensorFile = open(folder+'/IMUdata.dat')
    lines = sensorFile.readlines()
    for line in lines:
        result = line.split(';')
        readings = json.loads(result[1])
        time = int(result[0])
        bufferDict = {'ts':time, 'INSDict':readings}
        INSData.append(bufferDict)
    #print(INSData)
    sensorFile.close()
    import grid_generator as gg
    from matplotlib import pyplot as plt
    import json
    import numpy as np
    import pickle
    speed_file = open('speed.txt', 'rb')
    speed_data = pickle.load(speed_file)
    #print(sp)
    point_dis = 0.5
    delta_t = 0.2
    grids = gg.generateGridPoints(point_dis*30.9, ["shape_description.shape"])
    points = gg.quickDelete(grids, point_dis*10)
    print(points)
    #points = np.array(points)
    graph = getNhopGraph(points,point_dis*30.9*1.2,10)
    file = open('graph.txt','wb')
    pickle.dump(graph,file,0)
    file.close()
    n_hop_graph = loadGraph('graph.txt')
    #print(n_hop_graph)
    transition_probability = getTransitionProbability(speed_data,n_hop_graph,point_dis, delta_t)
    #print(transition_probability)
    neighbor_dict = generateGraph(points,point_dis*30.9*1.4)
    weight_dict = weight_initialize(points)
    
    for pos in weight_dict:
            pos_gt = [gt[0]['x'],gt[0]['y']]
            #print(pos)
            candidatePos = json.loads(pos)
            dis = euclideanDistance(candidatePos, pos_gt)
            #ble_weight = huber_gauss(dis, ble_sigma)
            weight_dict[pos] = huber_gauss(dis, 300)
    weight_dict = normalization(weight_dict)

    i = 0
    beaconTimeCursor  = 0
    count = 0
    start2 = ti.time()
    yaw_old = 0
    yaw = 0
    while INSData and len(INSData) >= 2*INS_samples:
        #print("hello")
        #---------------fetch data for 1 second--------------#
        insBuffer = [INSData.pop(0) for i in range(INS_samples+1)]
        beaconBuffer = {}
        #print(insBuffer)
        # see if WiFi data is available
        startTime = insBuffer[0]['ts']
        endTime = insBuffer[INS_samples]['ts']
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
        #ori = insBuffer[0]['INSDict']['orientation']
        ori = [0,0,0]
        yaw = direction( yaw , [ gt[1]['x'], gt[1]['y']],  [gt[0]['x'], gt[0]['y']],point_dis)
        #yaw_old = yaw
        orientation.append(yaw)
        
        if ori != [] and len(ori) > 0:
            orientation_error = yaw-ori[0]
        #print(orientation_error)
        #print(yaw)
        weight_dict = transition_prediction(weight_dict, transition_probability, yaw, 4)
        weight_dict = normalization(weight_dict)
        # ------------- Turning  detection and update-----------#
        #turn = isturning(insBuffer,10)
        #if turn != 0:
            #print("turn")
        #weight_dict = update_gyroscope(turn, weight_dict, neighbor_dict)
        #weight_dict = normalization(weight_dict)
        # -----------WiFi update----------------#
        ibeacon_pos = None
        if beaconBuffer:
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

        #print(weight_dict)
        
        for pos  in weight_dict:
            loc = json.loads(pos)
            x.append(loc[0])
            y.append(-loc[1])
            weight = weight_dict[pos]
            w.append(weight *500)
            w2.append(10)
            if weight > wMax:
                wMax = weight
                xMax  = loc[0]
                yMax = loc[1]
        debug = True
        
        plt.xlim(0,3000)
        plt.ylim(-2332,0)

        plt.scatter(x, y,s=w,color='blue')
        if groundTruth:
            plt.scatter(groundTruth['x'],-groundTruth['y'],s = 50,color = 'green')
        
        plt.pause((endTime-startTime)/3000)
        plt. clf()

    
        if groundTruth:
            gtx = groundTruth['x']
            gty = groundTruth['y']
            error = euclideanDistance([gtx,gty], [xMax,yMax]) / 30.9
            errorList.append(error)
        if ibeacon_pos:
            iBeaconPosError = euclideanDistance([gtx,gty],ibeacon_pos) /30.9
            #print(iBeaconPosError)
            iBeaconErrorList.append(iBeaconPosError)
        if groundTruth and ibeacon_pos:
            scatter.append([iBeaconPosError,error/1.3])

        #print(error/13.33)
        if False:
            plt.scatter(x, y,s=w,color='blue')
            plt.pause(0.2)
            plt. clf()
        
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

    period_hmm = (end2-start2)/len(errorList)
    rmsBeacon = rms(iBeaconErrorList)
    rmsHMM = rms(errorList)
    #error_with_time_draw()
    #INS_Error_Draw()
    #iBeacon_Error_Draw()
    #cdf_draw()
    #orientation,hmmErrorList,iBeaconErrorList = main('/home/zhangzheng/vision_city_sitesurvey',speed=6,INS_samples=20,point_dis=1.28)
   #cdf_draw('/home/zhangzheng/vision_city_sitesurvey')
    

