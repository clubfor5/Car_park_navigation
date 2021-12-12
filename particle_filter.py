# ------------------------------------------------------------------------
# coding=utf-8
# ------------------------------------------------------------------------
#
#  Created by Martin J. Laubach on 2011-11-15
#
# ------------------------------------------------------------------------

from __future__ import absolute_import
import grid_generator as gg 
import random
import math
import bisect
import json
from draw import Maze


import numpy as np
 
import time as ti
# 0 - empty square
# 1 - occupied square
# 2 - occupied square with a beacon at each corner, detectable by the robot

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
        theta = yaw-90
    return theta

import maze_generator as mg
maze_data = mg.generateMazeData(30.9)
world = Maze(maze_data)
#world.draw()
ROBOT_HAS_COMPASS = True # Does the robot know where north is? If so, it
# makes orientation a lot easier since it knows which direction it is facing.
# If not -- and that is really fascinating -- the particle filter can work
# out its heading too, it just takes more particles and more time. Try this
# with 3000+ particles, it obviously needs lots more hypotheses as a particle
# now has to correctly match not only the position but also the heading.
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
    x = [g['x'] for g in ground_truth]
    y = [g['y'] for g in ground_truth]
    #plt.scatter(x,y)
    #plt.show()
    return ground_truth, routes_tag
# ------------------------------------------------------------------------
# Some utility functions
def rms(list):
    sum = 0
    for term in list:
        sum+= term*term
    rms = math.sqrt(sum / len(list))
    return rms

def euclideanDistance(posA, posB):
    vector1 = np.array(posA)
    vector2 = np.array(posB)
    dis=np.linalg.norm(vector1-vector2)
    return dis

def add_noise(level, *coords):
    return [x + random.uniform(-level, level) for x in coords]

def add_little_noise(*coords):
    return add_noise(0.02, *coords)

def add_some_noise(*coords):
    return add_noise(0.1, *coords)

# This is just a gaussian kernel I pulled out of my hat, to transform
# values near to robbie's measurement => 1, further away => 0
sigma2 = 0.9 ** 2
def w_gauss(a, b):
    error = a - b
    g = math.e ** -(error ** 2 / (2 * sigma2))
    return g
def gauss(error,sigma):
    g = math.e ** -(error ** 2 / (2 * sigma*sigma))
    return g

def huber_gauss(error, sigma):
    if abs(error) < abs(sigma):
        Loss =  1/2 * error ** 2 
    else:
        Loss =   sigma*(abs(error)- 1/2*sigma)
    g = 100 * math.e ** (-Loss/(sigma*sigma))  
    return g
# ------------------------------------------------------------------------
def compute_mean_point(particles,PARTICLE_COUNT):
    """
    Compute the mean for all particles that have a reasonably good weight.
    This is not part of the particle filter algorithm but rather an
    addition to show the "best belief" for current position.
    """

    m_x, m_y, m_count = 0, 0, 0
    for p in particles:
        m_count += p.w
        m_x += p.x * p.w
        m_y += p.y * p.w

    if m_count == 0:
        return -1, -1, False

    m_x /= m_count
    m_y /= m_count

    # Now compute how good that mean is -- check how many particles
    # actually are in the immediate vicinity
    m_count = 0
    for p in particles:
        if world.distance(p.x, p.y, m_x, m_y) < 1:
            m_count += 1

    return m_x, m_y, m_count > PARTICLE_COUNT * 0.95

# ------------------------------------------------------------------------
class WeightedDistribution(object):
    def __init__(self, state):
        accum = 0.0
        self.state = [p for p in state if p.w > 0]
        self.distribution = []
        for x in self.state:
            accum += x.w
            self.distribution.append(accum)

    def pick(self):
        try:
            return self.state[bisect.bisect_left(self.distribution, random.uniform(0, 1))]
        except IndexError:
            # Happens when all particles are improbable w=0
            return None

# ------------------------------------------------------------------------
class Particle(object):
    def __init__(self, x, y, heading=None, w=1, noisy=False):
        if heading is None:
            heading = random.uniform(0, 360)
        if noisy:
            x, y, heading = add_some_noise(x, y, heading)

        self.x = x
        self.y = y
        self.h = heading
        self.w = w

    def __repr__(self):
        return "(%f, %f, w=%f)" % (self.x, self.y, self.w)

    @property
    def xy(self):
        return self.x, self.y

    @property
    def xyh(self):
        return self.x, self.y, self.h

    @classmethod
    def create_random(cls, count, maze):
        return [cls(*maze.random_free_place()) for _ in range(0, count)]

    def read_sensor(self, maze):
        """
        Find distance to nearest beacon.
        """
        return maze.distance_to_nearest_beacon(*self.xy)

    def advance_by(self, speed, checker=None, noisy=True):
        #chose_random_direction()
        h = self.h
        #checker=lambda r, dx, dy: maze.is_free(r.x+dx, r.y+dy)
        if noisy:
            speed, h = add_little_noise(speed, h)
            h += random.uniform(-30, 30) # needs more noise to disperse better
        r = math.radians(h)
        dx = math.sin(r) *  speed * random.uniform(-0.2,1)*0.6
        dy = math.cos(r) * speed* random.uniform(-0.2,1)*0.6
        if checker is None or checker(self, dx, dy):
            self.move_by(dx, dy)
            return True
        return False

    def move_by(self, x, y):
        self.x += x
        self.y += y

# ------------------------------------------------------------------------
class Robot(Particle):
    speed = 0.2

    def __init__(self, maze):
        super(Robot, self).__init__(*maze.random_free_place(), heading=90)
        self.chose_random_direction()
        self.step_count = 0

    def chose_random_direction(self):
        heading = random.uniform(0, 360)
        self.h = heading

    def read_sensor(self, maze):
        """
        Poor robot, it's sensors are noisy and pretty strange,
        it only can measure the distance to the nearest beacon(!)
        and is not very accurate at that too!
        """
        return add_little_noise(super(Robot, self).read_sensor(maze))[0]
    def position(self,x,y):
        self.x = x
        self.y = y
    def move(self, maze):
        """
        Move the robot. Note that the movement is stochastic too.
        """
        while True:
            self.step_count += 1
            if self.advance_by(self.speed, noisy=True,
                checker=lambda r, dx, dy: maze.is_free(r.x+dx, r.y+dy)):
                break
            # Bumped into something or too long in same direction,
            # chose random new direction
            self.chose_random_direction()

# ------------------------------------------------------------------------


def pf_process(particleNum, folderName):
    scale = 30.9
    grids = gg.generateGridPoints(scale, ["shape_description.shape"])
    xs,ys = mg.get_shape(grids)
    car_speed = 5
    import groundtruth as  GT
    import beaconLoc as bleLoc
    import beaconTable
    errorP = []
    gt,routes_tag = processGT(folderName+'/movements.dat')
    PARTICLE_COUNT = particleNum   # Total number of particles
    resultX = []
    resultY = []
        #--------------------load all the sensor readings--------------------------#
        # WiFi readings first
    beaconInfos = beaconTable.getBeaconInfo()
    beaconData =bleLoc. logToList2(folderName+"/ibeaconScanner.dat",beaconInfos)

        # INS data
    INSData = []
    sensorFile = open(folderName+'/IMUdata.dat')
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

    # initial distribution assigns each particle an equal probability
    particles = Particle.create_random(PARTICLE_COUNT, world)
    robbie = Robot(world)
    errorList = []
    start_time_pf = ti.time()
    while INSData and len(INSData)> 40:
        insBuffer = [INSData.pop(0) for i in range(21)]
        beaconBuffer = {}
        startTime = insBuffer[0]['ts']
        endTime = insBuffer[20]['ts']
        groundTruth = None
        while gt[0]['ts'] < startTime:
            groundTruth = gt.pop(0)
        #"here we need to show the ground truth of car"
        # Read robbie's sensor
        if groundTruth:
            gtMaze = mg.pixelToMaze(groundTruth['x'],groundTruth['y'],scale,xs[0],ys[0])
            #print(gtMaze)
            #print([groundTruth['x'],groundTruth['y']])
            robbie.position(gtMaze[0],gtMaze[1])
        if beaconData:
            beaconTimeCursor = beaconData[0]['ts']
        if abs(beaconTimeCursor - startTime) <= 1000 and  beaconData:
            beaconBuffer = beaconData.pop(0)
        if insBuffer[0]['INSDict']['orientation'] != []:
            yaw = insBuffer[0]['INSDict']['orientation'][0]
        yaw = direction( yaw , [ gt[1]['x'], gt[1]['y']],  [gt[0]['x'], gt[0]['y']], 1)+90
        
        #r_d = robbie.read_sensor(world)

        # Update particle weight according to how good every particle matches
        # robbie's sensor reading
        if beaconBuffer:
            beaconTable = beaconBuffer['table']
            for p in particles:
                #print(p.x,p.y)
                weight = 0
                for bleSignal in beaconTable:
                    blePos = [bleSignal['x'], bleSignal['y']]
                    rssi = bleSignal['rssi']
                    Loss = (abs(rssi)+40)/20
                    likelyhood = 1000*(10**(-Loss))
                    particlePixelPos = mg.mazeGridToPixel(p.x,p.y,scale,xs[0],ys[0])
                    #print("hello ", blePos)
                    dis = euclideanDistance(blePos,particlePixelPos)
                    #print(dis)
                    #weight += huber_gauss(dis,80)*likelyhood
                    weight += huber_gauss(dis,1009)*likelyhood
                p.w = weight
                #print(p.w)
        for p in particles:
            if not world.is_free(*p.xy):
                p.w = 0

        # ---------- Try to find current best estimate for display ----------
        m_x, m_y, m_confident = compute_mean_point(particles,PARTICLE_COUNT)
        #print(m_x,m_y)
        # ---------- Show current state ----------
        #world.show_particles(particles)
        #world.show_mean(m_x, m_y, m_confident)
        #world.show_robot(robbie)

        # ---------- Shuffle particles ----------
        new_particles = []

        # Normalise weights
        nu = sum(p.w for p in particles)
        if nu:
            for p in particles:
                p.w = p.w / nu

        # create a weighted distribution, for fast picking
        dist = WeightedDistribution(particles)

        for _ in particles:
            p = dist.pick()
            if p is None:  # No pick b/c all totally improbable
                new_particle = Particle.create_random(1, world)[0]
            else:
                new_particle = Particle(p.x, p.y,
                        heading=yaw,
                        noisy=True)
            new_particles.append(new_particle)

        particles = new_particles

        # ---------- Move things ----------
        #robbie.move(world)

        # Move particles according to my belief of movement (this may
        # be different than the real movement, but it's all I got)
        for p in particles:
            p.h = yaw# in case robot changed heading, swirl particle heading too
            p.advance_by(5)      
        if groundTruth:
            error = euclideanDistance(gtMaze,[m_x,m_y])
            errorList.append(error)

    errorListSorted = sorted(errorList)
    X =[]
    for i in range(len(errorList)):
        X.append(i / len(errorList))
    Y  = [i*0.2 for i in range(len(errorList))]
    #plt.plot(Y,errorList,color = 'red')
    rmsParticle  = rms(errorList)
    #plt.xticks([])
    #plt.ylim(0,0.07)
    #plt.xlim(0,3)
    #plt.plot([0,2],[rmsParticle,rmsParticle],linestyle='--',color='red')
    #plt.scatter(2,rmsParticle,s=50,color='red',marker='*')
    #plt.annotate("Particle Filter", xy=(2, rmsParticle), xytext = (1.8, rmsParticle+0.5)) 
    #plt.show()
    period_pf =(ti.time()-start_time_pf)/len(errorList)
    #plt.scatter(2,period_pf/len(errorList),s=50,color='red',marker='*')
    #plt.annotate("Particle Filter", xy=(2, period_pf/len(errorList)), xytext = (1.8,period_pf/len(errorList)+0.002))
    #plt.plot([0,2],[period_pf/len(errorList),period_pf/len(errorList)],linestyle='--',color='red')
    return errorList,period_pf

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    computationTime = []
    rmsError = []
    particleNum = 2000
    error,period_pf = pf_process(particleNum,'')
    plt.plot(error)
    plt.show()
