
# input parameters: batches of data including SPEED, BLE, Orientation and Turn
# output parameters: speed distribution, error distribution
# 用词一定要准确,避免被Gary骂,图要正确,符合规则.
# 明早把规则给定好,写在如下的框内:

#-------------------------------------------------------------------------------------
# 1. 真实的data点要突出出来
# 2. 图不要结束
# 3. Legend 不要挡住 图片
# 4. Legend 要跟实际curve 的顺序要一致
# 5. 即使黑白打印也可以看出来

# 6.坐标轴的刻度要放对

# 7. 图不要平淡无奇,要包含信息

# 8. 

# plot X overall everage speed distribution ---- speed constraints 
# plot 1,2 average speed distribution for car and driver heterogenity: different car and driver contribute to the speed distribution
############ To prove that 1) Speed constrain 2) driver heterogentity affect the speed distribution, but not much 
 
# plot X average speed among different road segments :  
# plot 3,4 average speed among different road segments with car and dirver heterogenity
######### To prove that the overall trend of speed among different road segment is similar, but we do have driver heterogenity.######### 

# plot 5 3D draw: speed distribution among different road segments
###### To prove that 车在不同的路段有着不同的速度.


# plot X overall BLE CDF. 
# plot 6 7  BLE CDF with car and phone heterogenity



# plot X BLE RMS error among different road points    
# plot 8 9 BLE RMS error among different road points with car and driver heterogenity.
########## 1011 1012 一起plot,证明不管什么车子或者什么手机,一些地方定位就是准,另一些地方定位就是不准


# plot 10 11 orientation error CDF with car and phone heterogenity
### 猜测: 不同手机可能会有不同,不同车可能也会不同,但相差不会过大,也不好说,可能室内停车场偏差真挺大


#-------------------------------------------------------------------------------------
# Car park engine
#plot 12 13 14 overall performance CDF
#Plot 15, cross validation, test Device heterogenity 的影响: Overall, Device 1,  Device 2, Device 3 .... 
#Plot 16, cross validation, Driver heterogenity 的影响: Overall, driver 1, driver 2, driver 3 .... 
#Plot 17, cross validation, car heterogenity 的影响



##### training data的transfer test: ######### 
# plot 18 Training phone 12, test 3, 3 sub-plots 
# plot 19 Training driver 123, test 4, 4 sub-plots
# plot 20 Training car 1, test 2, 2 sub-plots

# plot 20*+ : maybe we need more plots 


#### Performance over training-data size:
# Cross validation plot 21 22 RMS 一张, CDF一张 
import math
def pdf_cdf_plot(arr):
    from scipy import stats
    import matplotlib.pyplot as plt
    import numpy as np
    #import seaborn as sns

    #arr = np.random.normal(size=10000)
    #print(arr)
    hist, bin_edges = np.histogram(arr,bins=10)
    #print(hist)
    #print(bin_edges)
    width = (bin_edges[1] - bin_edges[0]) * 0.8
    plt.bar(bin_edges[1:], hist/max(hist), width=width, color='#5B9BD5')
    #width = (bin_edges[1] - bin_edges[0]) * 0.8
    #plt.plot(bin_edges[1:], hist/max(hist), color='#5B9BD5')

    cdf = np.cumsum(hist/sum(hist))
    plt.plot(bin_edges[1:], cdf, '-*', color='#ED7D31')
    plt.grid()
    plt.xlim(0,10)
    plt.show()

def checkFolders():
    folder = []
    import os
    dir = './'
    dbtype_list = os.listdir(dir)
    for dbtype in dbtype_list:
        if dbtype.find('2021') == 0:
            folder.append(dbtype)
    folder =sorted(folder,key=str.lower)
    return folder

# given a start point and end point, try to enrich the ground truth locations
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
            ground_truth.append(node)
    assert(route_cursor == 5)
    return ground_truth, routes_tag
    # Merge it as road segments 
def training_process():
    # train the average speed distribution
    # train the iBeacon localization error
    # train the average INS error
    # train the turn detection accuracy
    print("pending implementation")

def get_info():
    general_information = []
    for i in range(1,65):
        info = {}
        info['tag'] = i
        if i >= 1 and i <= 10:
            info['phone'] = 'Sumsung'
            info['car'] = 'Honda'  
            info['driver'] = 'D1'
        elif i >= 11 and i <= 20:
            info['phone'] = 'vivo'
            info['car'] = 'Honda'  
            info['driver'] = 'D1'
        elif i >= 21 and i <= 30:
            info['phone'] = 'Huawei'
            info['car'] = 'Honda'  
            info['driver'] = 'D1'     
        elif i >= 31 and i <= 40:
            info['phone'] = 'vivo'
            info['car'] = 'Honda'  
            info['driver'] = 'D2'
        elif i >= 41 and i <= 46:
            info['phone'] = 'vivo'
            info['car'] = 'BMW'  
            info['driver'] = 'D3'
        elif i >= 47 and i <= 52:
            info['phone'] = 'vivo'
            info['car'] = 'BMW'  
            info['driver'] = 'D1'
        elif i >= 53 and i <= 58:
            info['phone'] = 'vivo'
            info['car'] = 'BMW'  
            info['driver'] = 'D4'
        elif i>=59 and i<= 64:
            info['phone'] = 'vivo'
            info['car'] = 'Hyundai'  
            info['driver'] = 'D1'
        general_information.append(info)
    return general_information

def add_speed(speed_data, routes_tag):
    for i in range(5):
        sp = routes_tag[i]['speed']
        if sp <= 10:
            speed_data[i].append(sp)
    return speed_data

if __name__ == '__main__':
    #### experiment information ##### 
    #Honda Hyundai BMW
    

    #a = {'x':1, 'y':1, 'ts':1000}
    #b = {'x':2, 'y':1, 'ts':3000}
    #pos = expandPoints(a,b, 0.2)
    #print(pos)
    # check all folders first 
    geneal_info = get_info()
    folderList = checkFolders()
    ###### speed 
    speed_data  = {}
    for i in range(5):
        speed_data[i] = []
    for folder in folderList:
        print("processing: " + folder)
        points,routes_tag = processGT(folder+'/movements.dat')
        add_speed(speed_data, routes_tag)
    for i in range(5):
        pdf_cdf_plot(speed_data[i])
    #######
    #pdf_cdf_plot()
    '''
    from matplotlib import pyplot as plt 
    result = [3.29,2.91,4.51,4.02,3.26]
    plt.bar(range(len(result)), result)
    x = [-0.5,4.5]
    y = [2.7,2.7]
    optimal = plt.plot(x,y,color='r',linestyle='--')
    #plt.legend([optimal],['optimal parameter settings'])
    plt.show()
    '''