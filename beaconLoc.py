import json
import re
import beaconTable
def beacon_locs(beaconInfos, rssiTable):
    if not rssiTable['table']:
        return rssiTable['ts'],-10000,-10000
    #print(rssiTable)
    maxMac, smaxMac,maxRSSI,smaxRSSI = findMaxMac(rssiTable['table'])
    #print(maxMac,smaxMac,maxRSSI,smaxRSSI)
    #print(beaconInfos)
    if not maxMac:
        return rssiTable['ts'],-10000,-10000 
    for beacon in beaconInfos:
        if beacon['mac'] == maxMac:
            maxPositionX = beacon['x']
            maxPositionY = beacon['y']
        if beacon['mac'] == smaxMac:
            smaxPositionX = beacon['x']
            smaxPositionY = beacon['y']

    if smaxRSSI == -160:
        return rssiTable['ts'],maxPositionX,maxPositionY

    if maxRSSI - smaxRSSI  >= 6:
        positionX = maxPositionX
        positionY = maxPositionY        
    elif maxRSSI - smaxRSSI  >= 3:
        positionX = maxPositionX * 0.75 + smaxPositionX * 0.25
        positionY = maxPositionY * 0.75 + smaxPositionY * 0.25
    else:
        positionX = maxPositionX * 0.5 + smaxPositionX * 0.5
        positionY = maxPositionY * 0.5 + smaxPositionY * 0.5 
    return rssiTable['ts'],positionX,positionY
def findMaxMac(rssiTable):
    maxRSSI = -150
    smaxRSSI = -160
    maxMac = ""
    smaxMac=""
    for table in rssiTable: 
        mac = table['mac']
        rssi = table['rssi']
        if rssi > maxRSSI:
            maxRSSI = rssi
            maxMac = mac
        elif rssi > smaxRSSI:
            smaxRSSI = rssi
            smaxMac = mac
    return maxMac,smaxMac,maxRSSI,smaxRSSI


def logToList(fileName):
    file = open(fileName)
    lines = file.readlines()
    beaconLog = []
    for line in lines:
        info = line.split('\t')
        timeStamp = float(info[1])
        #print(info[2])
        temp = info[2].replace('[','')
        temp = temp.replace(']','')
        temp = temp.replace('{','')
        beaconRSSI = temp.split('}')
        #beaconRSSI = re.findall(r'[(](.*?)[)]', info[2]) 
        #print(beaconRSSI)
        # scan every beacon
        beaconList = []
        for beacon in beaconRSSI:
            beaconTemp = beacon.split(',')
            buffer = {}
            for  bt in beaconTemp:
                if bt.find('mac')!= -1:
                    buffer['mac'] = bt.replace('\"mac\":','')
                if bt.find('rssi')!= -1:
                    buffer['rssi'] = float(bt.split(':')[1])
            if buffer:
                sameMac = False
                for beacon in beaconList:
                    if beacon['mac'] == buffer['mac']:
                        sameMac = True
                if not sameMac:

                    beaconList.append(buffer)
        beaconDic = {}
        beaconDic['ts'] = timeStamp
        beaconDic['table'] = beaconList
        beaconLog.append(beaconDic)
    
    return beaconLog


def logToList2(fileName,beaconInfos):
    file = open(fileName)
    lines = file.readlines()
    beaconLog = []
    for line in lines:
        info = line.split('\t')
        timeStamp = float(info[1])
        #print(info[2])
        temp = info[2].replace('[','')
        temp = temp.replace(']','')
        temp = temp.replace('{','')
        beaconRSSI = temp.split('}')
        #beaconRSSI = re.findall(r'[(](.*?)[)]', info[2]) 
        #print(beaconRSSI)
        # scan every beacon
        beaconList = []
        for beacon in beaconRSSI:
            beaconTemp = beacon.split(',')
            buffer = {}
            newBuffer = {}
            for  bt in beaconTemp:
                if bt.find('mac')!= -1:
                    buffer['mac'] = bt.replace('\"mac\":','')
                if bt.find('rssi')!= -1:
                    buffer['rssi'] = float(bt.split(':')[1])
            if buffer:
                sameMac = False
                for beacon in beaconList:
                    if 'mac' in beacon.keys() and beacon['mac'] == buffer['mac']:
                        sameMac = True
                if not sameMac:
                    for bleInfo in beaconInfos:
                        if buffer['mac'] == bleInfo['mac']:
                            newBuffer['x'] = bleInfo['x']
                            newBuffer['y'] = bleInfo['y']
                            newBuffer['rssi'] = buffer['rssi']
                            beaconList.append(newBuffer)
        beaconDic = {}
        beaconDic['ts'] = timeStamp
        beaconDic['table'] = beaconList
        if beaconList:
            beaconLog.append(beaconDic)
    
    return beaconLog


if __name__ == '__main__':
    import matplotlib.pyplot as plt        
    beaconLog = logToList("ibeaconScanner.dat")
    for beacon in beaconLog:
        print(beacon)
    beaconInfos = beaconTable.getBeaconInfo()
    ts_pre = 0
    for rssiTable in beaconLog:
        ts,locx,locy = beacon_locs(beaconInfos, rssiTable)
        if locx!=-10000:
            print(ts, locx,locy)
            plt.xlim(600,1100)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
            plt.ylim(-2332,-1835)
            plt.scatter(locx, -locy)

        else:
            print(ts, {})
            continue
        if ts_pre != 0:
            plt.pause((ts-ts_pre)/1000)
            #plt.clf()
        ts_pre = ts
