
def getBeaconInfo():
    beaconInfos = []
    beaconFile = open('beaconTable.cfg')
    message = beaconFile.readlines()
    for beaconMsg in message:
        #print(beacon)
        beaconOperator = beaconMsg.split(',')
        beacon = {}
        beacon['tag'] = int(beaconOperator[0])
        beacon['x']  = float(beaconOperator[1])
        beacon['y'] = float(beaconOperator[2])
        beacon['mac'] = beaconOperator[3].replace('\n','')
        beaconInfos.append(beacon)
    return beaconInfos 
if __name__ == '__main__':
    #### Load things from the config file
	beaconInfos = getBeaconInfo()
	for beacon in beaconInfos:
		print(beacon['mac'], beacon['x'], beacon['y'])

