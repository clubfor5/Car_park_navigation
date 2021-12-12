import json
import math
import sys

def modifyTimeSlot(lines):
    newLines = []
    for i in range(len(lines)):
        data = lines[i].split()
        a = (data[2])
        gyroData = json.loads(a)
        time = data[1]
        time = int(time)
        timeReplace = time - time % 10
        newLines.append(lines[i].replace(str(time), str(timeReplace)))
    return newLines

# Transfer from raw data to dictionary
def get_sensor_data(sensor_file_name): 
    sensorFile = open(sensor_file_name)
    sensor_dataset = {}
    # Load gyro data
    lines = sensorFile.readlines()
    lines = modifyTimeSlot(lines)
    for line in lines:
        data = line.split()
        a = (data[2])
        sensor_field = json.loads(a)
        time = data[1]
        time = int(time)
        sensor_data = sensor_field['vals']
        sensor_dataset[str(time)] =  sensor_data 
    sensorFile.close()
    return sensor_dataset

def getTime(line):
    data = line.split()
    a = (data[2])
    sensor_field = json.loads(a)
    time = data[1]
    return int(time)

if __name__ == '__main__':
    folderName = sys.argv[1]
    sensorID = {'acc': '0', 'gyro': '3', 'mag':'1', 'orientation':'2' }
    gyroData = get_sensor_data(folderName+'/'+sensorID['gyro']+'.dat')
    accData = get_sensor_data(folderName+'/'+sensorID['acc']+'.dat')
    magData =get_sensor_data(folderName+'/'+sensorID['mag']+'.dat')
    orientationData = get_sensor_data(folderName+'/'+sensorID['orientation']+'.dat')
    IMUData = {}

    dataFile = open(folderName+'/'+'IMUdata.dat','w')
    sensorFile = open(folderName+'/'+sensorID['gyro']+'.dat')
    lines = sensorFile.readlines()
    lines = modifyTimeSlot(lines)
    startTime = getTime(lines[0])
    endTime = getTime(lines[-1])
    timeSlot = startTime
    #dataFile.write(str(data))
    while timeSlot < endTime:
        IMUPack = {}
        if str(timeSlot ) in orientationData:
            #print("hello")
            IMUPack['orientation'] = orientationData[str(str(timeSlot ))]
        else:
            IMUPack['orientation'] = []
        if str(timeSlot ) in magData:
            IMUPack['mag'] = magData[str(timeSlot )]
        else:
            IMUPack['mag'] = []
        if str(timeSlot ) in gyroData:
            IMUPack['gyro'] = gyroData[str(timeSlot )]    
        else:
            IMUPack['gyro'] = []
        if str(timeSlot ) in accData:
            IMUPack['acc'] = accData[str(timeSlot )]
        else:
            IMUPack['acc'] = []
        IMUData[str(timeSlot )] = IMUPack
        dataFile.write(str(timeSlot)+';'+json.dumps(IMUPack)+'\n')
        timeSlot  += 10
    #print(IMUData)
    dataFile.close()