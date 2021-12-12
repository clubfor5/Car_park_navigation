import numpy as np 
import math
def generateGridPoints(gridDistanceRequired, shapeFileNames):
    grids = []
    for name in shapeFileNames:
        shape_file = open(name)
        vertex_raw = shape_file.readlines()
        vertexs = []
        for line in vertex_raw:
             pos = line.split('  ') 
             vertexs.append([float(pos[0]),float(pos[1])])
        if len(vertexs) <= 2: 
            print("invalid path!")
            return "error"
        for i in range(len(vertexs)-1):
            currentNode = vertexs[i]
            nextNode = vertexs[i+1]
            gridPoints = path_to_grids(currentNode, nextNode, gridDistanceRequired)
            for point in gridPoints:
                grids.append([point[0], point[1],i])
    return grids

def quickDelete(grids, distanceThreshold):
    newGrids = []
    newGrids.append(grids[0])
    for grid in grids:
        repeat = False
        for newGrid in newGrids:
            if euclidean(grid[0:2],newGrid[0:2]) < distanceThreshold:
                repeat = True
                continue
        if not repeat:
            newGrids.append(grid)
    return newGrids
    #for i in range(grids):



            
def euclidean(a,b):
    point_a = np.array(a)
    point_b = np.array(b)
    distance = np.linalg.norm(point_a - point_b)
    return distance

def path_to_grids(pathStart,pathEnd,gridDistanceRequired):
    point_a = np.array(pathStart)
    point_b = np.array(pathEnd)
    distance = np.linalg.norm(point_a - point_b)
    if gridDistanceRequired > 0:
        num_of_grids = int(distance / gridDistanceRequired) + 1
    else: 
        print("invalid grid distance") 
        return []
    actual_grid_distance = distance / (num_of_grids-1)
    step_x = (pathEnd[0] - pathStart[0]) /  distance * actual_grid_distance 
    step_y = (pathEnd[1]- pathStart[1]) / distance * actual_grid_distance 
    #print(step_x,step_y)
    grids = []
    currentX = pathStart[0]
    currentY = pathStart[1]
    while(len(grids)< num_of_grids): 
        grids.append([currentX,currentY])
        currentX += step_x
        currentY += step_y
    return grids


if __name__ == '__main__':
    #grids, actual_grid_distance = path_to_grids([100,0], [0,-100], 10)
    #print(grids)
    #print(actual_grid_distance)
    debug = True
    grids = generateGridPoints(13.33, ["shape_description.shape"])
    gridsNew = quickDelete(grids, 5)
    #print(grids)
    print(gridsNew)
    out_file = open("grid_points.shape", 'w')
    for grid in gridsNew:
        out_line = str(grid[0]) + ',' + str(grid[1]) + '\n'
        out_file.writelines(out_line)
    if debug:
        import matplotlib.pyplot as plt
        for grid in gridsNew:
            plt.scatter(grid[0], -grid[1])
        plt.show()    
