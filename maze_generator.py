import grid_generator as gg
import numpy as np
import random
#695.2036  2227.3872
#695.1987  1830.3673
#1100.9381  1830.4353
#1100.6505  2227.0947

def euclideanDistance(posA, posB):
    vector1 = np.array(posA[0:2])
    vector2 = np.array(posB[0:2])
    dis=np.linalg.norm(vector1-vector2)
    return dis

def mazeGridToPixel(x,y,scale,x_bias, y_bias):
    return [x * scale + x_bias,  y*scale +y_bias]

def pixelToMaze(x,y,scale,x_bias, y_bias):
    return[int((x-x_bias)/scale), int((y-y_bias)/scale)]

def mazeToPixels(maze,scale,x_bias, y_bias):
    pixels = []
    for j in range(len(maze)):
        for i in range(len(maze[0])):
            pixels.append([j*scale+x_bias, i*scale+y_bias])
    return pixels

def isValidPoint(candidatePoint, valid_points,scale):
    for point in valid_points:
        if euclideanDistance(point,candidatePoint)<= scale * 3:
            return True
    return False

def generateMazeData(scale):
    grids = gg.generateGridPoints(scale, ["shape_description.shape"])
    xs,ys = get_shape(grids)
    valid_points = gg.quickDelete(grids, scale * 0.4)
    debug = False
    if debug:
        import matplotlib.pyplot as plt
        for grid in valid_points:
            plt.scatter(grid[0], grid[1],color='blue')
        plt.show()  
        #plt.clf()
    maze = []
    y_cursor = ys[0]
    while y_cursor < ys[1]:
        bufferRaw = []
        x_cursor = xs[0]
        while x_cursor < xs[1]:
            if isValidPoint([x_cursor,y_cursor],valid_points,scale):

                bufferRaw.append(0)
            else:
                bufferRaw.append(1)
            x_cursor+= scale
        maze.append(bufferRaw)
        y_cursor+= scale
    return maze

def get_shape(grids):
    x = []
    y = []
    xs = []
    ys = []
    for grid in grids:
        x.append(grid[0])
        y.append(grid[1])
    xs.append(min(x)-min(x)*0.2)
    xs.append(max(x)+max(x)*0.2)
    ys.append(min(y)-min(y)*0.2)
    ys.append(max(y)+max(y)*0.2)
    return xs,ys
if __name__=='__main__':
    debug = True
    scale = 30.9
    grids = gg.generateGridPoints(scale, ["shape_description.shape"])
    valid_points = gg.quickDelete(grids, scale*0.4)
    xs,ys = get_shape(valid_points)
    maze = generateMazeData(scale)

    pixels = mazeToPixels(maze,scale,xs[0],ys[0])
    #print(pixels)
    if debug:
        import matplotlib.pyplot as plt
        for grid in pixels:
            if isValidPoint(grid,valid_points,scale):
                plt.scatter(grid[0], grid[1],color='red')
            else:
                plt.scatter(grid[0], grid[1],color='blue')
        plt.show()  
