from matplotlib import pyplot as plt 
import math
def gauss(error,sigma):
    g = math.e ** -(error ** 2 / (2 * sigma*sigma))
    return g

def huber_gauss(error, sigma):
    if abs(error) < abs(sigma):
        Loss =  1/2 * error ** 2 
    else:
        Loss =   sigma*(abs(error)- 1/2*sigma)
    g =  math.e ** (-Loss/(sigma*sigma))  
    return g

error = [15.7,2.64,4.75]
ct = [0.001, 0.007,0.0338]
plt.xlim(0,0.05)
plt.ylim(0,18)
# iBeacon WCL
plt.scatter(ct[0],error[0], color='green',marker='D')
plt.annotate("iBeacon WCL", xy = (ct[0], error[0]), xytext = (ct[0], error[0]+0.5)) 
plt.plot([0,ct[0]],[error[0],error[0]],linestyle='--',color='green')
plt.plot([ct[0],ct[0]],[0,error[0]],linestyle='--',color='green')
# HMM 
plt.scatter(ct[1],error[1],color='blue',marker='*')
plt.annotate("Proposed HMM", xy = (ct[1], error[1]), xytext = (ct[1], error[1]+0.5)) 
plt.plot([0,ct[1]],[error[1],error[1]],linestyle='--',color='blue')
plt.plot([ct[1],ct[1]],[0,error[1]],linestyle='--',color='blue')

plt.scatter(ct[2],error[2],color='red')
plt.annotate("Particle Filter", xy = (ct[2], error[2]), xytext = (ct[2], error[2]+0.5)) 
plt.plot([0,ct[2]],[error[2],error[2]],linestyle='--',color='red')
plt.plot([ct[2],ct[2]],[0,error[2]],linestyle='--',color='red')
plt.show()
X= [-1+ i*0.01 for i in range(201)]
Y1 = []
Y2 = []
for i in range(201):
    Y1.append(gauss(X[i],0.2))
    Y2.append(huber_gauss(X[i],0.2))

plt.ylim(0,1.2)
plt.xlim(-1,1)
plt.plot(X,Y2,color = 'blue')
plt.plot(X,Y1,color = 'red')
plt.show()
