import numpy as np
from numpy.core.defchararray import title
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation
from numpy.linalg import norm
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
from scipy.interpolate import UnivariateSpline
import math
from vpython import cylinder, vector
from scipy.interpolate import interp1d
import numpy.matlib
import random
import copy

class model():
    def __init__(self):
        self.xobs = []
        self.yobs = []
        self.zobs = []
        self.robs = []
        self.hobs = []
        self.nobs = []
        self.n = []
        self.xmin = []
        self.xmax = []
        self.ymin = []
        self.ymax = []
        self.zmin = []
        self.zmax = []
        self.obstBuffer = []
        self.nUAVs = 0

    def update_param(self, xobs, yobs, zobs, robs, hobs, nobs, n, xmin, xmax, ymin, ymax, zmin, zmax, obstBuffer, xs, ys, zs, xt, yt, zt):
        self.xobs = xobs
        self.yobs = yobs
        self.zobs = zobs
        self.robs = robs
        self.hobs = hobs
        self.nobs = nobs
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.obstBuffer = obstBuffer
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.xt = xt
        self.yt = yt
        self.zt = zt
        self.nUAVs = len(xs)



class Position():
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []

class Velocity():
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
'''
class VarMin():
    def __init__(self):
        self.VarMin = Position()
        #self.x = []
        #self.y = []
        #self.z = []

class VarMax():
    def __init__(self):
        self.VarMax = Position()
        #self.x = []
        #self.y = []
        #self.z = []

class VelMax():
    def __init__(self):
        self.VelMax = Velocity()
        #self.x = []
        #self.y = []
        #self.z = []

class VelMin():
    def __init__(self):
        self.VelMin = Velocity()
        #self.x = []
        #self.y = []
        #self.z = []

class Sol():
    def __init__(self):
        self.Violation = []
        self.IsFeasible = (self.Violation==0)
'''       

class Best():
    def __init__(self):
        self.Position = Position()
        self.Velocity = Velocity()
        self.Cost = math.inf
        self.PathLength = []
        self.Sol = sol2()

class empty_particle():
    def __init__(self):
        self.Position = Position()
        self.Velocity = Velocity()
        self.Cost = []
        self.PathLength = []
        self.Sol = sol2()
        self.Best = Best()

class GlobalBest():
    def __init__(self):
        self.Cost = math.inf
        self.PathLength = []
        self.Best = Best()
        self.Position = Position()
        self.Sol = sol2()

'''
class sol1():
    def __init__(self):
        self.sol1 = Position()
        #sol1.x = x
        #sol1.y = y
        #sol1.z = z
'''

class sol2():
    def __init__(self):
        sol2.TS = []
        sol2.XS = []
        sol2.YS = []
        sol2.ZS = []
        sol2.tt = []
        sol2.xx = []
        sol2.yy = []
        sol2.zz = []
        sol2.dx = []
        sol2.dy = []
        sol2.dz = []
        sol2.L = [] 
        sol2.Violation = []
        sol2.IsFeasible = (sol2.Violation==0)

    def update_param(self, TS, XS, YS, ZS, tt, xx, yy, zz, dx, dy, dz, L, Violation):
        sol2.TS = TS
        sol2.XS = XS
        sol2.YS = YS
        sol2.ZS = ZS
        sol2.tt = tt
        sol2.xx = xx
        sol2.yy = yy
        sol2.zz = zz
        sol2.dx = dx
        sol2.dy = dy
        sol2.dz = dz
        sol2.L = L 
        sol2.Violation = Violation
        sol2.IsFeasible = (sol2.Violation==0)

class path_generation():
    '''
    input: 
    obstacles profiles [x,y,z,h,r]
    agent positions [xs,ys,zs]
    target position [xt,yt,zt]
    look_ahead_num: constant number

    output: 
    waypoints [x,y,z]
    '''

    def __init__(self):
        self.model = model()
        #self.particle = empty_particle()
        self.empty_particle = empty_particle()
        self.VarMin = Position()
        self.VarMax = Position()
        self.VelMax = Velocity()
        self.VelMin = Velocity()
        self.sol1 = Position()
        self.sol2 = sol2()
        nPop = 150
        self.particle = np.matlib.repmat(self.empty_particle,nPop,1)
        self.GlobalBest = GlobalBest()
        self.temp_particle = empty_particle()
        

        # self.pos = []
        #self.rad = []
        #self.fake_rad = []
        #self.height = []

    def pso(self, xobs, yobs, zobs, robs, hobs, nObs, xmin, xmax, ymin, ymax, zmin, zmax, xs, ys, zs, xt, yt, zt):
        '''
        This function generates path waypoints for agents
        '''

        droneSideLenght = 0.15
        obstBuffer = droneSideLenght*1.5
        nUAVs = len(xs) 

        # Number of intermediate way points
        # n = max(math.ceil(nObs/5)*3, 3)
        n = 3

        self.model.update_param(xobs, yobs, zobs, robs, hobs, nObs, n, xmin, xmax, ymin, ymax, zmin, zmax, obstBuffer, xs, ys, zs, xt, yt, zt)
        env = self.model
        # env = self.CreateModel(nObs, obstBuffer)
        
        #figure(1)

        # self.model = self.getGoals(env)

        nVar = self.model.n # Number of Decision Variables
        
        VarSize = [1, nVar] # Size of Decision Variable Matrix

        self.VarMin.x = self.model.xmin # Lower Bound of Variables
        self.VarMax.x = self.model.xmax # Upper Bound of Variables
        self.VarMin.y = self.model.ymin
        self.VarMax.y = self.model.ymax
        self.VarMin.z = self.model.zmin
        self.VarMax.z = self.model.zmax

        P_reloc_obs = 0.6 # Relocation probability of obstacles
        Maxstep_reloc_obs = 1 # max step obstacles are relocated from their current positions (assume moving on xy plane)
        P_reloc_t = 0 # Relocation probability of target

        '''
        PSO Parameters
        '''
        self.MaxIt = 2 # Maximum Number of Iterations
        time = np.zeros((1,self.MaxIt))
        nPop = 150 # population size (swarm size)
        w = 1 # inertia weight
        wdamp = 0.98 # inertia weight damping ratio
        c1 = 1.5 # personal learning coefficient
        c2 = 2 # Global Learning Coefficient

        alpha = 0.1
        self.VelMax.x = alpha*(self.VarMax.x-self.VarMin.x) # Maximum Velocity
        self.VelMin.x = -self.VelMax.x                # Minimum Velocity
        self.VelMax.y =  alpha*(self.VarMax.y-self.VarMin.y) # Maximum Velocity
        self.VelMin.y = -self.VelMax.y                # Minimum Velocity
        self.VelMax.z =  alpha*(self.VarMax.z-self.VarMin.z) # Maximum Velocity
        self.VelMin.z = -self.VelMax.z                # Minimum Velocity        
        
        '''
        empty_particle.Position = []
        empty_particle.Velocity = []
        empty_particle.Cost = []
        empty_particle.PathLength = []
        empty_particle.Sol = []
        empty_particle.Best.Position = []
        empty_particle.Best.Cost = []
        empty_particle.Best.Sol = []
        empty_particle.Best.PathLength = []
        '''

        # Initialize Global Best
        self.GlobalBest.Best.Cost = math.inf
        # GlobalBest.Cost = math.inf

        # Create Particles Matrix
        # self.particle = np.matlib.repmat(self.empty_particle,nPop,1)

        # Initialization Position
        for i in range(nPop):
            # print(i)
            if i > 0:
                self.particle[i][0].Position = self.CreateRandomSolution(self.model,self.particle[0][0].Position)
            else:
                for j in range(self.model.nUAVs):
                    #straight line from source to destination
                    xx = np.linspace(self.model.xs[j],self.model.xt[j],self.model.n+2)
                    yy = np.linspace(self.model.ys[j],self.model.yt[j],self.model.n+2)
                    zz = np.linspace(self.model.zs[j],self.model.zt[j],self.model.n+2)
                    self.particle[i][0].Position.x.extend((xx[1:-1]).tolist())
                    # print('self.particle[i][0].Position.x',self.particle[i][0].Position.x) 
                    self.particle[i][0].Position.y.extend((yy[1:-1]).tolist())
                    # print('self.particle[i][0].Position.y',self.particle[i][0].Position.y) 
                    self.particle[i][0].Position.z.extend((zz[1:-1]).tolist())
                    # print('self.particle[i][0].Position.z',self.particle[i][0].Position.z) 

            # Initialize Velocity
            # self.particle[i][0].Velocity.x = np.zeros(VarSize)
            # self.particle[i][0].Velocity.y = np.zeros(VarSize)
            # self.particle[i][0].Velocity.z = np.zeros(VarSize)
            self.particle[i][0].Velocity.x = np.zeros((1,VarSize[1]*self.model.nUAVs))[0] #[[0.,0.,0.]][0] = [0,0,0]
            self.particle[i][0].Velocity.y = np.zeros((1,VarSize[1]*self.model.nUAVs))[0]
            self.particle[i][0].Velocity.z = np.zeros((1,VarSize[1]*self.model.nUAVs))[0]
            # print(self.particle[i][0].Position.x)
            # print(self.particle[i][0].Position.y)
            # print(self.particle[i][0].Position.z)
            # Evaluation
            [self.particle[i][0].Cost, self.particle[i][0].PathLength, self.particle[i][0].Sol] = self.MyCost(self.particle[i][0].Position,self.model)
            #Update Personal Best
            self.particle[i][0].Best.Position.x = self.particle[i][0].Position.x.copy()
            self.particle[i][0].Best.Position.y = self.particle[i][0].Position.y.copy()
            self.particle[i][0].Best.Position.z = self.particle[i][0].Position.z.copy()

            self.particle[i][0].Best.Cost = self.particle[i][0].Cost.copy()
            self.particle[i][0].Best.Sol = self.particle[i][0].Sol 
            self.particle[i][0].Best.PathLength = self.particle[i][0].PathLength.copy()

            # Update Global Best
            if self.particle[i][0].Best.Cost < self.GlobalBest.Best.Cost:
                self.GlobalBest.Best = copy.deepcopy(self.particle[i][0].Best)
                print('self.GlobalBest.Best.Cost at initilization ',self.GlobalBest.Best.Cost)

        # Array to hold best cost values at each iteration
        BestCost = np.zeros((self.MaxIt,1))
        BestPathLength = np.zeros((self.MaxIt,1))

        # PSO Main Loop
        model_update = self.model
        for it in range(self.MaxIt):
            for i in range(nPop):
                # print(i)
                # x part
                # update velocity
                self.particle[i][0].Velocity.x = w*np.array(self.particle[i][0].Velocity.x) + \
                    c1*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.particle[i][0].Best.Position.x)-np.array(self.particle[i][0].Position.x)))+ \
                                                c2*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.GlobalBest.Best.Position.x)-np.array(self.particle[i][0].Position.x)))

                # Update velocity bounds
                self.particle[i][0].Velocity.x = np.maximum(self.particle[i][0].Velocity.x, self.VelMin.x)
                self.particle[i][0].Velocity.x = np.minimum(self.particle[i][0].Velocity.x, self.VelMax.x)

                # Update Position
                self.particle[i][0].Position.x = self.particle[i][0].Position.x + self.particle[i][0].Velocity.x
                # Velocity Mirroring
                # OutofTheRange = (self.particle[i][0].Position.x < self.VarMin.x or self.particle[i][0].Position.x > self.VarMax.x)
                OutofTheRange = (self.IsArr1Smaller(self.particle[i][0].Position.x, self.VarMin.x) or self.IsArr1Larger(self.particle[i][0].Position.x, self.VarMax.x))
                if OutofTheRange == True:
                    self.particle[i][0].Velocity.x = -self.particle[i][0].Velocity.x

                # Update Position Bounds
                self.particle[i][0].Position.x = np.maximum(self.particle[i][0].Position.x, self.VarMin.x)
                self.particle[i][0].Position.x = np.minimum(self.particle[i][0].Position.x, self.VarMax.x)
                # print(self.particle[i][0].Position.x)
                # print(self.particle[i][0].Velocity.x)

                # y part
                 # update velocity
                self.particle[i][0].Velocity.y = w*self.particle[i][0].Velocity.y + \
                    c1*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.particle[i][0].Best.Position.y)-np.array(self.particle[i][0].Position.y)))+ \
                        c2*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.GlobalBest.Best.Position.y)-np.array(self.particle[i][0].Position.y)))

                # Update velocity bounds
                self.particle[i][0].Velocity.y = np.maximum(self.particle[i][0].Velocity.y, self.VelMin.y)
                self.particle[i][0].Velocity.y = np.minimum(self.particle[i][0].Velocity.y, self.VelMax.y)

                # Update Position
                self.particle[i][0].Position.y = self.particle[i][0].Position.y + self.particle[i][0].Velocity.y

                # Velocity Mirroring
                OutofTheRange = (self.IsArr1Smaller(self.particle[i][0].Position.y, self.VarMin.y) or self.IsArr1Larger(self.particle[i][0].Position.y, self.VarMax.y))

                # OutofTheRange = (self.particle[i][0].Position.y < self.VarMin.y or self.particle[i][0].Position.y > self.VarMax.y)
                if OutofTheRange == True:
                    self.particle[i][0].Velocity.y = -self.particle[i][0].Velocity.y

                # Update Position Bounds
                self.particle[i][0].Position.y = np.maximum(self.particle[i][0].Position.y, self.VarMin.y)
                self.particle[i][0].Position.y = np.minimum(self.particle[i][0].Position.y, self.VarMax.y)    

                # z Part
                # update velocity
                self.particle[i][0].Velocity.z = w*self.particle[i][0].Velocity.z + \
                    c1*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.particle[i][0].Best.Position.z)-np.array(self.particle[i][0].Position.z)))+ \
                        c2*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.GlobalBest.Best.Position.z)-np.array(self.particle[i][0].Position.z)))

                # Update velocity bounds
                self.particle[i][0].Velocity.z = np.maximum(self.particle[i][0].Velocity.z, self.VelMin.z)
                self.particle[i][0].Velocity.z = np.minimum(self.particle[i][0].Velocity.z, self.VelMax.z)
                
                # Update Position
                self.particle[i][0].Position.z = self.particle[i][0].Position.z + self.particle[i][0].Velocity.z           
                
                # Velocity Mirroring
                # OutofTheRange = (self.particle[i][0].Position.z < self.VarMin.z or self.particle[i][0].Position.z > self.VarMax.z)
                OutofTheRange = (self.IsArr1Smaller(self.particle[i][0].Position.z, self.VarMin.z) or self.IsArr1Larger(self.particle[i][0].Position.z, self.VarMax.z))
                if OutofTheRange == True:
                    self.particle[i][0].Velocity.z = -self.particle[i][0].Velocity.z

                # Update Position Bounds
                self.particle[i][0].Position.z = np.maximum(self.particle[i][0].Position.z, self.VarMin.z)
                self.particle[i][0].Position.z = np.minimum(self.particle[i][0].Position.z, self.VarMax.z) 

                # covert position and velocity to list
                # self.particle[i][0].Position.x = self.particle[i][0].Position.x.tolist().copy()
                self.temp_particle.Position.x = self.particle[i][0].Position.x.tolist().copy()
                self.temp_particle.Position.y = self.particle[i][0].Position.y.tolist().copy()
                self.temp_particle.Position.z = self.particle[i][0].Position.z.tolist().copy()
                # self.particle[i][0].Position.y = self.particle[i][0].Position.y.tolist().copy()
                # self.particle[i][0].Position.z = self.particle[i][0].Position.z.tolist().copy()
                # self.particle[i][0].Velocity.x = self.particle[i][0].Velocity.x.tolist().copy()
                # self.particle[i][0].Velocity.y = self.particle[i][0].Velocity.y.tolist().copy()
                # self.particle[i][0].Velocity.z = self.particle[i][0].Velocity.z.tolist().copy()
                
                # Evaluation
                [self.particle[i][0].Cost, self.particle[i][0].PathLength, self.particle[i][0].Sol] = self.MyCost(self.temp_particle.Position, model_update)
                
                # Update Personal Best
                if (self.particle[i][0].Cost <self.particle[i][0].Best.Cost):
                    # print(self.particle[i][0].Cost)
                    self.particle[i][0].Best = copy.deepcopy(self.particle[i][0])
                    # self.particle[i][0].Best.Position.x = self.particle[i][0].Position.x.copy()
                    # self.particle[i][0].Best.Position.y = self.particle[i][0].Position.y.copy()
                    # self.particle[i][0].Best.Position.z = self.particle[i][0].Position.z.copy()
                    # self.particle[i][0].Best.Cost = self.particle[i][0].Cost.copy()
                    # self.particle[i][0].Best.Sol = self.particle[i][0].Sol.copy() #class sol2()
                    # self.particle[i][0].Best.PathLength = self.particle[i][0].PathLength.copy()
                    
                    # print('self.GlobalBest.Best.Cost at initilization ',self.GlobalBest.Best.Cost)

                    # Update Global Best
                    if self.particle[i][0].Best.Cost < self.GlobalBest.Best.Cost:
                        print('self.particle[i][0].Best.Cost ',self.particle[i][0].Best.Cost)
                        self.GlobalBest.Best = copy.deepcopy(self.particle[i][0].Best)
                        print('self.GlobalBest.Best.Cost after iteration',self.GlobalBest.Best.Cost)
            
            # Update Best Cost Ever Found
            BestCost[it] = self.GlobalBest.Best.Cost.copy()
            
            BestPathLength[it] = self.GlobalBest.Best.PathLength.copy()

            # Inertia Weight Damping
            w = w*wdamp

            # Show Iteration Information
            if self.GlobalBest.Sol.IsFeasible:
                Flag = '*'
            else:
                Flag = (",Violation = " + str(self.GlobalBest.Sol.Violation))
            print ("Iteration " + str(it) + ": Best Cost = " + str(BestCost[it]) + str(Flag))

            #figure(1)
            # self.PlotSolution(self.GlobalBest.Sol, model_update,it)

            # update target info and obstacles info in model
            # obstacles are moving at various velocities at various directions
            '''
            for obstacle_No in range(model_update.nobs):
                ran = random.random()
                if ran <= P_reloc_obs:
                    step_obs = random.uniform(-1.0, 1.0)
                    model_update.xobs[obstacle_No] = model_update.xobs[obstacle_No] + Maxstep_reloc_obs*step_obs
                ran = random.random()
                if ran <= P_reloc_obs:
                    step_obs = random.uniform(-1.0, 1.0)
                    model_update.yobs[obstacle_No] = model_update.yobs[obstacle_No] + Maxstep_reloc_obs*step_obs
                ran = random.random()
                if ran <= P_reloc_obs:
                    step_obs = random.uniform(-1.0, 1.0)
                    model_update.zobs[obstacle_No] = model_update.zobs[obstacle_No] + Maxstep_reloc_obs*step_obs 
            '''
        
        print('find best trajectory')
        return self.GlobalBest, model_update
    
    def IsArr1larger(self,arr1,arr2):
        count = 0
        # list = arr1.tolist()
        for i in range(len(arr1)):
            if(arr1[i] > arr2[i]):
                count += 1

        if(count == len(arr1)):
            # print("arr1 is greater")
            return True
        else:
            # print("arr1 is not greater")
            return False
            
    def IsArr1Larger(self,arr1,value):
        count = 0
        for i in range(len(arr1)):
            if(arr1[i]>value):
                count += 1
        if(count != 0): #arr1 has at least 1 element larger than value
            # print("arr1 is greater")
            return True
        else:
            # print("arr1 is not greater")
            return False

    def IsArr1Smaller(self,arr1,value):
        count = 0        
        for i in range(len(arr1)):
            if(arr1[i] < value):
                count += 1
        if(count != 0 ): # arr1 has at least one element smalled than value
            # print("arr1 is smaller")
            return True
        else:
            # print("arr1 is not smaller")
            return False

    def IsArr1smaller(self,arr1,arr2):
        count = 0
        for i in range(len(arr1)):
            if(arr1[i] < arr2[i]):
                count += 1

        if(count == len(arr1)):
            # print("arr1 is smaller")
            return True
        else:
            # print("arr1 is not smaller")
            return False

    def Create2DEnv(self, single_xobs,single_yobs,single_robs):
        circle=plt.Circle((single_xobs,single_yobs),single_robs,color='r')
        self.ax.add_patch(circle)


    def CreateCylinder(self,single_xobs,single_yobs,single_zobs,single_hobs,single_robs):
        
        # origin = np.array([0,0,0])
        # axis and radius
        p1 = np.array([single_xobs,single_yobs,single_zobs])
        # vector in direction of axis
        v = np.array([0,0,single_hobs])
        p0 = p1-v
        R = single_robs
        # find magnitude of vector
        mag = norm(v)
        # unit vector in direction of axis
        v = v/mag
        #make some vector not in the same direction as v
        not_v = np.array([1,0,0])
        if (v==not_v).all():
            not_v = np.array([0,1,0])
        # make unit vector perpendicular to v
        n1 = np.cross(v,not_v)
        # normalize n1
        n1 /= norm(n1)
        # make unit vector perpendicular to v and n1
        n2 = np.cross(v,n1)
        #surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0,mag,100)
        theta = np.linspace(0,2*np.pi,50) #divide the circle into 50 equal parts
        # use meshgrid to make 2d arrays
        t, theta = np.meshgrid(t,theta)
        # generate coordinates for surface
        X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        self.bx.plot_surface(X, Y, Z)
        #plot axis
        self.bx.plot(*zip(p0, p1), color = 'red')



    def PlotSolution(self, sol, model,iteraNo):
        # fig = plt.figure()
        # self.bx = plt.axes(projection='3d')
        fig, self.bx = plt.subplots(subplot_kw={"projection": "3d"})
        
        xs = model.xs
        ys = model.ys
        zs = model.zs
        xt = model.xt
        yt = model.yt
        zt = model.zt
        xobs = model.xobs
        yobs = model.yobs
        zobs = model.zobs
        hobs = model.hobs
        robs = [x - model.obstBuffer for x in model.robs]
        # robs = model.robs - model.obstBuffer
        nVar = model.n

        XS = sol.XS
        YS = sol.YS
        ZS = sol.ZS
        xx = sol.xx
        yy = sol.yy
        zz = sol.zz

        theta = np.linspace(0,2*math.pi,100)

        for k in range(len(xobs)):
            self.CreateCylinder(xobs[k],yobs[k],zobs[k],hobs[k],robs[k])
            #hold on   
        nUAVs = self.model.nUAVs
        
        #set(gca,'FontSize',12)
        # self.bx.set_xlabel('X[m]','FontSize',16,'FontWeight','bold')
        self.bx.set_xlabel('X')

        # self.bx.set_ylabel('Y[m]','FontSize',16,'FontWeight','bold')
        self.bx.set_ylabel('Y')
        
        # self.bx.set_zlabel('Z[m]','FontSize',16,'FontWeight','bold')
        self.bx.set_zlabel('Z')

        nc = matplotlib.colors.Normalize(vmin=0, vmax=nUAVs)
        for i in range(nUAVs):
            # self.bx.plot3D(xx[((i-1)*100):(100*i-1)], yy[((i-1)*100):(100*i-1)], zz[((i-1)*100):(100*i-1)],'m','LineWidth',2)
            self.bx.plot3D(xx[(i*100):(100*(i+1))], yy[(i*100):(100*(i+1))], zz[(i*100):(100*(i+1))],color = cm.hsv(nc(i)))
            # self.bx.plot3D(XS[(((nVar+2)*(i-1))):(((nVar+2)*(i-1))+nVar+2-1)], YS[(((nVar+2)*(i-1))):(((nVar+2)*(i-1))+nVar+2-1)], ZS[(((nVar+2)*(i-1))):(((nVar+2)*(i-1))+nVar+2-1)],'bo','LineWidth',2)
            self.bx.scatter(XS[((nVar+2)*i):((nVar+2)*(i+1))], YS[((nVar+2)*i):((nVar+2)*(i+1))], ZS[((nVar+2)*i):((nVar+2)*(i+1))],marker='x') 
        
        self.bx.scatter(xs,ys,zs,marker='*')
        # self.bx.plot3D(xs,ys,zs,'ks','MarkerSize',12,'MarkerFaceColor','g')
        self.bx.scatter(xt,yt,zt,marker='o')
        # self.bx.plot3D(xt,yt,zt,'kp','MarkerSize',16,'MarkerFaceColor','r')
        
        if (iteraNo == (self.MaxIt-1)):
            plt.show(block=True)
        else:
            # plt.show()
            # plt.close()
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()


    def ParseSolution(self, sol1, model):

        nUAVs = model.nUAVs
        nVar = model.n
        Violation = 0
        XS = []
        YS = []
        ZS = []
        L = 0
        temp_xx = []
        temp_yy = []
        temp_zz = []

        xobs = model.xobs.tolist().copy()
        yobs = model.yobs.tolist().copy()
        zobs = model.zobs.tolist().copy()
        robs = model.robs.tolist().copy()
        hobs = model.hobs.tolist().copy()
        
        for i in range(nUAVs):

            x = sol1.x[(nVar*i):(nVar*i+nVar)]
            
            # print(x)
            y = sol1.y[(nVar*i):(nVar*i+nVar)]
            z = sol1.z[(nVar*i):(nVar*i+nVar)] 

            xs = model.xs[i]
            ys = model.ys[i]
            zs = model.zs[i]

            xt = model.xt[i]
            yt = model.yt[i]
            zt = model.zt[i]

            kOld = len(XS)
            #print(kOld)

            x_temp = x.copy()
            x_temp.insert(0,xs)
            x_temp.append(xt)
            XS.extend(x_temp)
            #XS = [XS, xs, x, xt]

            y_temp = y.copy()
            y_temp.insert(0,ys)
            y_temp.append(yt)
            YS.extend(y_temp)
            #YS = [YS, ys, y, yt]

            z_temp = z.copy()
            z_temp.insert(0,zs)
            z_temp.append(zt)
            ZS.extend(z_temp)
            #ZS = [ZS, zs, z, zt] 
            
            k = len(XS)-kOld

            TS = np.linspace(0,1,k)

            tt = np.linspace(0,1,100)
            temp_xx2 = x.copy()
            temp_xx2.insert(0,xs)
            temp_xx2.append(xt)
            # temp_xx2 = [xs, x, xt]
            spl_xx = InterpolatedUnivariateSpline(TS, temp_xx2)
            xx = spl_xx(tt)

            temp_yy2 = y.copy()
            temp_yy2.insert(0,ys)
            temp_yy2.append(yt)
            # temp_yy2 = [ys, y, yt]
            spl_yy = InterpolatedUnivariateSpline(TS, temp_yy2)
            yy = spl_yy(tt)
            temp_zz2 = z.copy()
            temp_zz2.insert(0,zs)
            temp_zz2.append(zt)
            # temp_zz2 = [zs, z, zt]
            spl_zz = InterpolatedUnivariateSpline(TS, temp_zz2)
            zz = spl_zz(tt)
      
            dx = np.diff(xx)
            dy = np.diff(yy)
            dz = np.diff(zz)

            temp_xx.extend(xx.tolist())
            temp_yy.extend(yy.tolist())
            temp_zz.extend(zz.tolist())

            L = L+ sum(np.sqrt(np.square(dx)+np.square(dy)+np.square(dz)))

            nobs = len(xobs) # number of obstacles
            n = len(xx) # number of points to be seperated
            
            for k in range(nobs):

                xx_filtered = []
                yy_filtered = []
                zz_filtered = []

                for j in range(n):
                    if(zz[j] <= zobs[k]) and (zz[j] >= zobs[k]-hobs[k]):
                        xx_filtered.append(xx[j])
                        yy_filtered.append(yy[j])
                        zz_filtered.append(zz[j])
                d = ((np.array(xx_filtered)-np.array(xobs)[k])**2 + (np.array(yy_filtered)-yobs[k])**2)**0.5
                temp = []
                if (robs[k] != 0):
                    temp = 1-d/robs[k]
                # print(temp)
                zero_array = np.zeros_like(temp)
                v = np.maximum(temp,zero_array)
                # print(v)
                if (len(v)!=0):
                    Violation = Violation + np.mean(v)
                # print(Violation)
                if(math.isnan(Violation)):
                    print("STOP")
                
            xobs.extend(xx.tolist()[9:90])
            yobs.extend(yy.tolist()[9:90])
            zobs.extend((zz[9:90]+(self.model.obstBuffer+0.15)).tolist())
            robs.extend((self.model.obstBuffer+0.15)*np.ones(81))
            hobs.extend((self.model.obstBuffer+0.15)*2*np.ones(81))


        self.sol2.update_param(TS, XS, YS, ZS, tt, temp_xx, temp_yy, temp_zz, dx, dy, dz, L, Violation)
        sol = self.sol2

        return sol

        '''
        self.sol2.TS = TS
        self.sol2.XS = XS
        self.sol2.YS = YS
        self.sol2.ZS = ZS
        self.sol2.tt = tt
        self.sol2.xx = xx
        self.sol2.yy = yy
        self.sol2.zz = zz
        self.sol2.dx = dx
        self.sol2.dy = dy
        self.sol2.dz = dz
        self.sol2.L = L 
        self.sol2.Violation = Violation
        self.sol2.IsFeasible = (self.sol2.Violation==0)
        '''

    def CreateModel(self, nObs, obstBuffer):
        '''
        This function generates a random static environment with circular
        obstacles to test path planning algorithms
        ---------
        '''
        # set up the map
        # state bounds
        Boundary = 0.20
        xMax = [2, 2, 2]
        xMin = [-2, -2, 0]

        xMin = np.array(xMin) + Boundary
        xMax = np.array(xMax) - Boundary
        
        # set up obstacles
        minRad = 0.05 # minimum radius in meters
        maxRad = 0.10 # max radius in meters
        maxCount = 100000 # Iterations to search for obstacle locations
        minHeight = 0.1 # minimum height of obstacles in meters
        maxHeight = 2 # max height of obstacles in meters

        # Find obstacles that fit:
        obsPtsStore = self.circular_world(xMin, xMax, minRad, maxRad, nObs, obstBuffer, maxCount, minHeight, maxHeight)

        xobs = []
        yobs = []
        zobs = []
        robs = []
        hobs = []

        for i in range(nObs):
            xobs.append(obsPtsStore[i][0][0])
            yobs.append(obsPtsStore[i][0][1])
            zobs.append(obsPtsStore[i][0][2])
            robs.append(obsPtsStore[i][1][0])
            hobs.append(obsPtsStore[i][2][0])

        # Number of intermediate way points
        n = math.ceil(nObs/5)*3

        xMin = xMin - Boundary
        xMax = xMax + Boundary

        # model2 = model(...)

        xmin = xMin[0]
        xmax = xMax[0]
        ymin = xMin[1]
        ymax = xMax[1]
        zmin = xMin[2]
        zmax = xMax[2]
        
        self.model.update_param(xobs, yobs, zobs, robs, hobs, nObs, n, xmin, xmax, ymin, ymax, zmin, zmax, obstBuffer)
        env = self.model

        return env

            

    def circular_world(self,posMinBound, posMaxBound, minRad, maxRad, numObsts, obst_buffer, max_count, minHeight, maxHeight):
        ptsStore = []
        count = 0
        pos = np.zeros((numObsts,3)) 
        rad = np.zeros((numObsts,1))
        fake_rad = np.zeros((numObsts,1))
        height = np.zeros((numObsts,1))
        #NotNeed = 0.0

        for i in range (numObsts):
            # loop while there are collisions with obstacles
                       
            # generate random positions and lengths
            # center position
            temp = posMinBound + np.array([random.random()*(posMaxBound[0]-posMinBound[0]),
                                        random.random()*(posMaxBound[1]-posMinBound[1]),
                                        random.random()*(posMaxBound[2]-posMinBound[2])])
            
            pos[i] = temp #2Darray
                                                    
            rad[i]=random.random()*(maxRad-minRad)+minRad #list

            fake_rad[i]= rad[i] + obst_buffer #list

            height[i]=random.random()*(maxHeight-minHeight)+minHeight #list
            #print(height[i])
            
            #print(pos)
            #print(pos[i][0])
            #print(pos[i][1])
            #print(pos[i][2])
            #print([pos[i][0]-fake_rad[i],pos[i][1]-fake_rad[i],pos[i][2]-height[i]])
            # find the points 
             
            '''
            while(1):
                # generate random positions and lengths
                # center position
                temp = posMinBound + np.array([random.random()*(posMaxBound[0]-posMinBound[0]),
                                            random.random()*(posMaxBound[1]-posMinBound[1]),
                                            random.random()*(posMaxBound[2]-posMinBound[2])])
                #print(temp)
                pos[i] = temp #2Darray
                                                        
                rad[i]=random.random()*(maxRad-minRad)+minRad #list
                #print(rad[i])
                fake_rad[i]= rad[i] + obst_buffer #list
                #print(fake_rad[i])
                height[i]=random.random()*(maxHeight-minHeight)+minHeight #list
                #print(height[i])
                
                print(pos)
                print(pos[i])
                print(pos[i+1])
                print(pos[i+2])
                #print([pos[i][0]-fake_rad[i],pos[i][1]-fake_rad[i],pos[i][2]-height[i]])
                # find the points
                
                fake_pts = np.array([[pos[i][0]-fake_rad[i],pos[i][1]-fake_rad[i],pos[i][2]-height[i]], 
                                    [pos[i][0]+fake_rad[i], pos[i][1]-fake_rad[i],pos[i][2]-height[i]],
                                    [pos[i][0]+fake_rad[i], pos[i][1]+fake_rad[i],pos[i][2]-height[i]],
                                    [pos[i][0]-fake_rad[i], pos[i][1]+fake_rad[i],pos[i][2]-height[i]],
                                    [pos[i][0]-fake_rad[i], pos[i][1]-fake_rad[i],pos[i][2]],
                                    [pos[i][0]+fake_rad[i], pos[i][1]-fake_rad[i],pos[i][2]],
                                    [pos[i][0]+fake_rad[i], pos[i][1]+fake_rad[i],pos[i][2]],
                                    [pos[i][0]-fake_rad[i], pos[i][1]+fake_rad[i],pos[i][2]]])
                print(fake_pts[:,0])
                # check to see if it is outside the region
                if(min(fake_pts[:,0]) <= posMinBound[0] or max(fake_pts[:,0]) >= posMaxBound[0] or min(fake_pts[:,1]) <= posMinBound[1] or max(fake_pts[:,1]) >= posMaxBound[1] or min(fake_pts[:,2]) <= posMinBound[2] or max(fake_pts[:,2]) >= posMaxBound[2]):
                  #arr = np.delete(arr1, obj, axis)  # axis = 0 row---, axis=1 column |.
                    continue

                # check to see if it collided with any of the other obstacles
                collided = 0
                for j in range(i-1):
                    if (self.cylinderOverlap(pos[i:i+2], fake_rad[i], height[i], pos[j:j+2], fake_rad[j], height[j])):
                        collided = 1
                        break
                    
                if  collided==1:
                    break
                
                count = count + 1
                if (count >= max_count):
                    ptsStore = []
                    return
            '''
                
            ptsStore.append([pos[i], fake_rad[i], height[i]])

        return ptsStore


    def cylinderOverlap(self, center1, radius1, height1, center2, radius2, height2):
        val = 0
        dist_hor = math.sqrt((center1[0]-center2[0])^2+(center1[1]-center2[1])^2)
        dist_ver = math.sqrt((center1[2]-center2[2])^2) # check for vertical collisions

        if (dist_hor <= (radius1+radius2)):
            val = 1
        
        return val

    def CreateRandomSolution(self, model, position):
        # print(model.xs-model.xt)
        dist = np.zeros(model.nUAVs)
        sigma = np.zeros(model.nUAVs)
        x,y,z = [],[],[]
        for k in range(model.nUAVs):
            dist[k] = math.sqrt((model.xs[k]-model.xt[k])**2+(model.ys[k]-model.yt[k])**2+(model.zs[k]-model.zt[k])**2)
            sigma[k] = (dist[k]/(model.n+1))/2
            temp_x = np.random.normal(position.x[(k*model.n):((k+1)*model.n)], sigma[k])
            x.extend(temp_x.tolist())
            # print('position.x',position.x)
            # print('x',x)
            temp_y = np.random.normal(position.y[(k*model.n):((k+1)*model.n)], sigma[k])
            y.extend(temp_y.tolist())
            # print('position.y',position.y)
            # print('y',y)
            temp_z = np.random.normal(position.z[(k*model.n):((k+1)*model.n)], sigma[k])
            z.extend(temp_z.tolist())
            # print('position.z',position.z)
            # print('z',z)

        self.sol1.x = x.copy()
        self.sol1.y = y.copy()
        self.sol1.z = z.copy()
        sol = self.sol1
        return sol


    def MyCost(self, sol1, model):
        # print(sol1)
        sol = self.ParseSolution(sol1,model) #class sol2()
        beta = 10
        z = sol.L*(1+beta*sol.Violation)
        zl = sol.L

        return [z, zl, sol]

    def getGoals(self, env):
        #fig= plt.figure()
        #self.ax = plt.axes()
        #self.ax = plt.axes(projection='3d')
        fig, self.ax = plt.subplots() # 2D plot

        model = env
        xobs = model.xobs
        yobs = model.yobs
        zobs = model.zobs
        robs = np.array(model.robs) - model.obstBuffer
        hobs = model.hobs

        for k in  range(len(xobs)):
            #self.CreateCylinder(xobs[k],yobs[k],zobs[k],hobs[k],robs[k])
            self.Create2DEnv(xobs[k],yobs[k],robs[k])
            #hold on
        
        #set(gca,'FontSize',12)
        plt.xlabel('X[m]',fontsize=16)
        plt.ylabel('Y[m]',fontsize=16)
        # self.ax.set_zlabel('Z[m]',fontsize=16)
        

        plt.xlim(model.xmin, model.xmax)
        plt.ylim(model.ymin, model.ymax)
        # self.ax.set_zlim(model.zmin, model.zmax)

        # plt.show()
        AAAA=plt.ginput(1)
        # model.xs = input("Please enter the UAV starting position x:\n")
        # model.ys = input("Please enter the UAV starting position y:\n")
 
        model.xs,model.ys = AAAA[0]#plt.ginput(1)
        xs = model.xs
        ys = model.ys
        zs = 0
        model.zs = zs
        #self.ax.plot3D(xs,ys,zs)]
        plt.scatter(xs,ys,marker='*')
        #self.ax.plot3D(xs,ys,zs,'bs','MarkerSize',16,'MarkerFaceColor','y')
        # model.xt = input("Please enter the goal position x:\n")
        # model.yt = input("Please enter the goal position y:\n")
        # model.zt = input("Please enter the goal position z:\n")
        temp_t = plt.ginput(1)
        model.xt,model.yt = temp_t[0]
        xt = model.xt
        yt = model.yt
        # zt = random.random()*(model.zmax-model.zmin)+model.zmin
        zt = 1.6
        model.zt = zt
        #zt = model.zt
        #self.ax.plot3D(xt,yt,zt)
        plt.scatter(xt,yt,marker='o')
        #self.ax.plot3D(xt,yt,zt,'kp','MarkerSize',16,'MarkerFaceColor','g')

        plt.show()
        
        #hold off 
        #plt.hold(False)
        #plt.grid(True)
        #plt.show()

        return model

if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    Path_Generation = path_generation()
    
    # xobs = np.array([])
    # yobs = np.array([])
    # zobs = np.array([])
    # robs = np.array([])
    # hobs = np.array([])
    xobs = np.array([-1.8, -1, -0.5,    0,      0.5,    1.7,  0])*5
    yobs = np.array([-1.8,  0,  1.5,    -0.5,   1.5,    -0.5, -2])*5
    zobs = np.array([2,   1.8,    1,    0.5,    1.5,    1.3,  2])*5
    robs = np.array([0.3, 0.1,  0.5,    0.2,    0.4,    0.5,  0.5])*5
    # hobs = np.array([1,   0.2,  0.6,    0.8,    0.9,       2,  2])
    hobs = np.array([2,   2,  2,   2,    2,    2,  2])*5
    nObs = len(xobs)
    xmin = -2*5
    xmax = 2*5
    ymin = -2*5
    ymax = 2*5
    zmin = 0*5
    zmax = 2*5
    # xs = np.array([1.5, 1.5, 1.5, 1.5])
    # ys = np.array([1.5, 0.5, -0.5, -1.5])
    # zs = np.array([0, 0, 0, 0])
    xs = np.array([-1.5, -2, -1.5])*5
    # xs = np.array([-1.5, -2, -1.5])
    # ys = np.array([-1.7, -1.5, -1.5])
    ys = np.array([-1.7, -1.5, -1.5])*5
    # zs = np.array([0, 0, 0])
    zs = np.array([0, 0, 0])*5
    # target_init = np.array([1.5, 0.0, 1.6])
    # xt = np.array([1.3, 1.5, 1.5, 1.0]) 
    # yt = np.array([0.0, 1.0, 0.0, 0.0]) 
    # zt = np.array([1.6, 1.6, 1.8, 2.0])  
    xt = np.array([1.9, 1.7, 1.3])*5
    # xt = np.array([1.9, 1.7, 1.6])
    yt = np.array([1.5, 1.8, 1.1])*5
    # yt = np.array([1.5, 1.8, 1.4])
    zt = np.array([1.8, 1.5, 1.4])*5
    # zt = np.array([1.8, 1.5, 1.2])

    Path_Generation.pso(xobs, yobs, zobs, robs, hobs, nObs, xmin, xmax, ymin, ymax, zmin, zmax, xs, ys, zs, xt, yt, zt)








