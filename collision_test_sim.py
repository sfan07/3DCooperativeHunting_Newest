import numpy as np
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation
from numpy.linalg import norm
from matplotlib.lines import Line2D
from collision_avoidance import CollisionAvoidance

class PID():
    def __init__(self, kp, ki, kd, u_min, u_max, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt
        self.e_prev = None
        self.e_accum = 0
        
    def control(self,ref,state):
        e = ref - state
        self.e_accum += e
        #self.e_accum = np.clip(self.e_accum, self.i_min, self.i_max)
        if self.e_prev is None:
            self.e_prev = e

        u = self.kp*e + self.ki*self.e_accum + self.kd*(e - self.e_prev)/self.dt
        u = np.clip(u, self.u_min, self.u_max)

        self.e_prev = e
        return u
        
    def reset(self):
        self.e_prev = None
        self.e_accum = 0

class Collision_Sim2D():   
    def __init__(self, agent_init, target_init, obs_init, num_iter, dt, order = 1):
        '''
        Parameters
        ----------
        agent_init: (4+)xN array, each column is [x,y,vx,vy,etc...].T
        target_init: 4x1 array [x,y,vx,vy].T
        obs_init: 5xN array, each column is [x,y,vx,vy,r].T, or None
        dt: timestep for discrete model
        order: approximation order for crazyflie dynamics, default is 2
        '''
        # full state is [x,y,vx,vy,roll,pitch,roll_dot,pitch_dot]
        # assume second order approximation
        A = np.zeros([8,8]) 
        A[0,2], A[1,3] = 1, 1
        A[2,5], A[3,4] = 9.81*np.pi/180, -9.81*np.pi/180 
        A[4,6], A[5,7] = 1, 1
        A[6,4], A[7,5] = -37.2808, -37.2808
        A[6,6], A[7,7] = -8.1488, -8.1488

        B = np.vstack((np.zeros([6,2]), np.diag([37.2808, 37.2808])))
        
        # handle zero and first order cases
        if order == 0:
            A = A[0:4,0:4]
            B = np.vstack((np.zeros([2,2]), np.eye(2)))
    
        if order == 1:
            A = A[0:6,0:6]
            A[4,4], A[5,5] = -5.5759, -5.5759
            B = np.vstack((np.zeros([4,2]), np.diag([5.5759, 5.5759])))

        # discretized dynamics
        ncols = A.shape[1] + B.shape[1]
        H = np.zeros([ncols,ncols])
        H[0:A.shape[0],0:A.shape[0]] = A*dt
        H[0:B.shape[0],A.shape[0]:] = B*dt
        H = expm(H)
        
        self.order = order
        self.num_agents = agent_init.shape[1]
        self.num_iter = num_iter
        self.dt = dt
        self.A = A
        self.B = B
        
        self.A_d = H[0:A.shape[0],0:A.shape[0]]
        self.B_d = H[0:B.shape[0],A.shape[0]:]
        if self.order == 2:
            self.K = np.array([[       0,      10,        0, 21.6188, -2.1450,       0, -0.8598,       0],
                               [     -10,       0, -21.6188,       0,       0, -2.1450,       0, -0.8598]])
        elif self.order == 1:
            self.K = np.array([[       0,      10,        0, 17.4568, -0.7527,       0],
                               [     -10,       0, -17.4568,       0,       0, -0.7527]])
    
        self.agent_init = np.copy(agent_init)
        self.agent_state = np.copy(agent_init)
        self.target_init = np.copy(target_init)
        self.target_state = np.copy(target_init)
        self.obs_init = None if obs_init is None else np.copy(obs_init)
        self.obs_state = None if obs_init is None else np.copy(obs_init)

        # kp, ki, kd, u_min, u_max, dt
        kp = 0.1
        ki = 0
        kd = 0.1
        u_min = -0.01
        u_max =  0.01
        self.pids = [PID(kp, ki, kd, u_min, u_max, self.dt),\
                     PID(kp, ki, kd, u_min, u_max, self.dt),\
                     PID(kp, ki, kd, u_min, u_max, self.dt),\
                     PID(kp, ki, kd, u_min, u_max, self.dt)]

        # for collision test
        self.waypoints = np.array([[ 0.4,  0.5, 0.5],
                                   [-0.5, -0.5, 0.5]])
        # self.waypoints = np.array([[ 0.4,  0.5, 0.45],
        #                            [-0.5, -0.5, 0.5]])
        self.reached = np.array([False, False])
        c1_alp = 3.5
        c2_alp = 1
        d = 0.4
        d_p = 0.3
        h_bet = 0.5
        eps = 0.1
        r = 0.2
        self.avoidance_controller = CollisionAvoidance(c1_alp, c2_alp, d, d_p, h_bet, eps, r)

    def step(self, agent_input, target_input, obs_input, return_copy = True):
        '''
        Parameters
        ----------
        agent_input : 2xN inputs, each column is desired [roll, pitch].T
                      or accelerations [ax, ay].T if using 0 order assumption
        target_input : 2x1 velocity, [vx, vy].T
        obs_input : 2xN velocities, each column is [vx, vy].T
        
        If any inputs are None, update occurs assuming 0 input
​
        Returns
        -------
        References or copies of [agent_state, target_state, obs_state]
        '''
        self.agent_state = self.A_d @ self.agent_state
        if agent_input is not None:
            self.agent_state += self.B_d @ agent_input
            
        if target_input is not None:
            self.target_state[:,2:] = np.copy(target_input)
            self.target_state[0:2] += self.target_state[2:]*self.dt
        
        if self.obs_state is not None:
            if obs_input is not None:
                self.obs_state[:,2:] = np.copy(obs_input)
            self.obs_state[:,0:2] += self.obs_state[:,2:]*self.dt
            
        if return_copy:
            return [np.copy(self.agent_state), 
                    np.copy(self.target_state),
                    None if self.obs_state is None else np.copy(self.obs_state)]
        else:
            return [self.agent_state, self.target_state, self.obs_state]

    def run(self):
        self.agent_pos = np.hstack((self.agent_state.T[:,:2], np.zeros((self.num_agents,1))))
        self.agent_h = np.zeros((self.num_agents, 2))
        #################################################################
        # for plotting/verification
        self.agent_coords = np.zeros((self.num_agents, self.num_iter, 3))
        #################################################################
        swapped = False
        for i in range(self.num_iter):
            ref_state = np.zeros((6, self.num_agents))
            error_state = np.zeros((6, self.num_agents))
            agent_input = np.zeros((2, self.num_agents))

            self.agent_pos = np.hstack((self.agent_state.T[:,:2], self.agent_h[:, 0].reshape((self.num_agents, 1))))
            
            ########################################
            # for plotting/verification
            self.agent_coords[:, i, :] = self.agent_pos
            ########################################


            # collision avoidance
            # input is [x, y, z, vx, vy, vz]
            collision_input = np.zeros((self.num_agents, 6))
            collision_input[:,:2] = self.agent_state.T[:,:2]
            collision_input[:,2] = self.agent_h[:,0]
            collision_input[:,3:5] = self.agent_state.T[:,2:4]
            collision_input[:,5] = self.agent_h[:,1] 
            collision_acc = self.avoidance_controller.get_control_all(collision_input)

            for j in range (self.num_agents):
                # check for reach
                pos = np.hstack((self.agent_state.T[j,:2], self.agent_h[j,0]))
                vel = np.hstack((self.agent_state.T[j,2:], self.agent_h[j,1]))
                if norm(pos - self.waypoints[j]) < 0.05 and norm(vel) < 0.15:
                    self.reached[j] = True
                else:
                    self.reached[j] = False

                # update reference state
                ref_state[:2, j] = self.waypoints[j,:2]
                agent_input[0, j] = -collision_acc[j, 0]*180/(9.81*3.14)
                agent_input[1, j] = collision_acc[j, 1]*180/(9.81*3.14)
                acc = self.pids[j].control(self.waypoints[j,2], self.agent_h[j, 0])
                acc += collision_acc[j, 2]

                # control z
                self.agent_h[j, 0] += self.agent_h[j, 1]*self.dt+(1/2)*acc*self.dt*self.dt
                self.agent_h[j, 1] += acc

            error_state = self.agent_state-ref_state
            agent_input += self.K@error_state
            self.agent_state, self.target_state, self.obs_state = self.step(agent_input, None, None)
            
            if np.all(self.reached) and not swapped:
                self.waypoints[:2,:2] *= -1
                swapped = True
        self.vis()

    def vis(self):
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.set_xlim3d(-3, 3, auto = False)
        ax.set_ylim3d(-2, 2, auto = False)  
        ax.set_zlim3d(0, 2, auto = False)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')

        for i in range (self.num_agents):
            ax.scatter(self.agent_coords[i, :, 0], self.agent_coords[i, :, 1], self.agent_coords[i, :, 2], s = 3, label = 'agent'+str(i))
            ax.scatter(self.waypoints[i, 0], self.waypoints[i, 1], self.waypoints[i, 2], s = 100, marker = 'x', color = 'red')
        ax.legend()
        plt.show()



if __name__ == '__main__':
    agent_init = np.zeros((6, 2)) 

    agent_init[:2, 0] =  0.5, 0.5
    agent_init[:2, 1] = -0.5, -0.5

    target_init = None
    obs_init = None

    collision_sim2D = Collision_Sim2D(agent_init, target_init, obs_init, num_iter = 2000, dt = 0.01, order = 1)
    collision_sim2D.run()


