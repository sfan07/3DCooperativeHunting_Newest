import numpy as np
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation
from numpy.linalg import norm
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
from scipy.interpolate import UnivariateSpline
from collision_avoidance import CollisionAvoidance
import time

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
        
    def control(self, ref, state):
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

class Sim2D():   
    def __init__(self, agent_init, target_waypts, obs_init, num_iter, dt, order = 2):
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
        
        # the waypoint target is currently on
        self.point_idx = 0

        self.agent_init = np.copy(agent_init)
        self.agent_state = np.copy(agent_init)
        self.target_init = np.copy(target_waypts[0].T)
        # (6,1)
        self.target_state = np.zeros((6, 1))
        self.target_state[:4] = target_waypts[0].reshape((4,1))
        # self.target_state = np.copy(target_waypts[0].T)
        self.target_waypts = np.copy(target_waypts)
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

        c1_alp = 3
        c2_alp = 1
        d = 0.4
        d_p = 0.3
        r_bet = 0.5
        h_bet = 0.5
        eps = 0.1
        r = 0.3 #0.3
        A_UAV = np.pi*((1/2)*0.092)**2
        h_alpha = 0.5
        dw_h = 0.5
        #self.avoidance_controller = CollisionAvoidance(c1_alp, c2_alp, d, d_p, h_bet, eps, r)
        self.avoidance_controller = CollisionAvoidance(c1_alp, c2_alp, d, d_p, r_bet, h_bet, eps, r, A_UAV, h_alpha, dw_h)
        
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

        # if target_input is not None:
        #     self.target_state[:,2:] = np.copy(target_input)
        # self.target_state[0:2] += self.target_state[2:]*self.dt

        # UAV dynamics for target as well
        self.target_state = self.A_d @ self.target_state
        if target_input is not None:
            self.target_state += self.B_d @ target_input

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

    def get_vertices(self, l = 0.10):
        xt, yt, zt = self.target_pos
        vertex_pos = np.array([[xt + l, yt + l, zt + l],
                             [xt + l, yt - l, zt - l],
                             [xt - l, yt + l, zt - l],
                             [xt - l, yt - l, zt + l]])
        return vertex_pos

    def hungarian_assignment(self, vertex_vec, agent_vec):
        cost_mtx = -1*(vertex_vec@agent_vec.T)
        assignment = linear_sum_assignment(cost_mtx)[1]
        return assignment

    def get_cross_mtx(self, vec):
        x, y, z = vec
        cross_mtx = np.array([[0, -z, y],
                              [z, 0, -x],
                              [-y, x, 0]])
        return cross_mtx

    def rotation_from_axis_angle(self, axis, angle):
        n_cross = self.get_cross_mtx(axis)
        C = np.eye(3)+np.sin(angle)*n_cross+(1-np.cos(angle))*n_cross@n_cross
        return C
                
    def run(self, h = 1.25, l = 0.30, look_ahead_num = 10, look_ahead_dt = 0.2):
        '''
        h: height of the target
        l: sizing coefficient of the vertex_pos
        look_ahead_num: points for spline interpolation
        look_ahead_dt: dt for spline interpolation
        '''
        # height of the agents, starting from 0, each row is [z, vz]
        self.agent_h = np.zeros((self.num_agents, 2))

        # spline interpolation parameters 
        self.look_ahead_num = look_ahead_num 
        self.look_ahead_dt = look_ahead_dt

        #################################################################
        # for plotting/verification
        self.agent_coords = np.zeros((self.num_agents, self.num_iter, 3))
        self.target_coords = np.zeros((self.num_iter, 3))
        #################################################################

        self.reached = np.array([False]*self.num_agents)
        i_old = 0
        for i in range (self.num_iter):
            if self.order == 2:
                ref_state = np.zeros((8, self.num_agents))
                error_state = np.zeros((8, self.num_agents))
            elif self.order == 1:
                ref_state = np.zeros((6, self.num_agents))
                error_state = np.zeros((6, self.num_agents))
            agent_input = np.zeros((2, self.num_agents))

            # augment agent and target state with height
            self.agent_pos = np.hstack((self.agent_state.T[:,:2], self.agent_h[:,0].reshape((self.num_agents, 1))))
            self.target_pos = np.hstack((self.target_state.T[0,:2], h))
            # get vertex position and vectors
            self.vertex_pos = self.get_vertices(l)
            self.vertex_vec = normalize(self.vertex_pos-self.target_pos, axis = 1, norm = 'l2')

            #use the next look_ahead_num points for spline interpolation
            ts = np.zeros((self.look_ahead_num, self.num_agents))
            xs = np.zeros((self.look_ahead_num, self.num_agents))
            ys = np.zeros((self.look_ahead_num, self.num_agents))
            look_ahead_pts = np.zeros((self.look_ahead_num+1, self.num_agents, 3))
            look_ahead_pts[0] = self.agent_pos[:,:3]

            ########################################
            # for plotting/verification
            self.agent_coords[:, i, :] = self.agent_pos
            self.target_coords[i, :] = self.target_pos
            ########################################

            collision_input = np.zeros((self.num_agents, 6))
            collision_input[:,:2] = self.agent_state.T[:,:2]
            collision_input[:,2] = self.agent_h[:,0]
            collision_input[:,3:5] = self.agent_state.T[:,2:4]
            collision_input[:,5] = self.agent_h[:,1] 
            # collision_acc = self.avoidance_controller.get_control_all(collision_input)
            collision_acc, downwash_flag, neighbors_pos = self.avoidance_controller.get_control_all(collision_input) # collision_input: [x,y,z,vx,vy,vz]
            agent_coord = collision_input[:,:3]

            for n in range (self.look_ahead_num):    
                self.agent_vec = normalize(look_ahead_pts[n]-self.target_pos, axis = 1, norm = 'l2')
                # Hungarian Algorithm for vertex assignment
                assignment = self.hungarian_assignment(self.vertex_vec, self.agent_vec)
                self.assigned_vertex_pos = self.vertex_pos[assignment]
                self.assigned_vertex_vec = self.vertex_vec[assignment]

                axis = normalize(np.cross(self.agent_vec, self.assigned_vertex_vec), axis = 1, norm = 'l2')           
                d_agent_target = norm(look_ahead_pts[n]-self.target_pos, axis = 1)
                d_vertex_target = norm(self.assigned_vertex_pos-self.target_pos, axis = 1)
                angular_diff = np.arccos(np.diag(self.agent_vec@self.assigned_vertex_vec.T))
                radial_diff = d_agent_target-d_vertex_target

                for j in range (self.num_agents):
                    # get the next waypoint
                    perc = 0.07
                    C = self.rotation_from_axis_angle(axis[j], perc*angular_diff[j])
                    v = C@self.agent_vec[j]/norm(C@self.agent_vec[j])
                    waypoint = self.target_pos+(d_vertex_target[j]+(1-perc)*radial_diff[j])*v
                    # cap the height to be between 0.1m and 2m
                    waypoint[-1] = np.clip(waypoint[-1], 0.1, 2.0)
                    # populates look ahead points to be interpolated
                    ts[n, j] = n*self.look_ahead_dt
                    xs[n, j] = waypoint[0]
                    ys[n, j] = waypoint[1]
                    look_ahead_pts[n+1, j] = waypoint

            for j in range (self.num_agents):
                # interpolate for velocity and (accelerations --> roll, pitch)
                x_interp = UnivariateSpline(ts[:, j], xs[:, j])
                y_interp = UnivariateSpline(ts[:, j], ys[:, j])
                vx_interp = x_interp.derivative()
                vy_interp = y_interp.derivative()
                ax_interp = vx_interp.derivative()
                ay_interp = vy_interp.derivative()

                waypt = look_ahead_pts[self.look_ahead_num//2, j]
                vx, vy = vx_interp(self.look_ahead_num//2*self.look_ahead_dt), vy_interp(self.look_ahead_num//2*self.look_ahead_dt)
                ax, ay = ax_interp(self.look_ahead_num//2*self.look_ahead_dt), ay_interp(self.look_ahead_num//2*self.look_ahead_dt)
                '''
                if ax > 1 or ax < -1:
                    print("acc_x are ",ax)
                if ay > 1 or ay < -1:
                    print("acc_y are ",ay)
                '''
                # update agent reference state
                ref_state[0, j] = waypt[0]
                ref_state[1, j] = waypt[1]
                ref_state[2, j] = vx
                ref_state[3, j] = vy
                ref_state[4, j] = -ax*180/(9.81*3.14)
                ref_state[5, j] = ay*180/(9.81*3.14)
                agent_input[0, j] = -collision_acc[j, 0]*180/(9.81*3.14)
                agent_input[1, j] = collision_acc[j, 1]*180/(9.81*3.14)

                # update agent height using PID + double integrator
                acc = self.pids[j].control(waypt[-1], self.agent_h[j, 0])
                if self.agent_h[j, 0] < 0.2:
                   collision_acc[j, 2] = np.abs(collision_acc[j, 2])
                
                acc += collision_acc[j, 2]
                acc_xyz = [ax, ay, acc]
                '''
                if acc > 1 or acc < -1:
                    print("acc_z are ",acc)
                '''
                downwash_acc = self.get_dw_acc(downwash_flag[j], acc_xyz, neighbors_pos[j], agent_coord[j], agent_input[:2,j])
                
                self.agent_h[j, 0] += self.agent_h[j, 1]*self.dt+(1/2)*acc*self.dt*self.dt-downwash_acc*(1/2)*self.dt*self.dt 
                # self.agent_h[j, 0] += downwash_acc* 1/2*self.dt*self.dt 
                # self.agent_h[j, 1] += acc*self.dt-downwash_acc*self.dt
                self.agent_h[j, 1] += acc-downwash_acc

                if downwash_acc != 0.0:
                    print("downwash_accel is",downwash_acc)
                    print("agent no. is ",j)
                    print("next agent")

            # get agent input
            error_state = self.agent_state-ref_state
            agent_input += self.K@error_state

            # get target input
            if self.point_idx == 0:
                target_input = None
            else:
                ref_state = np.zeros((6, 1))
                error_state = np.zeros((6, 1))
                # update target reference state
                ref_state[0] = self.target_waypts[self.point_idx, 0]
                ref_state[1] = self.target_waypts[self.point_idx, 1]
                error_state = self.target_state - ref_state
                target_input = self.K@error_state

            # step
            self.agent_state, self.target_state, self.obs_state = self.step(agent_input, target_input, None)
            
            # check for reach
            for ii in range(self.num_agents):
                pos = np.hstack((self.agent_state.T[ii,:2], self.agent_h[ii,0]))
                vel = np.hstack((self.agent_state.T[ii,2:], self.agent_h[ii,1]))
                if norm(pos - self.assigned_vertex_pos[ii]) < 0.05 and norm(vel) < 0.15:
                    self.reached[ii] = True
                else:
                    self.reached[ii] = False
            print('reaching waypt', str(self.point_idx))
            print(self.reached)

            if np.all(self.reached == True) and i - i_old > 200:
                if self.point_idx == len(target_waypts)-1:
                    break
                # target moves to the next point
                self.point_idx = (self.point_idx + 1)%self.target_waypts.shape[0]
                self.reached = np.array([False]*self.num_agents)
                i_old = i
        #self.vis()

    def get_dw_acc(self, dw_flag, acc_xyz, neighbors_pos, agent_coord, agent_input):
        multi = 10
        downwash_acc1 = 0.0
        DEG2RAD = np.pi/180
        pitch = agent_input[1]*DEG2RAD
        roll = agent_input[0]*DEG2RAD
        acc1 = acc_xyz
        # neighbors_pos = neighbors_pos
        # agent_coord = agent_coord
        A_UAV = np.pi*((1/2)*0.092)**2
        for k in range(len(dw_flag)):
            a_des = (-np.sin(pitch)*acc1[0] + np.cos(pitch)*np.sin(roll)*acc1[1] + np.cos(pitch)*np.cos(roll)*(acc1[2]+9.81))
            z = neighbors_pos[k][2] - agent_coord[2]
            if dw_flag[k] == 1:
                print("delta z is ",z)
                print("a_des is", a_des)
                downwash_acc = 25 * A_UAV / 2 / np.pi * a_des / (z**2)
                downwash_acc1 += downwash_acc
                print("downwash_acc is ", downwash_acc)
        temp = multi*downwash_acc1
        return temp
        
    def vis(self):
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        # ax.set_title('Pure Pursuit w Online Waypoint Generation')
        ax.set_xlim3d(-6.5, 1.5, auto = False)
        ax.set_ylim3d(-6, 1.5, auto = False)  
        ax.set_zlim3d(0, 3, auto = False)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')

        ax.scatter(self.target_coords[:, 0], self.target_coords[:, 1], self.target_coords[:, 2], color = 'red', label = 'target', marker = '*')
        ax.scatter(self.vertex_pos[:, 0], self.vertex_pos[:, 1], self.vertex_pos[:, 2], color = 'brown', label = 'vertex_pos', marker = 'X')
        
        for i in range (self.num_agents):
            ax.scatter(self.agent_coords[i, :, 0], self.agent_coords[i, :, 1], self.agent_coords[i, :, 2], s = 3, label = 'agent'+str(i))
        
        
        '''
        # 4 agents separetly
        i = 0
        ax.scatter(self.agent_coords[i, :, 0], self.agent_coords[i, :, 1], self.agent_coords[i, :, 2], s = 3, label = 'agent'+str(i))
        
        i = 1
        ax.scatter(self.agent_coords[i, :, 0], self.agent_coords[i, :, 1], self.agent_coords[i, :, 2], s = 3, label = 'agent'+str(i))
        
        i = 2
        ax.scatter(self.agent_coords[i, :, 0], self.agent_coords[i, :, 1], self.agent_coords[i, :, 2], s = 3, label = 'agent'+str(i))
        
        i = 3
        ax.scatter(self.agent_coords[i, :, 0], self.agent_coords[i, :, 1], self.agent_coords[i, :, 2], s = 3, label = 'agent'+str(i))
        '''
        ax.legend()
        plt.show()
        fig.savefig("sim_3D_moving_target_agentLoc1",dpi = 300)
    '''
    def plots2D(self):
        fig, axs = plt.subplots(3)
        axs[0,0].plot(self.num_iter, agent_coords[1,:,0])
        axs[0,0].plot(self.num_iter,target_coords[1,0])
        fig, axs = plt.subplots(3)
        axs[1,0].plot(self.num_iter, agent_coords[1,:,1])
        axs[1,0].plot(self.num_iter,target_coords[1,1])
        
    '''
if __name__ == '__main__':
    order = 1

    if order == 2:
        agent_init = np.zeros((8, 4)) 
    elif order == 1:
        agent_init = np.zeros((6, 4)) 
    # agentLoc 1
    # agent_init[:2, 0] = -4.32, 1.766
    # agent_init[:2, 1] = -6.21, -3.6
    # agent_init[:2, 2] = -4.63, -5.51
    # agent_init[:2, 3] = 1.647, -4.30
    
    # TEST
    # agent_init[:2, 0] = -4.32, 1.766
    # agent_init[:2, 1] = -6.21, -3.6
    # agent_init[:2, 2] = -4.63, -5.51
    # agent_init[:2, 3] = 1.647, -4.30

    # agentLoc 2
    agent_init[:2, 0] = -1.5, 0.3
    agent_init[:2, 1] = -1.5, 0.0
    agent_init[:2, 2] = -1.5, -0.3
    agent_init[:2, 3] = -1.5, -0.6

    # agentLoc 3
    # agent_init[:2, 0] = -0.5, 0
    # agent_init[:2, 1] = -0.7, 0
    # agent_init[:2, 2] = -0.9, 0
    # agent_init[:2, 3] = -1.1, 0

    # straight line
    # target_waypts = np.array([[ 1.5,  0, 0, 0],
    #                           [-1.5,  1, 0, 0],
    #                           [ 1.5,  0, 0, 0]])

    # triangular
    target_waypts = np.array([[ 1.5,  0, 0, 0],
                              [-1.5,  1, 0, 0],
                              [-1.5, -1, 0, 0],
                              [ 1.5,  0, 0, 0]])

    # parellelogram
    # target_waypts = np.array([[ 1.5,  0, 0, 0],
    #                           [ 1.5,  2, 0, 0],
    #                           [-1.5,  1, 0, 0],
    #                           [-1.5, -1, 0, 0],
    #                           [ 1.5,  0, 0, 0]])

    # target_waypts = np.array([[ 1.5,  0, 0, 0],
    #                           [ 1.5,  2, 0, 0],
    #                           [ 1.5,  0, 0, 0]])

    obs_init = None

    sim_2D = Sim2D(agent_init, target_waypts, obs_init, num_iter = 5000, dt = 0.01, order = order)
    start = time.time()
    sim_2D.run()
    end = time.time()
    print(f"Runtime of the program is {end - start}")
