from scipy.optimize import linear_sum_assignment
from numpy import linalg as LA
import numpy as np
import time
import math

class CollisionAvoidance():
    '''
    input: 
    r_bet and h_bet for bump function
    h_alpha is the height when downwash effect close to zero       

    '''
    def __init__(self, c1_alp, c2_alp, d, d_p, r_bet, h_bet, eps, r, A_UAV, h_alpha, dw_h):
        self.c1_alp = c1_alp
        self.c2_alp = c2_alp
        self.d = d
        self.d_p = d_p 
        self.r_bet=r_bet
        self.h_bet = h_bet
        self.eps = eps
        self.r = r
        # self.dt = dt
        self.A_UAV = A_UAV
        self.h_alpha = h_alpha # to be determined from experiment
        self.d_alp = self.sig_norm(self.d)
        self.d_bet = self.sig_norm(self.d_p)
        self.dw_h = dw_h
    def get_control_all(self, agent_state):
        # DONE
        '''
        Returns
        ------
        u: 0*3 numpy array, [ax,ay,az]
        u2: 0*len[neighbors_pos] numpy array, dw_flag
        u3: 0*3 numpy array, [x,y,z]
        u4: list of 0*3 numpy array, [x,y,z]
        '''
        self.agent_coords = agent_state[:, :3] ###0,1,2 [x,y,z]
        self.agent_vels = agent_state[:, 3:]  ###3,4,5 [vx,vy,vz]
        self.num_agents = len(self.agent_coords)
        #print("agent_coords is ",self.agent_coords)
        #print("agent_vels is ",self.agent_vels)
        #print("num of agents is ", self.num_agents)
        u = np.zeros_like(self.agent_coords) ##acc
        u2 = np.zeros((self.num_agents,self.num_agents,1)) ##dw_flag
        #u3 = np.zeros_like(self.agent_coords) ##agent_coords
        #u4 = np.zeros_like(self.agent_coords) ##neighbors_pos
        u4 = np.zeros((self.num_agents,self.num_agents,3))
        for j in range(self.num_agents):
            # acc = self.get_control(self.agent_coords[j], self.agent_vels[j])
            acc = self.get_control(self.agent_coords[j], self.agent_vels[j])
            # dw_acc has to be stored, just as how the u[]
            #print("collision_acceleration is ",acc)
            u[j] = acc
            #u2[j] = dw_flag
            #print(j)
            temp = len(self.dw_flag)
            for l in range(temp):
                u2[j,l] = self.dw_flag[l]
                u4[j,l] = self.neighbors_pos[l]
            #u3[j] = self.agent_coords[j]
           #return u
           
        return u, u2, u4

    def get_control(self, agent_coord, agent_vel):
        # agent-agent interaction, repulsion onlyd_p
        '''
        Parameters
        ----------
        agent_coord: 1*3 numpy array, [x, y, z]
        agent_vel: 1*3 numpy array, [vx, vy, vz]
        Returns
        -----------
        acc1: 0*3 numpy array, [ax,ay,az]
        dw_flag:list of 0*1 numpy array of int. 1 means the agent experinces a downwash effect from other agent(s) above
        neighbor_pos: list of 0*3 numpy arrays of floats, each element is a position [x,y,z]
        '''
        neighbor_pos, neighbor_vels= self.get_neighbors(agent_coord)
        # print("neighbor_vels are ",neighbor_vels)
        self.neighbors_pos = neighbor_pos
        sum1_hor = np.zeros(3)
        sum1_ver = np.zeros(3)
        sum2_hor = np.zeros(3)
        sum2_ver = np.zeros(3)
        sum1=np.zeros(3)
        sum2=np.zeros(3)
        for i in range(len(self.neighbors_pos)):
            if self.neighbors_pos[i][0]<100 :
                # sum1 += self.phi_bet(self.sig_norm(neighbors_pos[i]-agent_coord))*self.normal_ik(agent_coord, neighbors_pos[i]) 
                #sum1 += self.phi_bet(neighbors_pos[i]-agent_coord)*self.normal_ik(agent_coord, neighbors_pos[i])
                #sum2 += self.bik(agent_coord, neighbors_pos[i])*(neighbor_vels[i]-agent_vel)
                # xy-plane
                sum1_hor = self.phi_bet_hor(self.sig_norm(self.neighbors_pos[i][:2]-agent_coord[:2]))*self.normal_ik_hor(agent_coord, self.neighbors_pos[i])
                # sum1_hor = self.phi_bet_hor(self.sig_norm(self.neighbors_pos[i][:3]-agent_coord[:3]))*self.normal_ik_hor(agent_coord, self.neighbors_pos[i])
                # print("neighbors_pos ",self.neighbors_pos[i])
                sum2_hor = self.bik_hor(agent_coord[:2], self.neighbors_pos[i][:2])*self.vel_dif_hor(agent_vel, neighbor_vels[i])
                # print("sum2_hor is ",sum2_hor)
                # z-dir
                sum1_ver = self.phi_bet_ver(self.sig_norm(self.neighbors_pos[i][2]-agent_coord[2]))*self.normal_ik_ver(agent_coord, self.neighbors_pos[i])
                # sum1_ver = self.phi_bet_ver(self.sig_norm(self.neighbors_pos[i][:3]-agent_coord[:3]))*self.normal_ik_ver(agent_coord, self.neighbors_pos[i])
                #print("sum1_ver is ",sum1_ver)
                sum2_ver = self.bik_ver(agent_coord[2], self.neighbors_pos[i][2])*self.vel_dif_ver(agent_vel, neighbor_vels[i])
                #print("sum2_ver is ",sum2_ver)
                # sum1_temp = self.phi_bet_hor(self.sig_norm(self.neighbors_pos[i][:3]-agent_coord[:3]))*self.normal_ik_hor(agent_coord, self.neighbors_pos[i])
                # sum2_temp = self.bik_ver(agent_coord[2], self.neighbors_pos[i][2])*(neighbor_vels[i]-agent_vel)
                
                # sum acc in xy-plane and z-dir together
                sum1 += sum1_hor + sum1_ver
                sum2 += sum2_hor + sum2_ver
                # sum1 += sum1_temp
                # sum2 += sum2_temp
                # add dw_acc from Chih-Chun's discovery
                # dw_acc += (dw_flag[i])* self.get_dw_acc(agent_coord[2], neighbors_pos[i][2]) #!!!!!)####!!!!!
        acc1 = self.c1_alp*sum1+self.c2_alp*sum2
        # print(acc1)
        return acc1

    def vel_dif_ver(self, agent_vel, obstacle_vel):
        '''
        Parameters
        ----------
        agent_vel: 0*3 numpy array, [vx, vy, vz]
        obstacle_vel: 0*3 numpy array, [vx, vy, vz]

        Returns
        ----------
        velocity_ver: 0*3 numpy array, [0, 0, vz]
        '''
        temp = obstacle_vel - agent_vel
        temp[:2]=[0,0]
        velocity_ver = temp
        return velocity_ver

    def vel_dif_hor(self, agent_vel, obstacle_vel):
        '''
        Parameters
        ----------
        agent_vel: 0*3 numpy array, [vx, vy, vz]
        obstacle_vel: 0*3 numpy array, [vx, vy, vz]

        Returns
        ----------
        velocity_hor: 0*3 numpy array, [vx, vy, 0]
        '''
        temp = obstacle_vel-agent_vel
        temp[2]=0
        velocity_hor = temp
        return velocity_hor

    def get_virtual_qik_and_pik(self, agent_coord, agent_vel, obstacle):
        ##NOT USED!!!
        '''
        Parameters
        ----------
        agent_coord: 0*3 numpy array, [x, y, z]
        agent_vel: 0*3 numpy array, [vx, vy, vz]
        obstacle: 0*7 numpy array, [x, y, z, r, vx, vy, vz]

        Returns
        -------
        qik: 
        pik:
        '''
        agent_coord = agent_coord.reshape([-1, 1]) #[[x], [y], [z]] 3x1
        agent_vel = agent_vel.reshape([-1, 1]) # [[vx],[vy],[vz]] 3x1
        obs_coord = np.array([obstacle[0], obstacle[1], obstacle[2]]).reshape([-1, 1]) #[[obs_x], [obs_y], [obs_z]] 3x1
        r = obstacle[3] 

        norm = LA.norm(agent_coord-obs_coord) 
        miu = r/norm
        ak = (agent_coord-obs_coord)/norm
        P = np.identity(3)-ak@ak.T
        qik = (miu*agent_coord+(1-miu)*obs_coord).reshape([-1, 3])[0]
        pik = (miu*P@agent_vel).reshape([-1, 3])[0]
        return qik, pik
    
    def phi_bet_hor(self, z):
        '''
        Parameters
        ----------
        z: float, input to action function for agent-obstacles
        
        Returns
        -------
        action: float, repulsion acting on finite distance in xy-plane[deltap_x, deltap_y, delta_z]
        '''
        ## horizontal   
        mult1 = self.bump_func_hor(z/self.d_bet)
        # mult1 = 1
        mult2 = self.sig_norm_grad_scalar(z-self.d_bet) - 1 
        action = mult1 * mult2
        return action

    def phi_bet_ver(self, z):
        '''
        Parameters
        ----------
        z: 0*3 array, input to action function for agent-obstacles
        
        Returns
        -------
        action: float, repulsion acting on finite distance in z-direction [deltap_x, deltap_y, delta_z]
        '''
        ## vertical
        mult1 = self.bump_func_ver(z/self.d_bet)
        # mult1 = 1
        mult2 = self.sig_norm_grad_scalar(z-self.d_bet) - 1 
        action = mult1 * mult2
        return action

    def sig_norm(self, z):
        '''
        Parameters
        ----------
        z: 0*2 for xy or float for z

        Returns
        -------
        sig_nor: float, sigma norm of vector z
        '''
        norm_2 = LA.norm(z)
        temp = np.sqrt(1+self.eps*(norm_2**2))-1
        sig_nor = temp/self.eps
        return sig_nor

    def bik_hor(self, qi, qik):
        '''
        Parameters
        ----------
        qi: 0*2 numpy array, x y position of agent i
        qik: 0*2 numpy array, x y position of obstacle j

        Returns
        -------
        adj: float, adjacency matrix element corresponding to agent i and obstacle k, value between 0 and 1
        '''
        norm_hor = self.sig_norm(qik-qi)
        adj_hor = self.bump_func_hor(norm_hor/self.d_bet)
        return adj_hor

    def bik_ver(self, qi, qik):
        '''
        Parameters
        ----------
        qi: z position of agent i
        qik: z position of obstacle j

        Returns
        -------
        adj: float, adjacency matrix element corresponding to agent i and obstacle k, value between 0 and 1
        '''
        norm_ver = self.sig_norm(qik-qi)
        adj_ver = self.bump_func_ver(norm_ver/self.d_bet)
        return adj_ver
        # return 1

    def normal_ik_hor(self, qi, qik):
        '''
        Parameters
        ----------
        qi: 0*3 numpy array, position of agent i
        qik: 0*3 numpy array, position of obstacle j

        Returns
        -------
        normal_hor: 0*3 numpy array,  horizontal normal vector from agent i to obstacle j
        '''
        denom_hor = np.sqrt(1+self.eps*(LA.norm(qik-qi)**2))
        normal_hor = (qik-qi)/denom_hor 
        # normal_hor = (qi-qik)/denom_hor 
        normal_hor[2] = 0 
        #normal = LA.norm(qik-qi)/denom
        return normal_hor

    def normal_ik_ver(self, qi, qik):
        '''
        Parameters
        ----------
        qi: 0*3 numpy array, position of agent i
        qik: 0*3 numpy array, position of obstacle j

        Returns
        -------
        normal_ver: 0*3 numpy array,  vertical normal vector from agent i to obstacle j
        '''
        denom_ver = np.sqrt(1+self.eps*(LA.norm(qik-qi)**2))
        normal_ver = (qik-qi)/denom_ver
        # normal_ver = (qi-qik)/denom_ver
        normal_ver[:2]=[0,0]
        #normal = LA.norm(qik-qi)/denom
        return normal_ver


    def normal_ik(self, qi, qik):
        '''
        Parameters
        ----------
        qi: 0*3 numpy array, position of agent i
        qik: 0*3 numpy array, position of obstacle j

        Returns
        -------
        normal: 0*3 numpy array,  normal vector from agent i to obstacle j
        '''
        denom = np.sqrt(1+self.eps*(LA.norm(qik-qi)**2))
        normal = (qik-qi)/denom
        #normal = LA.norm(qik-qi)/denom
        return normal

    def sig_norm_grad_scalar(self, z):
        '''
        Parameters
        ----------
        z: float

        Returns
        -------
        grad: float, 1D gradient/derivative of z
        '''
        grad = z/np.sqrt(1+z**2)
        return grad

    def bump_func_hor(self, z):
      #### bump fcn for xy-plane      
        '''
        Parameters
        ----------
        z: float      
        Returns
        -------
        float, soft-cutted value based on r parameter 
        '''
        if z >= 0 and z < self.r_bet:
            return 1
        elif z >= self.r_bet and z <= 1:
            return 0.5*(1+np.cos(np.pi*((z-self.r_bet)/(1-self.r_bet))))
        else:
            return 0 

    def bump_func_ver(self, z):
      #### bump fcn for z-dir      
        '''
        Parameters
        ----------
        z: float      
        Returns
        -------
        float, soft-cutted value based on h parameter 
        '''
        if z >= 0 and z < self.h_bet:
            return 1
        elif z >= self.h_bet and z <= 1:
            return 0.5*(1+np.cos(np.pi*((z-self.h_bet)/(1-self.h_bet))))
        else:
            return 0 

    def get_neighbors(self, agent_coord):
        '''
        Parameters
        ----------
        agent_coord: 1*3 numpy array, [x, y, z]
        
        Returns
        -------
        neighbor_list: 1*3 numpy arrays of floats, each element is a position [x,y,z]
        neighbor_vels: 1*3 numpy arrays of floats, each element is a velocity [vx,vy,vz]
        dw_flag: 1*1 numpy arrays of int. 1 means the agent experiences a downwash effect from other agent(s) above
        '''
        neighbor_list = np.ones((self.num_agents,3))*999
        neighbor_vels = np.ones((self.num_agents,3))*999
        self.dw_flag = np.ones((self.num_agents,1))*999
        for i in range (self.num_agents):
            if not (self.agent_coords[i] == agent_coord).all():
                dist_hor = LA.norm(self.agent_coords[i][:2] - agent_coord[:2])
                dist_ver = LA.norm(self.agent_coords[i][2] - agent_coord[2])
                ### dist= LA.norm() needs to be changed to be cylinder.
                if dist_hor < self.r and dist_ver < self.h_alpha:
                    #neighbor_list.append(self.agent_coords[i])
                    neighbor_list[i] = self.agent_coords[i]
                    #neighbor_vels.append(self.agent_vels[i])
                    neighbor_vels[i] = self.agent_vels[i]
                # check if agent i experinces downwash effect in shape of cylinder
                if self.agent_coords[i][2] - agent_coord[2] > 0 and dist_ver < self.dw_h and dist_hor < math.sqrt(self.A_UAV/2/math.pi): # the agent experiences downwash effect
                    #dw_flag.append(1) # experience downwash effect
                    self.dw_flag[i] = 1
                else:
                    self.dw_flag[i] = 0
                    #dw_flag.append(0) # not experience downwash effect
        # print("downwash_flag are in size of 4*1, ", self.dw_flag)
        # print("next agent ")
        return neighbor_list, neighbor_vels

if __name__ == '__main__':
    # c1_alp, c2_alp, d, d_p, h_bet, eps, r, x_lim, y_lim, z_lim)
    # collision_avoidance = CollisionAvoidance(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2)
    # (c1_alp, c2_alp, d, d_p, r_bet, h_bet, eps, r, dt, A_UAV, h_alpha, dw_h)
    # collision_avoidance = CollisionAvoidance(0, 0, 0, 0, 0, 0, 0.3, 0, 2, 2, 2)
    collision_avoidance = CollisionAvoidance(3, 1, 0.4, 0.3, 0.5, 0.5, 0.1, 0.3, 0.001, 0.008464/4*np.pi, 0.5, 0.5)
