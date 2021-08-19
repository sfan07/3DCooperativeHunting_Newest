import os
import numpy as np
import transformations as tf
import yaml
import rospy
import rospkg
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32MultiArray
from scipy.optimize import linear_sum_assignment


class history_saver():
    def __init__(self):
        self.history = []

    def update(self, data):
        self.history.append(data)
        print(len(self.history))

    def save(self, filename):
        rospack = rospkg.RosPack()
        save_path = rospack.get_path('CoHunting3D_exp')
        save_path = os.path.join(save_path, 'history')
        save_path = os.path.join(save_path, filename)
        self.history = np.array(self.history)
        print(save_path)
        np.save(save_path, self.history)
        print('history saved')

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

    def __repr__(self):
        description = 'PID controller with kp = {}, '.format(self.kp) + \
                      'ki = {}, kd = {}, limits = [{}, {}]'.format(self.ki,
                       self.kd, self.u_min, self.u_max)
        return description


class AssignmentSubscriber():
    def __init__(self, topic_name='assignment'):
        self._assignment_msg = None
        rospy.Subscriber(topic_name, Int32MultiArray, self._assignment_callback)

    def get_assignment(self):
        if self._assignment_msg is None:
            return None
        else:
            data = np.array(self._assignment_msg.data)
            self._assignment_msg = None
            return data

    def _assignment_callback(self, msg):
        self._assignment_msg = msg


class StateSubscriber():
    def __init__(self, topic_name):
        """
        Subscribes to the state estimate given the full topic name
        get_state returns the state [x,y,z,vx,vy,vz,yaw] and the covariance
        of the translation, [x,y,z,vx,vy,vz]
        """
        self._state_msg = Odometry()
        self._state = np.zeros(7)
        self._P = np.eye(6)
        rospy.Subscriber(topic_name, Odometry, self._state_callback)

    def get_state(self, state_format='row'):
        if state_format == 'row':
            return (np.copy(self._state), np.copy(self._P))
        else:
            return np.copy(self._state.reshape([-1,1]), np.copy(self._P))

    def _state_callback(self, msg):
        self._mocap_msg = msg
        self._state[0] = msg.pose.pose.position.x
        self._state[1] = msg.pose.pose.position.y
        self._state[2] = msg.pose.pose.position.z
        self._state[3] = msg.twist.twist.linear.x
        self._state[4] = msg.twist.twist.linear.y
        self._state[5] = msg.twist.twist.linear.z
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

        self._state[6] = quat2rpy(quat)[2]
        self._P = np.reshape(msg.pose.covariance, [6,6])

def load_yaml(filename, path=None):
    '''
    Loads a yaml file given a filename and path
    
    Parameters
    ----------
    filename : yaml filename, without leading slashes
    path : yaml file path, if None defaults to the config file in crazyflie_mpc

    Returns
    -------
    yaml_dict: yaml file dictionary
    '''
    if path is None:
        path = rospkg.RosPack().get_path('CoHunting3D_exp') + '/config/'

    with open(path + filename) as f:
        yaml_dict = yaml.safe_load(f)

    return yaml_dict

def get_neighbour_obs(agent_state, obstacle_state, r):
    '''
    Parameters
    ----------
    agent_state : Nx4 array, each row is [x,y,vx,vy]
    obstacle_state: Nx5 array, each row is [x,y,r,vx,vy]
    r : distance to obstacle surface to be considered neighbours

    Returns
    -------
    neighbour_list : list of numpy arrays
        Each element in the list is a numpy array containing neighbouring obstacle states
        e.g. neighbour_list[n] is a Mx5 array of obstacle states neighbouring agent n
        if none nearby, an empty array is returned in neighbour_list[n]
    '''
        
    # get the distance matrix between all agents
    Dx = (agent_state[:,0].reshape([-1,1]) - obstacle_state[:,0])**2
    Dy = (agent_state[:,1].reshape([-1,1]) - obstacle_state[:,1])**2
    D = Dx + Dy

    # radius of obstacles with added safety distance
    r = obstacle_state[:,2] + r
    
    # the diagonal contains all 0s, change to prevent being queried by argany
    # np.fill_diagonal(D, np.inf)
    
    # get neighbouring indexes for all agents in the distance matrix
    neighbour_idx = np.argwhere(D < r**2)
    
    neighbour_list = []
    num_neighbours = np.zeros(agent_state.shape[0])
    
    for i in range(agent_state.shape[0]):
        # n_idx is a 1d array of row indexes for obstacle_states
        n_idx = neighbour_idx[neighbour_idx[:,0] == i][:,1]
        neighbour_list.append(obstacle_state[n_idx])
        num_neighbours[i] = obstacle_state[n_idx].shape[0]
    
    return neighbour_list, num_neighbours


def quat2rotm(q,order='xyzw'):
    '''
    Parameters
    ----------
    q: list or array of length 4 (array can be 2d or 1d)
        quaternion vector
    order: string, optional
        quaternion notation order. The default is 'xyzw'.
        
    Returns
    -------
    R: 3x3 rotation matrix
    '''    
    if type(q) is np.ndarray:
        q = q.flatten()
    if order == 'wxyz':
        R = tf.quaternion_matrix(q)[0:3,0:3]
    elif order == 'xyzw':
        R = tf.quaternion_matrix([q[3],q[0],q[1],q[2]])[0:3,0:3]
    return R

def quat2rpy(q,order='xyzw'):
    '''
    Parameters
    ----------
    q: list or array of length 4 (array can be 2d or 1d)
        quaternion vector
    order: string, optional
        quaternion notation order. The default is 'xyzw'.
        
    Returns
    -------
    rpy: 1D array of [roll, pitch, yaw]
    '''
    if type(q) is np.ndarray:
        q = q.flatten()
    if order == 'wxyz':
        rpy = np.array(tf.euler_from_quaternion(q))
    elif order == 'xyzw':
        rpy = np.array(tf.euler_from_quaternion([q[3],q[0],q[1],q[2]]))
    return rpy
        
def rpy2rotm(rpy):
    '''
    Parameters
    ----------
    rpy: 1D array of [roll, pitch, yaw]
        
    Returns
    -------
    R: 3x3 rotation matrix
    '''
    if type(rpy) is np.ndarray:
        rpy = rpy.flatten()
    return tf.euler_matrix(*rpy)[0:3,0:3]

def rotm2quat(R, order='xyzw'):
    '''
    Parameters
    ----------
    R: 3x3 array
       rotation matrix
    order: string, optional
        ordering of quaternion output. The default is 'xyzw'.

    Returns
    -------
    quaternion: 1D array
    '''
    R_h = np.eye(4)
    R_h[0:3,0:3] = R
    q = tf.quaternion_from_matrix(R_h)
    if order == 'xyzw':
        return np.array([q[1],q[2], q[3], q[0]])
    else:
        return q
    
def rotm2rpy(R):
    H = np.eye(4)
    H[0:3,0:3] = R
    return np.array(tf.euler_from_matrix(H))
    
def rpy2quat(rpy, order='xyzw'):
    q = np.array(tf.quaternion_from_euler(rpy[0], rpy[1], rpy[2]))
    if order == 'xyzw':
        return np.array([q[1],q[2], q[3], q[0]])
    else:
        return q

def get_cross_mtx(vec):
    x, y, z = vec
    cross_mtx = np.array([[0, -z, y],
                          [z, 0, -x],
                          [-y, x, 0]])
    return cross_mtx

def rotation_from_axis_angle(axis, angle):
    n_cross = get_cross_mtx(axis)
    C = np.eye(3)+np.sin(angle)*n_cross+(1-np.cos(angle))*n_cross@n_cross
    return C

def get_vertices(target_loc ,l):
    xt, yt, zt = target_loc
    vertex_pos = np.array([[xt + l, yt + l, zt + l],
                         [xt + l, yt - l, zt - l],
                         [xt - l, yt + l, zt - l],
                         [xt - l, yt - l, zt + l]])
    return vertex_pos

def hungarian_assignment(self, vertex_vec, agent_vec):
    cost_mtx = -1*(vertex_vec@agent_vec.T)
    assignment = linear_sum_assignment(cost_mtx)[1]
    return assignment