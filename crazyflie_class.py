#!/usr/bin/env python3
import numpy as np
import rospy

from utils import quat2rpy, PID, rpy2rotm
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class StateSubscriber():
    def __init__(self, topic_name):
        """
        Subscribes to the state estimate given the full topic name
        get_state returns the state [x,y,z,vx,vy,vz,roll,pitch,yaw] and the covariance
        of the translation, [x,y,z,vx,vy,vz]
        """
        self._state_msg = Odometry()
        self._state = np.zeros(9)
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

        self._state[6] = quat2rpy(quat)[0]
        self._state[7] = quat2rpy(quat)[1]
        self._state[8] = quat2rpy(quat)[2]
        self._P = np.reshape(msg.pose.covariance, [6,6])


class Crazyflie():
    def __init__(self, params):
        '''
        Subscriber to state estimation
        Publisher to cmd_vel
        
        pub_acc: maps acceleration to angles/thrust and publish
        land: perform emergency landing
        approach: go to a desired waypoint, 
                 return True if reached, measured by position and speed error
        
        '''
        self.name = params['name']

        # land and takeoff parameters
        self.min_thrust = params['min_thrust']
        self.max_thrust = params['max_thrust']

        self.land_thrust = params['land_thrust']
        self.land_height = params['land_height']

        self.takeoff_height = params['takeoff_height']
        self.takeoff_point = None

        self.land_twist = Twist()
        self.land_twist.linear.z = self.land_thrust

        # distance and speed error tolerances
        self.d_thresh = params['d_thresh']
        self.v_thresh = params['v_thresh']

        # PID gains
        z_gains_v = params['pid_z_v']
        z_gains_v['dt'] = params['dt']
        z_gains_a = params['pid_z_a']
        z_gains_a['dt'] = params['dt']
        yaw_gains = params['pid_yaw']
        yaw_gains['dt'] = params['dt']
        self.pid_z_v = PID(**z_gains_v)
        self.pid_z_a = PID(**z_gains_a)
        self.pid_yaw = PID(**yaw_gains)

        # Q = diag([1000, 1000, 10])*0.001
        # R = 0.01
        self.K = np.array([[       0,      10,        0, 17.4568, -0.7527,       0],
                           [     -10,       0, -17.4568,       0,       0, -0.7527]])
        

        # subscriber and publisher setup
        self.state_sub = StateSubscriber(self.name + '/state')
        self.angle_pub = rospy.Publisher(self.name + '/cmd_vel', Twist,
                                         queue_size = 32)
    
    def get_state(self, state_format='row'):
        return self.state_sub.get_state(state_format)

    def pub_acc(self, u_x, u_y, a_z):
        '''
        Parameters
        ----------
        u_x: desired roll
        u_y: desired pitch
        a_z: acceleration in z direction

        automatic yaw control is performed here.
        '''
        state = self.get_state()[0]
        u_z = (a_z + 1)*(self.max_thrust-self.min_thrust)/2 + self.min_thrust
        u_yaw = -self.pid_yaw.control(0, state[8])

        pub_twist = Twist()
        pub_twist.linear.x = u_x
        pub_twist.linear.y = u_y
        pub_twist.linear.z = u_z
        pub_twist.angular.z = u_yaw
        self.angle_pub.publish(pub_twist)

    def pub_raw(self, a):
        pub_twist = Twist()
        pub_twist.linear.x = a[0]
        pub_twist.linear.y = a[1]
        pub_twist.linear.z = a[2]
        pub_twist.angular.z = a[3]
        self.angle_pub.publish(pub_twist)

    def land(self):
        # send a low thrust to descend slowly, then turn off motors
        # return true once the motors are turned off
        state = self.get_state()[0]
        if state[2] < self.land_height:
            self.angle_pub.publish(Twist())
            return True
        else:
            self.angle_pub.publish(self.land_twist)
            return False

    def approach(self, waypoint, vx, vy, ax, ay, roll_col = 0, pitch_col = 0, thrust_col = 0):
        '''
        Use the PID controller to go to a waypoint
        
        Parameters
        ----------
        waypoint : [x,y,z] or [x,y]

        If z is missing, the takeoff height is used.

        Returns true if the waypoint is reached and within distance and speed
        error, else returns false
        '''
        state = self.get_state()[0]   #[x,y,z,vx,vy,vz,roll,pitch,yaw]

        ref_state = np.zeros((6, 1))
        ref_state[0] = waypoint[0]
        ref_state[1] = waypoint[1]
        # need desired velocity and roll, pitch, too
        ref_state[2] = vx
        ref_state[3] = vy
        ref_state[4] = -ax*180/(9.81*3.14)
        ref_state[5] = ay*180/(9.81*3.14)
        

        agent_state = np.zeros(6)
        agent_state[0] = state[0] #x
        agent_state[1] = state[1] #y
        agent_state[2] = state[3] #vx
        agent_state[3] = state[4] #vy
        agent_state[4] = state[6] #roll
        agent_state[5] = state[7] #pitch

        error_state = agent_state.reshape((6, 1))-ref_state
        u_y, u_x = self.K@error_state

        #clipping
        cap = 15
        u_y = np.clip(u_y, -cap, cap)
        u_x = np.clip(u_x, -cap, cap)

        a_z_v = self.pid_z_v.control(waypoint[2], state[2])
        a_z_a = self.pid_z_a.control(a_z_v, state[5])

        print(roll_col, pitch_col)
        u_x += pitch_col
        u_y += roll_col
        a_z_a += thrust_col
        self.pub_acc(u_x, u_y, a_z_a)

        # print('distance to waypoint:', np.linalg.norm(state[0:3] - waypoint))
        if np.linalg.norm(state[0:3] - waypoint) < self.d_thresh:
             if np.linalg.norm(state[3:6]) < self.v_thresh:
                 return True

        return False