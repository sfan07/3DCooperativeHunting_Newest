#!/usr/bin/env python3
import rospy
import numpy as np
import sys

from utils import rpy2rotm, quat2rotm, quat2rpy, rotm2quat

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

import time


class SlidingWindow():
    def __init__(self, init_state, window_size,  dt):
        """ 
        size: size of the sliding window
        x0: initial state of the form [x,y,z].T
        dt: timestep 
        """
        self.window = np.tile(init_state, [1, window_size])
        self.window_idx = 0
        self.window_size = window_size
        self.dt = dt

    def update(self, state):
        # given state [x,y,z].T, return [vx, vy, vz].T
        self.prev_avg = np.mean(self.window, axis =1)
        self.window[:, self.window_idx] = state.flatten()
        self.window_idx = (self.window_idx + 1) % self.window_size
        self.cur_avg = np.mean(self.window, axis = 1)
        return ((self.cur_avg - self.prev_avg) / self.dt).reshape((-1,1))


class StateEstimator():
    """
    A class which estimates the position and velocity of a crazyflie,
    assuming a constant discretized acceleration model
    
    Only the translation vector is fed into KF, angular data is unchanged.
    """
    def __init__(self, trackable_name, mocap_R, mocap_t, init_state, window_size,  dt):
        """
        Parameters
        ----------
        trackable_name : string, name of the trackable
        mocap_R : 3x3 rotation matrix from the raw mocap frame to desired frame
        mocap_t : 3x1 translation from the raw mocap frame to desired frame
        init_state : initial state in form [x,y,z,vx,vy,vz,yaw].T
        dt : discretization time for the estimator, also the publish delay
        """
        # mass, correction transforms, and Kalman filter initialization


        self.m = 0.03
        self.dt = dt
        self.mocap_R = np.copy(mocap_R)
        self.mocap_t = np.copy(mocap_t)
        self.window = SlidingWindow(init_state[:3], window_size, dt)
        
        # topic names
        self.mocap_topic = '/vrpn_client_node/' + trackable_name + '/pose'
        self.state_topic = trackable_name + '/state'

        # messages and numpy arrays that change in the callbacks
        self._mocap_msg = PoseStamped()
        self._mocap_msg_prev = PoseStamped()
        self._state_msg = Odometry()
        self._state_msg.pose.pose.orientation.w = 1
        self._state = np.copy(init_state)

        # subscriber and publisher initialization-
        rospy.Subscriber(self.mocap_topic, PoseStamped, self.mocap_callback)
        self.state_pub = rospy.Publisher(self.state_topic, Odometry, queue_size=32)
        self.update_timer = rospy.Timer(rospy.Duration(dt), self.update_callback)


    def get_state(self):
        return np.copy(self.state)

    def mocap_callback(self, msg):
        self._mocap_msg = msg

    def update_callback(self, update_timer):
        """
        Compute state estimates and publish them, update the state if mocap
        data is available by feeding it to a Kalman filter. The state estimate
        is then published as an odometry message.
        
        Orientation is handled here rather than in the KF, which is sloppy but
        good enough if the mocap publishes at ~100Hz.
        """
        start_time = time.time()

        y = np.array([[self._mocap_msg.pose.position.x], 
                      [self._mocap_msg.pose.position.y],
                      [self._mocap_msg.pose.position.z]])
        y = self.mocap_R @ y + self.mocap_t

        quat_raw = [self._mocap_msg.pose.orientation.x, 
                    self._mocap_msg.pose.orientation.y, 
                    self._mocap_msg.pose.orientation.z,
                    self._mocap_msg.pose.orientation.w,]
        R = self.mocap_R @ quat2rotm(quat_raw)
        quat = rotm2quat(R)


        self._state[0:3] = y        
        self._state[3:6] = self.window.update(self._state[0:3])
        self._state[-1,0] = quat2rpy(quat)[2]

        # update and publish the estimated state
        timestamp = rospy.get_rostime()
        self._state_msg.header.stamp.secs = timestamp.secs
        self._state_msg.header.stamp.nsecs = timestamp.nsecs
        self._state_msg.pose.pose.position.x = self._state[0]
        self._state_msg.pose.pose.position.y = self._state[1]
        self._state_msg.pose.pose.position.z = self._state[2]
        self._state_msg.twist.twist.linear.x = self._state[3]
        self._state_msg.twist.twist.linear.y = self._state[4]
        self._state_msg.twist.twist.linear.z = self._state[5]
        self._state_msg.pose.pose.orientation.x = quat[0]
        self._state_msg.pose.pose.orientation.y = quat[1]
        self._state_msg.pose.pose.orientation.z = quat[2]
        self._state_msg.pose.pose.orientation.w = quat[3]
        self.state_pub.publish(self._state_msg)

if '__main__' == __name__:

    rospy.init_node('sliding_window_test', anonymous=True, disable_signals=True)
    argv = rospy.myargv(argv=sys.argv)
    cf_list = eval(argv[1])

    cf_state_estimators = []
    for i in cf_list:
        c = StateEstimator('cf'+str(i), np.eye(3), np.zeros((3,1)), np.zeros((7,1)),  3, 1/100)
        cf_state_estimators.append(c)

    target_estimator = StateEstimator('irobot', np.eye(3), np.zeros((3,1)), np.zeros((7,1)),  3, 1/100)
    rospy.spin() 