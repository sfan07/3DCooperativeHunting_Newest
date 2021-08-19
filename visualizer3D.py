#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import cv2
from matplotlib.lines import Line2D
import rospy
import os
import sys
from utils import StateSubscriber, AssignmentSubscriber, load_yaml
from visualizer3D_class import Visualizer3D


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:^4.2f}".format(x)})
    rospy.init_node('visualizer3D', anonymous = True, disable_signals = True)
    
    argv = rospy.myargv(argv=sys.argv)
    cf_list = eval(argv[1])

    target_cf = 2
    target_sub = StateSubscriber('cf{}/state'.format(target_cf))

    cf_subs = []
    names = []
    for i in cf_list:
        cf_sub = StateSubscriber('cf{}/state'.format(i))
        cf_subs.append(cf_sub)
        names.append('cf'+str(i))

    x_min = -3
    x_max =  3
    y_min = -2
    y_max =  2
    z_min =  0
    z_max =  2
    trace = None

    vis3d = Visualizer3D(x_min, x_max, y_min, y_max, z_min, z_max, rotate_xy = False, trace = 30000)
    agent_state = np.ones((len(cf_list), 7))

    rospy.sleep(0.5)
    t = 0
    dt = 1/60

    try:
        while not rospy.is_shutdown():
            for i,cf in enumerate(cf_subs):
                agent_state[i] = cf.get_state()[0]
            target_state = target_sub.get_state()[0]
            t += dt
            vis3d.add_state(t, agent_state, target_state, None)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print('closing visualizer')

    