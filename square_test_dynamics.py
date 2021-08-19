#!/usr/bin/env python3
import rospy
import numpy as np
import time

from crazyflie_class import Crazyflie
from utils import PID, load_yaml

if __name__ == '__main__':
    yaml_filename = 'cf_params.yaml'
    yaml_dict = load_yaml(yaml_filename)

    cf_name = 'cf2'

    common_params = yaml_dict['common']
    cf_params = yaml_dict[cf_name]
    cf_params.update(common_params)
    cf = Crazyflie(cf_params)

    points = np.array([[ 0.5,  0.5, 1],
                       [-0.5,  0.5, 1], 
                       [-0.5, -0.5, 1],
                       [ 0.5, -0.5, 1]])
    point_idx = 0


    rospy.init_node('square_test', anonymous=True, disable_signals=True)

    try:
        print('publish zero first')
        start_time = time.time()
        while(time.time()-start_time < 1):
            cf.pub_raw([0,0,0,0])

        waypoint = points[point_idx]
        start_time = time.time()

        while True:
            reached = cf.approach(waypoint, 0, 0, 0, 0)
            rospy.sleep(0.01)

            if reached:
                print('reached waypoint in {} s'.format(time.time() - start_time))
                # if point_idx == points.shape[0]-1:
                #    break

                point_idx = (point_idx + 1)%points.shape[0]
                waypoint = points[point_idx]
                start_time = time.time()

        print('trajectory complete, landing...')
        while not rospy.is_shutdown():
            cf.land()
            rospy.sleep(0.01)

    except KeyboardInterrupt:
        print('emergency landing')
        while not rospy.is_shutdown():
            landed = cf.land()
            rospy.sleep(0.01)
            if landed:
                rospy.signal_shutdown('landing complete')