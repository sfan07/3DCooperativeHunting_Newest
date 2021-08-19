#!/usr/bin/env python3
import rospy
import numpy as np
import time
from datetime import datetime
from trajectory_plotter import trajectory_plotter
from crazyflie_class import Crazyflie
from utils import PID, load_yaml
from collision_avoidance import CollisionAvoidance

if __name__ == '__main__':
    rospy.init_node('collision_test', anonymous=True, disable_signals=True)
    np.set_printoptions(formatter={'float': lambda x: "{0:.3f}".format(x)})

    cf_list = [0, 1]
    num_agents = len(cf_list)
    active = np.array([True, True])
    assert(len(active) == len(cf_list))

    # crazyflie setup
    cf_params = load_yaml('cf_params.yaml')
    common_params = cf_params['common']
    cfs = []
    for i in cf_list:
        params = cf_params['cf' + str(i)]
        params.update(common_params)
        cfs.append(Crazyflie(params))

    waypoints = np.array([[ 0.5,  0.5, 0.5],
                          [-0.5, -0.5, 0.5]])
    reached = np.array([False, False])

    # collision avoidance controller
    c1_alp = 8    # float, gain for the alpha position term
    c2_alp = 3    # float, gain for the alpha velocity term
    d = 0.3       # float, optimal separation distance between agents for an agent
    d_p = 0.4     # float, distance where the repulsive potential goes to 0
    h_bet = 0.5   # float, steepness of saturation curve for the beta bump
    eps = 0.1     # float, scaling factor in the sigma gradient norm
    r = 1         # float, cut off distance for neightborhood

    avoidance_controller = CollisionAvoidance(c1_alp, c2_alp, d, d_p, h_bet, eps, r)
    agent_state = np.zeros((num_agents, 9))
    trajectory = []

    try:
        print('publish zero first')
        start_time = time.time()
        while(time.time()-start_time < 1):
            for cf in cfs:
                cf.pub_raw([0,0,0,0])

        while True:
            for i, cf in enumerate(cfs):
                agent_state[i] = cf.get_state()[0]
            trajectory.append(agent_state.tolist())

            # collision avoidance
            acc = avoidance_controller.get_control_all(agent_state[:,:6])
            for i, cf in enumerate(cfs):
                roll_d = -acc[i,0]*180/(9.81*3.14)
                pitch_d = acc[i,1]*180/(9.81*3.14)
                thrust_d = (acc[i,2]+1)*(cf.max_thrust-cf.min_thrust)/2+cf.min_thrust
                thrust_d = 0

                # approach(waypoint, vx, vy, ax, ay, roll_col, pitch_col, thrust_col)
                reached[i] = cf.approach(waypoints[i], 0, 0, 0, 0, roll_d, pitch_d, thrust_d)
                rospy.sleep(0.01)

            if np.all(reached):
                print('swapping places...')
                waypoints[:2,:2] *= -1

                # for i, cf in enumerate(cfs):
                #     cf.approach(waypoints[i], 0, 0, 0, 0)

            rospy.sleep(0.01)

    except KeyboardInterrupt:
        print('emergency landing')
        reached = np.array([False]*num_agents)

        while not np.all(reached):
            for i,cf in enumerate(cfs):
                reached[i] = cf.land()

        # save trajectory
        trajectory = np.array(trajectory)

        timestamp = datetime.now()
        filename = timestamp.strftime("traj%m_%d_%Y_%H_%M_%S")
        np.save('/home/charles/catkin_ws/src/CoHunting3D_exp/trajectories/'+filename, trajectory)

        trajectory_plotter(timestamp.strftime("%m_%d_%Y_%H_%M_%S"))

        rospy.signal_shutdown('landing complete')