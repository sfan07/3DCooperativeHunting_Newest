#!/usr/bin/env python3
import rospy
import numpy as np
import time
from datetime import datetime
from crazyflie_class import Crazyflie
from utils import load_yaml, StateSubscriber, get_cross_mtx, rotation_from_axis_angle,\
get_vertices, hungarian_assignment
from std_msgs.msg import Int32MultiArray
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from scipy.interpolate import UnivariateSpline
from trajectory_plotter import trajectory_plotter
from collision_avoidance import CollisionAvoidance

if __name__ == '__main__':
    rospy.init_node('CoHunting3D', anonymous=True, disable_signals=True)
    np.set_printoptions(formatter={'float': lambda x: "{0:.3f}".format(x)})

    cf_list = [0, 1, 5, 4]
    num_agents = len(cf_list)
    active = np.array([True, True, True, True])
    assert(len(active) == len(cf_list))

    # crazyflie setup
    cf_params = load_yaml('cf_params.yaml')
    common_params = cf_params['common']
    cfs = []
    for i in cf_list:
        params = cf_params['cf' + str(i)]
        params.update(common_params)
        cfs.append(Crazyflie(params))

    agent_state = np.zeros((num_agents, 9))
    # initialize target state, hardcode for now
    target_height = 1
    target_state = np.array([1.5, 0, target_height, 0, 0, 0])

    # get the coordinates of the vertices
    l = 0.20
    vertices = get_vertices(target_state[:3], l)

    ##########################################
    # for plotting
    trajectory = []
    target_and_vertices = []
    target_and_vertices.append(target_state[:3])
    for v in vertices:
        target_and_vertices.append(v)
    ##########################################

    # get unit agent and vertex vectors, centered at the target
    vertex_vec = normalize(vertices-target_state[:3], axis = 1, norm = 'l2')

    # define look_ahead for interpolation
    look_ahead_num = 10
    dt = 0.2    

    # collision avoidance controller
    c1_alp = 4    # float, gain for the alpha position term
    c2_alp = 2    # float, gain for the alpha velocity term
    d = 0.3       # float, optimal separation distance between agents for an agent
    d_p = 0.4     # float, distance where the repulsive potential goes to 0
    h_bet = 0.5   # float, steepness of saturation curve for the beta bump
    eps = 0.1     # float, scaling factor in the sigma gradient norm
    r = 1         # float, cut off distance for neightborhood
    avoidance_controller = CollisionAvoidance(c1_alp, c2_alp, d, d_p, h_bet, eps, r)

    try:
        print('publish zero first')
        start_time = time.time()
        while(time.time()-start_time < 1):
           for status, cf in zip(active, cfs):
                if status:
                    cf.pub_raw([0,0,0,0])

        reached = np.array([False]*num_agents) | ~active
        while True:
            for i, cf in enumerate(cfs):
                agent_state[i] = cf.get_state()[0]
            trajectory.append(agent_state.tolist())

            # collision avoidance
            acc = avoidance_controller.get_control_all(agent_state[:,:6])

            # use the next look_ahead_num points for spline interpolation
            ts = np.zeros((look_ahead_num, num_agents))
            xs = np.zeros((look_ahead_num, num_agents))
            ys = np.zeros((look_ahead_num, num_agents))
            look_ahead_pts = np.zeros((look_ahead_num+1, num_agents, 3))
            look_ahead_pts[0] = agent_state[:, :3]

            for n in range (look_ahead_num):
                agent_vec = normalize(look_ahead_pts[n]-target_state[:3], axis = 1, norm = 'l2')
                
                # hungarian algorithm for assignment
                assignment = hungarian_assignment(vertex_vec, agent_vec)
                assigned_vertex_pos = vertices[assignment]
                assigned_vertex_vec = vertex_vec[assignment]

                axis = normalize(np.cross(agent_vec, assigned_vertex_vec), axis = 1, norm = 'l2')
                d_agent_target = norm(look_ahead_pts[n]-target_state[:3], axis = 1)
                d_vertex_target = norm(assigned_vertex_pos-target_state[:3], axis = 1)
                angular_diff = np.arccos(np.diag(agent_vec@assigned_vertex_vec.T))
                radial_diff = d_agent_target-d_vertex_target

                for j in range (num_agents):
                    C = rotation_from_axis_angle(axis[j], 0.05*angular_diff[j])
                    v = C@agent_vec[j]/norm(C@agent_vec[j])
                    waypoint = target_state[:3]+(d_vertex_target[j]+0.95*radial_diff[j])*v
                    # cap the height at 0.1 m
                    if waypoint[-1] < 0.1:
                        waypoint[-1] = 0.1
                    ts[n, j] = n*dt
                    xs[n, j] = waypoint[0]
                    ys[n, j] = waypoint[1]
                    look_ahead_pts[n+1, j] = waypoint

            for j in range(num_agents):
                # interpolate for velocity and (accelerations --> roll, pitch)
                x_interp = UnivariateSpline(ts[:, j], xs[:, j])
                y_interp = UnivariateSpline(ts[:, j], ys[:, j])
                vx_interp = x_interp.derivative()
                vy_interp = y_interp.derivative()
                ax_interp = vx_interp.derivative()
                ay_interp = vy_interp.derivative()

                waypt = look_ahead_pts[5, j]
                vx, vy = vx_interp(5*dt), vy_interp(5*dt)
                ax, ay = ax_interp(5*dt), ay_interp(5*dt)
                
                if active[j]:
                    roll_d = -acc[i,0]*180/(9.81*3.14)
                    pitch_d = acc[i,1]*180/(9.81*3.14)
                    thrust_d = (acc[i,2]+1)*(cf.max_thrust-cf.min_thrust)/2+cf.min_thrust
                    thrust_d = 0
                    print('Agent:', j)
                    print('Distance to vertex:', norm(agent_state[j,:3]-assigned_vertex_pos[j]))
                    cfs[j].approach(waypt, vx, vy, ax, ay, roll_d, pitch_d, thrust_d)

    except KeyboardInterrupt:
        print('emergency landing')
        reached = np.array([False]*num_agents)

        while not np.all(reached):
            for i,cf in enumerate(cfs):
                reached[i] = cf.land()

        # save trajectory
        trajectory = np.array(trajectory)
        target_and_vertices = np.array(target_and_vertices)

        timestamp = datetime.now()
        filename = timestamp.strftime("traj%m_%d_%Y_%H_%M_%S")
        np.save('/home/charles/catkin_ws/src/CoHunting3D_exp/trajectories/'+filename, trajectory)
        filename = timestamp.strftime("setup%m_%d_%Y_%H_%M_%S")
        np.save('/home/charles/catkin_ws/src/CoHunting3D_exp/trajectories/'+filename, target_and_vertices)
        print('trajectory saved:', timestamp.strftime("%m_%d_%Y_%H_%M_%S"))

        trajectory_plotter(timestamp.strftime("%m_%d_%Y_%H_%M_%S"))

        rospy.signal_shutdown('landing complete')