import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.lines import Line2D
import argparse

def trajectory_plotter(file_path):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.set_xlim3d(-3, 3, auto = False)
    ax.set_ylim3d(-2, 2, auto = False)  
    ax.set_zlim3d(0, 2, auto = False)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    agent_states = np.load('/home/charles/catkin_ws/src/CoHunting3D_exp/trajectories/traj'+file_path+'.npy')
    # target_and_vertices = np.load('/home/charles/catkin_ws/src/CoHunting3D_exp/trajectories/setup'+file_path+'.npy')
    # target_coord = target_and_vertices[0]
    # vertex_coord = target_and_vertices[1:5]
    num_agents = agent_states.shape[1]

    # ax.scatter(target_coord[0], target_coord[1], target_coord[2], color = 'red', marker = '*', label = 'target')
    # ax.scatter(vertex_coord[:, 0], vertex_coord[:, 1], vertex_coord[:, 2], color = 'brown', marker = 'X', label = 'vertex_pos')
    for i in range(num_agents):
        ax.scatter(agent_states[:, i, 0], agent_states[:, i, 1], agent_states[:, i, 2], s = 3, label = 'agent'+str(i))
    ax.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','-file_name', help='name of the trajectory npy file', required=True)
    args = vars(parser.parse_args())
    trajectory_plotter(args['f'])





