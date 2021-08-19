from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.animation
import numpy as np
import cv2
from matplotlib.lines import Line2D

class Visualizer3D():
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, rotate_xy = True, trace = 20):
        self.agent_clr = 'b'
        self.target_clr = 'r'
        self.obstacle_clr = 'k'

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.rotate_xy = rotate_xy
        
        self.init_ax()

    def get_vertices(self, target_loc ,l):
        xt, yt, zt = target_loc
        vertex_pos = np.array([[xt + l, yt + l, zt + l],
                             [xt + l, yt - l, zt - l],
                             [xt - l, yt + l, zt - l],
                             [xt - l, yt - l, zt + l]])
        return vertex_pos

    def init_ax(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')
        self.legend_elements = [Line2D([0], [0], marker = 'o', color = 'w', label = 'Agents',
                                  markerfacecolor = self.agent_clr, markersize = 10),
                                Line2D([0], [0], marker = 'o', color = 'w', label = 'Target',
                                  markerfacecolor = self.target_clr, markersize = 10),
                                Line2D([0], [0], marker = 'o', color = 'w', label = 'vertex_pos',
                                  markerfacecolor = 'black', markersize=10)]
        
    def reset_ax(self, t):
        self.ax.clear()
        self.ax.set_title('t: {:.2f}s'.format(t))
        if self.rotate_xy:
            self.ax.set_xlabel('y(m)')
            self.ax.set_ylabel('x(m)')
            self.ax.set_xlim3d(self.y_min, self.y_max, auto = False)
            self.ax.set_ylim3d(self.x_min, self.x_max, auto = False)
        else:
            self.ax.set_xlabel('x(m)')
            self.ax.set_ylabel('y(m)')
            self.ax.set_xlim3d(self.x_min, self.x_max, auto = False)
            self.ax.set_ylim3d(self.y_min, self.y_max, auto = False)
        self.ax.set_zlim3d(self.z_min, self.z_max, auto = False)
        self.ax.legend(handles = self.legend_elements, loc = 'upper right')

    def add_state(self, t, agent_state, target_state, obstacle_state):
        self.reset_ax(t)
        if agent_state is not None:
            agent_xs, agent_ys, agent_zs = agent_state[:, 0], agent_state[:, 1], agent_state[:, 2]
            self.ax.scatter(agent_xs, agent_ys, agent_zs, c = self.agent_clr, s = 100)
        if target_state is not None:
            l = 0.30
            vertex_pos = self.get_vertices(target_state[:3], l)
            target_x, target_y, target_z = target_state[0], target_state[1], target_state[2]
            self.ax.scatter(target_x, target_y, target_z, c = self.target_clr, s = 200)
            self.ax.scatter(vertex_pos[:, 0], vertex_pos[:,1], vertex_pos[:,2], c = 'black', s = 50)
        if obstacle_state is not None: 
            obstacle_xs, obstacle_ys, obstacle_zs = obstacle_state[:, 0], obstacle_state[:, 1], obstacle_state[:, 2]     
            self.ax.scatter(obstacle_xs, obstacle_ys, obstacle_zs, c = self.obstacle_clr, s = 100)
        plt.pause(0.001)
