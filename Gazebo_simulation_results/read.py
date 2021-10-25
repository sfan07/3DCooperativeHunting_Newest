import rosbag
import numpy as np
import math
# from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as ptch
import matplotlib.cm as cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# import scipy.io as sio # For save .mat
# from scipy.interpolate import interp1d # Linear interpolation

import sys
import os

path = os.path.dirname(os.path.realpath('__file__'))


bag1 = rosbag.Bag(path + '/4_agent_single_target_3.bag')

# print(str(np.size(bag.get_type_and_topic_info()[1].keys())) + ' topics in bag')
print(bag1.get_type_and_topic_info().topics)
# print(bag.get_start_time())
# print(bag.get_message_count())

t_start = {}
topic_restart_time = True
# topic_restart_time = False
for topic in bag1.get_type_and_topic_info().topics:
    print(topic)
    t_start[topic] = 0.0
    # print(message_count)

class uav_data():
    def __init__(self):
        self.local_t = []
        self.x = []
        self.y = []
        self.z = []
        
        self.imu_t = []
        self.roll = []
        self.pitch = []
        self.yaw = []
        
        self.setpoint_t = []
        self.x_d = []
        self.y_d = []
        self.z_d = []        

# uavids = ['uav1', 'uav2', 'uav3', 'uav4', 'uav5'] #,'uav1']
# uavids = ['uav1', 'uav2', 'uav3', 'uav4', 'uav5', 'uav6', 'uav7', 'uav8', 'uav9'] #,'uav1']
uavids = ['uav1', 'uav2', 'uav3', 'uav4', 'uav5']
# uavids = ['uav1', 'uav2']
# uavids = ['uav1']
uavs_data = {}
for i in uavids:
    uavs_data[i] = uav_data()

# gz = False
gz = True
def readbag(bag_name):
    start_time = 0
    # start_time = get_start_time()
    for topic, msg, t in bag_name.read_messages():
        time = msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
        if start_time > time:   continue
        if start_time == 0:     start_time = time
        if t_start[topic] == 0: t_start[topic] = time
        if not gz:
            for i in uavids:
                if not i in topic or time-t_start[topic]<210.75 or time-t_start[topic]>260.0:  continue # or time-t_start[topic]<210.75 or time-t_start[topic]>260.0:  continue # or time-t_start[topic]>50  or time-t_start[topic]<121.75 or time-t_start[topic]>175.0
                if 'global_position' in topic:
                    if topic_restart_time:
                        uavs_data[i].local_t.append(time-t_start[topic])
                    else:
                        uavs_data[i].local_t.append(time-start_time)
                    uavs_data[i].x.append(msg.pose.pose.position.x)
                    uavs_data[i].y.append(msg.pose.pose.position.y)
                    uavs_data[i].z.append(msg.pose.pose.position.z)
                elif 'imu' in topic:
                    if topic_restart_time:
                        uavs_data[i].imu_t.append(time-t_start[topic])
                    else:
                        uavs_data[i].imu_t.append(time-start_time)
                    quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
                    uavs_data[i].roll.append(math.atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]), 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]))*180/3.1415)
                    uavs_data[i].pitch.append(math.asin(2.0 * (quat[2] * quat[0] - quat[3] * quat[1]))*180/3.1415)
                    uavs_data[i].yaw.append(math.atan2(2.0 * (quat[3] * quat[0] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]))*180/3.1415)
                elif 'setpoint' in topic:
                    if topic_restart_time:
                        uavs_data[i].setpoint_t.append(time-t_start[topic])
                    else:
                        uavs_data[i].setpoint_t.append(time-start_time)
                    uavs_data[i].x_d.append(msg.position.x)
                    uavs_data[i].y_d.append(msg.position.y)
                    uavs_data[i].z_d.append(msg.position.z)
                elif 'control_command' in topic:
                    if topic_restart_time:
                        uavs_data[i].setpoint_t.append(time-t_start[topic])
                    else:
                        uavs_data[i].setpoint_t.append(time-start_time)
                    uavs_data[i].x_d.append(msg.Reference_State.position_ref[0])
                    uavs_data[i].y_d.append(msg.Reference_State.position_ref[1])
                    uavs_data[i].z_d.append(msg.Reference_State.position_ref[2])
        '''Gazebo path planning'''
        if gz:
            j = 0
            for i in uavids:
                if msg.cur_position[3*j] != 0:
                    uavs_data[i].x.append(msg.cur_position[3*j])
                    uavs_data[i].y.append(msg.cur_position[3*j+1])
                    uavs_data[i].z.append(msg.cur_position[3*j+2])
                    uavs_data[i].x_d.append(msg.des_position[3*j])
                    uavs_data[i].y_d.append(msg.des_position[3*j+1])
                    uavs_data[i].z_d.append(msg.des_position[3*j+2])
                    uavs_data[i].local_t.append(time-start_time)
                j+=1


def get_start_time():
    flag = False
    for topic, msg, t in bag.read_messages():
        if 'control_command' in topic:
            x_d, y_d, z_d = msg.Reference_State.position_ref[0], msg.Reference_State.position_ref[1], msg.Reference_State.position_ref[2]
            if flag:
                if (x_dd != x_d or y_dd != y_d or z_dd != z_d):
                    print("Start time: ", msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9)
                    return msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
            x_dd, y_dd, z_dd = x_d, y_d, z_d
            flag = True    

def plot_path3D(plot_desire = True, plot_actual = True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    leg = []
    # print(len(uavids))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(uavids))
    for c, i in enumerate(uavids):
        if plot_desire:
            ax.plot(uavs_data[i].x_d, uavs_data[i].y_d, uavs_data[i].z_d, '-', color = cm.hsv(norm(c)))
            leg.extend([i+' goal'])
        if plot_actual:
            ax.plot(uavs_data[i].x, uavs_data[i].y, uavs_data[i].z, '-', linewidth=3, color = cm.hsv(norm(c)))
            leg.extend([i]) #+' local'
    for c, i in enumerate(uavids):
        if (len(uavs_data[i].x_d) == 0): 
            print(i+'desired position is empty!')
            continue
        # ax.plot(uavs_data[i].x_d[-1], uavs_data[i].y_d[-1], uavs_data[i].z_d[-1], '*', markersize=15, color = cm.hsv(norm(c)))
        # ax.plot(uavs_data[i].x_d[0], uavs_data[i].y_d[0], uavs_data[i].z_d[0], 'o', markersize=10, color = cm.hsv(norm(c)))
        

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(azim=180, elev=90) #-110,28 , 2D: -90,90, -120,30
    ax.legend(leg, fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.show()
    
def plot_2D(Title, Legend, x_lab, y_lab, x_data, y_data, num = 1, sub_num = 1):  
    fig = plt.figure()
    for i in range(sub_num):
        plt.subplot(sub_num, 1, i+1)
        for j in range(num):
            plt.plot(x_data[j+num*i], y_data[j+num*i], '-')
        plt.legend(Legend, fontsize=10, loc='lower right')
        plt.xlabel(x_lab)
        plt.ylabel(y_lab[i])
        plt.title(Title[i])
        plt.grid()
    fig.tight_layout()
    plt.show()      

def plot_euclid(Title, Legend, x_lab, y_lab, x_data, y_data, dim):
    fig = plt.figure()
    y = 0
    for i in range(dim):
        y += y_data[i]**2
    y = y**0.5
    plt.plot(np.array(x_data), y, '-')
    plt.legend(Legend, fontsize=5)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(Title)
    plt.grid()
    plt.show()      

def total_travel(data, dim):
    y = 0
    for i in range(dim):
        nxt = np.delete(data[i], 0)
        nxt = np.append(nxt, data[i][-1])
        y += (data[i]-nxt)**2
    y = y**0.5
    print(np.sum(y))

def plot_MAE(xdata, ydata):
    # fig = plt.figure()
    ydata = np.abs(ydata)
    mae = np.mean(ydata)
    plt.plot([xdata[0], xdata[-1]], np.full((2,), mae))
    # plt.show()      


def save_to_mat(file_name, var_name, array):
    sio.savemat(file_name + '.mat', {var_name: array})
    print(file_name + '.mat is saved. With size of ', np.size(array))

def switch_sign(data, dim):
    n = 0
    for i in range(dim):
        diff = np.diff(data[i][::10])
        # diff = np.diff(data[i])
        # diff = diff[diff < 0.001]
        n += len(np.where(diff[:-1]*diff[1:]<0)[0])
        # print(np.where(diff[:-1]*diff[1:]<0)[0])
    print(n)

readbag(bag1)
# readbag(bag2)
# readbag(bag3)
plot_path3D()
# print(uavs_data[agent].x)
# plot_path3D(plot_desire=False)

# uavs_data['uav2'].y = np.asarray(uavs_data['uav2'].y)-2
# uavs_data['uav5'].x = np.asarray(uavs_data['uav5'].x)/5*-3
# uavs_data['uav5'].y = np.asarray(uavs_data['uav5'].y)+1
# uavs_data['uav5'].x_d = np.asarray(uavs_data['uav5'].x_d)/5*-3
# uavs_data['uav5'].y_d = np.asarray(uavs_data['uav5'].y_d)+1
# uavs_data['uav5'].local_t = np.asarray(uavs_data['uav5'].local_t) -210.75
agent = 'uav5'

titles  = ['x-t', 'y-t', 'z-t']
legends = ['GPS Local']
y_label = ['x', 'y', 'z']
x_data  = [uavs_data[agent].local_t, uavs_data[agent].local_t, uavs_data[agent].local_t]
x_data  = [uavs_data[agent].local_t[::10], uavs_data[agent].local_t[::10]]
y_data  = [uavs_data[agent].x, uavs_data[agent].y, uavs_data[agent].z]
y_data  = [uavs_data[agent].x[::10], uavs_data[agent].y[::10]]
# y_data  = [uavs_data[agent].x_d, uavs_data[agent].y_d, uavs_data[agent].z_d]
# plot_2D(titles, legends, 'time (s)', y_label, x_data, y_data, 1, 2)
# total_travel(y_data,2)
# switch_sign(y_data,2)

# fig = plt.figure()

# plt.plot(uavs_data[agent].local_t, uavs_data[agent].x)
# plt.plot(uavs_data[agent].local_t, uavs_data[agent].y)
# plt.plot(uavs_data[agent].local_t, uavs_data[agent].z)
# plot_MAE(uavs_data[agent].local_t, uavs_data[agent].x)
# plot_MAE(uavs_data[agent].local_t, uavs_data[agent].y)
# plot_MAE(uavs_data[agent].local_t, uavs_data[agent].z)
# # plt.plot([xdata[0], xdata[-1]], np.full((2,), mae))
# # plot_2D(titles, legends, 'time (s)', y_label, x_data, y_data, 1, 3)

# plt.legend(['x', 'y', 'z', 'x (MAE)', 'y (MAE)', 'z (MAE)'], fontsize=10)
# plt.xlabel('t (s)')
# plt.ylabel('pos (m)')
# plt.grid()
# plt.show()      


# for agent in uavids:
#     y_data  = [uavs_data[agent].x, uavs_data[agent].y, uavs_data[agent].z]
#     # switch_sign(y_data,3)
#     y_data  = [uavs_data[agent].x, uavs_data[agent].y]
    # switch_sign(y_data,2)
    # total_travel(y_data,2)



goal = [-3, -3, 0]
dis_x = uavs_data[agent].x - np.ones(np.size(uavs_data[agent].x))*goal[0]
dis_y = uavs_data[agent].y - np.ones(np.size(uavs_data[agent].y))*goal[1]
dis_z = uavs_data[agent].z - np.ones(np.size(uavs_data[agent].y))*goal[2]
x_data = uavs_data[agent].local_t
# x_data = np.asarray(x_data).reshape(np.size(x_data))
y_data = [dis_x, dis_y]
# plot_euclid('Euclidean Distance', ['uav1'], 'time (s)', 'Distance (m)', x_data, y_data, 2)


# agent = ''
titles = ['x-t', 'y-t', 'z-t']
y_label = ['x', 'y', 'z']
legends = ['Command Sent', 'GPS Local']
x_data = [uavs_data[agent].setpoint_t, uavs_data[agent].local_t, uavs_data[agent].setpoint_t, uavs_data[agent].local_t,
         uavs_data[agent].setpoint_t, uavs_data[agent].local_t]
y_data = [uavs_data[agent].x_d, uavs_data[agent].x, uavs_data[agent].y_d, uavs_data[agent].y, uavs_data[agent].z_d, uavs_data[agent].z]
# plot_2D(titles, legends, 'time', y_label, x_data, y_data, 2, 3)

# legends = ['Command Sent', 'GSP Local', 'Shifted']
# X = np.array(uavs_data[agent].x) - uavs_data[agent].x[0] + uavs_data[agent].x_d[0]
# Y = np.array(uavs_data[agent].y) - uavs_data[agent].y[0] + uavs_data[agent].y_d[0]
# Z = np.array(uavs_data[agent].z) - uavs_data[agent].z[0] + uavs_data[agent].z_d[0]
# x_data = [uavs_data[agent].setpoint_t, uavs_data[agent].local_t, uavs_data[agent].local_t, uavs_data[agent].setpoint_t, uavs_data[agent].local_t,
#          uavs_data[agent].local_t, uavs_data[agent].setpoint_t, uavs_data[agent].local_t, uavs_data[agent].local_t]
# y_data = [uavs_data[agent].x_d, uavs_data[agent].x, X, uavs_data[agent].y_d, uavs_data[agent].y, Y, uavs_data[agent].z_d, uavs_data[agent].z, Z]
# x_data = [uavs_data[agent].setpoint_t, uavs_data[agent].local_t, uavs_data[agent].setpoint_t, uavs_data[agent].local_t,
#          uavs_data[agent].setpoint_t, uavs_data[agent].local_t]
# y_data = [uavs_data[agent].x_d, uavs_data[agent].x, uavs_data[agent].y_d, uavs_data[agent].y, uavs_data[agent].z_d, uavs_data[agent].z]
agent = 'uav1'
agent2 = 'uav2'
legends = ['uav1', 'uav2']
x_data =    [uavs_data[agent].local_t, uavs_data[agent2].local_t, uavs_data[agent].local_t, uavs_data[agent2].local_t, 
            uavs_data[agent].local_t,uavs_data[agent2].local_t]
y_data =    [uavs_data[agent].x, uavs_data[agent2].x, uavs_data[agent].y, uavs_data[agent2].y, 
            uavs_data[agent].z, uavs_data[agent2].z]
# plot_2D(titles, legends, 'time', y_label, x_data, y_data, 2, 3)

'''Interpolated data'''
'''Get the data with equal time stamp'''
legends = ['Command Sent', 'GPS Local', 'Interp']
uavs_data[agent].x = uavs_data[agent].x[:-100]
uavs_data[agent].y = uavs_data[agent].y[:-100]
uavs_data[agent].z = uavs_data[agent].z[:-100]
uavs_data[agent].local_t = uavs_data[agent].local_t[:-100]

time_interp = uavs_data[agent].local_t
fxd_interp = interp1d(uavs_data[agent].setpoint_t, uavs_data[agent].x_d, fill_value="extrapolate")
fyd_interp = interp1d(uavs_data[agent].setpoint_t, uavs_data[agent].y_d, fill_value="extrapolate")
fzd_interp = interp1d(uavs_data[agent].setpoint_t, uavs_data[agent].z_d, fill_value="extrapolate")
fx_interp = interp1d(uavs_data[agent].local_t, uavs_data[agent].x, fill_value="extrapolate")
fy_interp = interp1d(uavs_data[agent].local_t, uavs_data[agent].y, fill_value="extrapolate")
fz_interp = interp1d(uavs_data[agent].local_t, uavs_data[agent].z, fill_value="extrapolate")

'''Get the data with equal time stamp'''
time_lin = np.linspace(time_interp[0], time_interp[-1], int(time_interp[-1])*100) #(start, stop, num)
x_data = [time_lin, time_lin, time_lin, time_lin, time_lin, time_lin]
y_data = [fxd_interp(time_lin), fx_interp(time_lin), fyd_interp(time_lin), fy_interp(time_lin), fzd_interp(time_lin), fz_interp(time_lin)]
# plot_2D(titles, legends, 'time', y_label, x_data, y_data, 2, 3)


'''Plot Orientation'''
# agent = ''
titles = ['roll-t', 'pitch-t', 'yaw-t']
legends = ['Imu']
y_label = ['roll', 'pitch', 'yaw']
x_data = [uavs_data[agent].imu_t, uavs_data[agent].imu_t, uavs_data[agent].imu_t]
y_data = [uavs_data[agent].roll, uavs_data[agent].pitch, uavs_data[agent].yaw]
# plot_2D(titles, legends, 'time', y_label, x_data, y_data, 1, 3)

'''dwheight.bag'''
titles  = ['']
legends = ['1','2','3','4','5','6','7','8']
# legends = ['alt = 4.5 m', 'alt = 4.0 m', 'alt = 3.5 m','alt = 3.0 m']
y_label = ['z (m)']
x_data  = [uavs_data[agent].x[22:142], -np.asarray(uavs_data[agent].x[182:289]), uavs_data[agent].x[334:460], -np.asarray(uavs_data[agent].x[472:596]),
        uavs_data[agent].x[624:753], -np.asarray(uavs_data[agent].x[759:872]), uavs_data[agent].x[874:1117], -np.asarray(uavs_data[agent].x[1145:1263]),
        uavs_data[agent].x[1288:1428], -np.asarray(uavs_data[agent].x[1431:1689]), uavs_data[agent].x[1728:1856], -np.asarray(uavs_data[agent].x[1896:2006])]
y_data  = [uavs_data[agent].z[22:142], uavs_data[agent].z[182:289], uavs_data[agent].z[334:460], uavs_data[agent].z[472:596],
        uavs_data[agent].z[624:753], uavs_data[agent].z[759:872], uavs_data[agent].z[874:1117], uavs_data[agent].z[1145:1263],
        uavs_data[agent].z[1288:1428], uavs_data[agent].z[1431:1689], uavs_data[agent].z[1728:1856], uavs_data[agent].z[1896:2006]]
# plot_2D(titles, legends, 'x (m)', y_label, x_data, y_data, 12, 1)

x_data  = [uavs_data[agent].x[22:142], uavs_data[agent].x[334:460],
        -np.asarray(uavs_data[agent].x[759:872]), uavs_data[agent].x[874:1000],
        -np.asarray(uavs_data[agent].x[1288:1428]), uavs_data[agent].x[1455:1580], uavs_data[agent].x[1728:1856]]
y_data  = [uavs_data[agent].z[22:142], uavs_data[agent].z[334:460],
        uavs_data[agent].z[759:872], uavs_data[agent].z[874:1000],
        uavs_data[agent].z[1288:1428], uavs_data[agent].z[1455:1580], uavs_data[agent].z[1728:1856]]
legends = ['r','h=0.4','h=0.6','h=1.0','h=1.7','h=2.0','h=2.5']
# plot_2D(titles, legends, 'x (m)', y_label, x_data, y_data, 7, 1)


# dis = abs(np.asarray(uavs_data[agent].x) - np.asarray(uavs_data[agent].x_d))
# print(dis<0.05)
# print(np.nonzero(dis<0.05))
# print(np.asarray(uavs_data[agent].z)[dis<0.05])
# spit = 22:142, 182:289, 334:460, 472:596, 624:753, 759:872, 874:1117, 1145:1263, 1288:1428, 
# 1431:1689, 1728:1856, 1896:2006
'''dw2.bag'''
legends = ['alt = 4.5 m', 'alt = 4.0 m', 'alt = 3.5 m','alt = 3.0 m']
titles  = ['']
y_label = ['z (m)']
x_data  = [uavs_data[agent].x[:86], -np.asarray(uavs_data[agent].x[108:203]), uavs_data[agent].x[216:290], -np.asarray(uavs_data[agent].x[372:468])]
y_data  = [uavs_data[agent].z[:86], uavs_data[agent].z[108:203], uavs_data[agent].z[216:290], uavs_data[agent].z[372:468]]
# plot_2D(titles, legends, 'x (m)', y_label, x_data, y_data, 4, 1)
# print(min(np.asarray(uavs_data[agent].z[:86])))
# print(min(np.asarray(uavs_data[agent].z[108:203])))
# print(min(np.asarray(uavs_data[agent].z[216:290])))
# print(min(np.asarray(uavs_data[agent].z[372:468])))
# print(min(np.asarray(uavs_data[agent].y[:86])))
# print(min(np.asarray(uavs_data[agent].y[108:203])))
# print(min(np.asarray(uavs_data[agent].y[216:290])))
# print(min(np.asarray(uavs_data[agent].y[372:468])))
# print(max(np.asarray(uavs_data[agent].y[:86])))
# print(max(np.asarray(uavs_data[agent].y[108:203])))
# print(max(np.asarray(uavs_data[agent].y[216:290])))
# print(max(np.asarray(uavs_data[agent].y[372:468])))

'''Save array to mat'''
# save_to_mat('GPSLocal_x', 'result_x', fxd_interp(time_lin))
# save_to_mat('GPSLocal_y', 'result_y', fyd_interp(time_lin))
# save_to_mat('GPSLocal_z', 'result_z', fzd_interp(time_lin))
# save_to_mat('setpoint_x', 'result_xd', fx_interp(time_lin))
# save_to_mat('setpoint_y', 'result_yd', fy_interp(time_lin))
# save_to_mat('setpoint_z', 'result_zd', fz_interp(time_lin))
# save_to_mat('time', 'time', time_lin)
