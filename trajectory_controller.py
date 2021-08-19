import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import time

class Trajectory():
    def __init__(self, dpose, \
                 kp_x, kd_x, \
                 kp_y, kd_y, \
                 kp_z, kd_z, \
                 ks_roll,  k1_roll,  k2_roll, \
                 ks_pitch, k1_pitch, k2_pitch, \
                 ks_yaw,   k1_yaw,   k2_yaw,
                 M, g):

        # initialize euler angles and position
        self.roll, self.pitch, self.yaw = np.zeros(3)
        self.vroll, self.vpitch, self.vyaw = np.zeros(3)
        self.x, self.y, self.z  = np.zeros(3)
        self.vx, self.vy, self.vz = np.zeros(3)

        # desired pose
        self.dposition = dpose[:3]
        self.yawd = dpose[-1]

        # initialize the controllers
        self.position_controller = Position_controller(self.dposition, kp_x, kd_x, kp_y, kd_y, kp_z, kd_z, g)
        self.attitude_controller = Attitude_controller(self.yawd, ks_roll, k1_roll, k2_roll, ks_pitch, k1_pitch, k2_pitch, ks_yaw, k1_yaw, k2_yaw, M)

        # step size
        self.dt = 0.001

        # for plotting
        self.t = 0
        self.ts = []
        self.xs = []
        self.ys = []
        self.zs = []
        self.rolls = []
        self.rollds = []
        self.pitchs = []
        self.pitchds = []
        self.yaws = []

    def run(self):
        for i in range (2000):
            # get control
            u1, rolld, pitchd = self.position_controller.get_control(self.yaw)
            u2, u3, u4 = self.attitude_controller.get_control(rolld, pitchd)

            # get position accelerations
            x_acc = (np.cos(self.roll)*np.sin(self.pitch)*np.cos(self.yaw) + np.sin(self.roll)*np.sin(self.yaw))*u1
            y_acc = (np.sin(self.roll)*np.sin(self.pitch)*np.cos(self.yaw) + np.sin(self.roll)*np.cos(self.yaw))*u1
            z_acc = (np.cos(self.roll)*np.cos(self.pitch))*u1

            # get angular accelerations
            roll_acc = u2
            pitch_acc = u3
            yaw_acc = u4

            # update position
            self.vx += self.dt*x_acc
            self.vy += self.dt*y_acc
            self.vz += self.dt*z_acc

            self.x += self.vx*self.dt + 0.5*self.dt*self.dt*x_acc
            self.y += self.vy*self.dt + 0.5*self.dt*self.dt*y_acc 
            self.z += self.vz*self.dt + 0.5*self.dt*self.dt*z_acc

            # update angles 
            self.vroll  += self.dt*roll_acc
            self.vpitch += self.dt*pitch_acc
            self.vyaw   += self.dt*yaw_acc

            self.roll  += self.vroll*self.dt  + 0.5*self.dt*self.dt*roll_acc
            self.pitch += self.vpitch*self.dt + 0.5*self.dt*self.dt*pitch_acc
            self.yaw   += self.vyaw*self.dt   + 0.5*self.dt*self.dt*yaw_acc

            # update errors
            self.position_controller.update_errors(self.x, self.y, self.z)
            self.attitude_controller.update_errors(self.roll, self.pitch, self.yaw, rolld, pitchd)

            # save data for plotting
            self.ts.append(self.t)
            self.xs.append(self.x)
            self.ys.append(self.y)
            self.zs.append(self.z)
            self.rolls.append(self.roll)
            self.rollds.append(rolld)
            self.pitchs.append(self.pitch)
            self.pitchds.append(pitchd)
            self.yaws.append(self.yaw)
            self.t += self.dt
        self.vis()

    def vis(self):
        # POSITION
        fig, axs = plt.subplots(3, 1, constrained_layout = True)
        fig.suptitle('Position Control', fontsize=16)

        dx = self.dposition[0]
        axs[0].plot(np.array(self.ts), dx*np.ones_like(np.array(self.ts)), 'r--', label = 'x_d')
        axs[0].plot(np.array(self.ts), np.array(self.xs), label = 'x')
        axs[0].set_ylabel('x')
        # axs[0].set_ylim(-5, 10)
        axs[0].legend(loc = 1)

        dy = self.dposition[1]
        axs[1].plot(np.array(self.ts), dy*np.ones_like(np.array(self.ts)), 'r--', label = 'y_d')
        axs[1].plot(np.array(self.ts), np.array(self.ys), label = 'y')
        axs[1].set_ylabel('y')
        # axs[1].set_ylim(-5, 10)
        axs[1].legend(loc = 1)

        dz = self.dposition[2]
        axs[2].plot(np.array(self.ts), dz*np.ones_like(np.array(self.ts)), 'r--', label = 'z_d')
        axs[2].plot(np.array(self.ts), np.array(self.zs), label = 'z')
        axs[2].set_ylabel('z')
        # axs[2].set_ylim(-5, 10)
        axs[2].legend(loc = 1)
        plt.show()

        # ATTITUDE
        fig, axs = plt.subplots(3, 1, constrained_layout = True)
        fig.suptitle('Attitude Control', fontsize=16)

        axs[0].plot(np.array(self.ts), np.array(self.rollds), 'r--', label = 'roll_d')
        axs[0].plot(np.array(self.ts), np.array(self.rolls), label = 'roll')
        axs[0].set_ylabel('roll')
        # axs[0].set_ylim(-1, 1)
        axs[0].legend(loc = 1)

        axs[1].plot(np.array(self.ts), np.array(self.pitchds), 'r--', label = 'pitch_d')
        axs[1].plot(np.array(self.ts), np.array(self.pitchs), label = 'pitch')
        axs[1].set_ylabel('pitch')
        # axs[1].set_ylim(-1, 1)
        axs[1].legend(loc = 1)

        axs[2].plot(np.array(self.ts), self.yawd*np.ones_like(np.array(self.ts)), 'r--', label = 'yaw_d')
        axs[2].plot(np.array(self.ts), np.array(self.yaws), label = 'yaw')
        axs[2].set_ylabel('yaw')
        # axs[2].set_ylim(-1, 1)
        axs[2].legend(loc = 1)
        plt.show()

class Position_controller():
    def __init__(self, dposition, kp_x, kd_x, kp_y, kd_y, kp_z, kd_z, g):
        # proportional gains and derivative gains
        self.kp_x = kp_x
        self.kd_x = kd_x
        self.kp_y = kp_y
        self.kd_y = kd_y
        self.kp_z = kp_z
        self.kd_z = kd_z        

        # desired position
        self.xd = dposition[0]
        self.yd = dposition[1]
        self.zd = dposition[2]

        # init errors
        self.x_e  = 0.
        self.x_de = 0.
        self.y_e  = 0.
        self.y_de = 0.
        self.z_e  = 0.
        self.z_de = 0.

        # step size
        self.dt = 0.001

        # constants
        self.g = g

    def get_control(self, yaw):
        '''
        takes in desired position, output the u1, desired roll, and desired pitch
        '''
        u1x = -1 * self.kp_x * self.x_e - self.kd_x * self.x_de
        u1y = -1 * self.kp_y * self.y_e - self.kd_y * self.y_de
        u1z = -1 * self.kp_z * self.z_e - self.kd_z * self.z_de + self.g

        rolld =  np.arctan((np.sin(yaw)*np.cos(yaw)*u1x - np.cos(yaw)*np.cos(yaw)*u1y)/(u1z))
        pitchd = np.arcsin((np.cos(yaw)*np.cos(yaw)*u1x - np.sin(yaw)*np.cos(yaw)*u1y)/(u1z))
        u1 = u1z / (np.cos(pitchd)*np.cos(rolld))   

        return u1, rolld, pitchd

    def update_errors(self, x, y, z):
        self.x_de = ((x - self.xd) - self.x_e)/self.dt
        self.y_de = ((y - self.yd) - self.y_e)/self.dt
        self.z_de = ((z - self.zd) - self.z_e)/self.dt

        self.x_e = x - self.xd
        self.y_e = y - self.yd
        self.z_e = z - self.zd

class Attitude_controller():
    def __init__(self, yawd, ks_roll,  k1_roll,  k2_roll, \
                            ks_pitch, k1_pitch, k2_pitch, \
                              ks_yaw,   k1_yaw,   k2_yaw, \
                                                       M):
        # sliding function gains, proportional gains, and derivative gains
        self.ks_roll = ks_roll
        self.k1_roll = k1_roll
        self.k2_roll = k2_roll
        self.ks_pitch = ks_pitch
        self.k1_pitch = k1_pitch
        self.k2_pitch = k2_pitch
        self.ks_yaw = ks_yaw
        self.k1_yaw = k1_yaw
        self.k2_yaw = k2_yaw
        self.M = M

        # init errors
        self.roll_e = 0.
        self.roll_de = 0.
        self.roll_ie = 0.
        self.pitch_e = 0.
        self.pitch_de = 0.
        self.pitch_ie = 0.
        self.yaw_e = 0.
        self.yaw_de = 0.
        self.yaw_ie = 0.

        self.yawd = yawd

        # step size
        self.dt = 0.0001
        
    def get_control(self, rolld, pitchd):
        S_roll  = self.roll_de + self.k1_roll*(self.roll_e)  + self.k2_roll*self.roll_ie
        S_pitch = self.pitch_de + self.k1_pitch*(self.pitch_e) + self.k2_pitch*self.pitch_ie
        S_yaw   = self.yaw_de + self.k1_yaw*(self.yaw_e)   + self.k2_yaw*self.yaw_ie

        u2 = -1 * self.M*np.sign(S_roll)  - self.ks_roll*S_roll   - self.k1_roll*self.roll_de   - self.k2_roll*self.roll_e
        u3 = -1 * self.M*np.sign(S_pitch) - self.ks_pitch*S_pitch - self.k1_pitch*self.pitch_de - self.k2_pitch*self.pitch_e
        u4 = -1 * self.M*np.sign(S_yaw)   - self.ks_yaw*S_yaw     - self.k1_yaw*self.yaw_de     - self.k2_yaw*self.yaw_e

        return u2, u3, u4

    def sat(self, S):
        '''
        saturation function, not used for now
        '''
        if np.abs(y) <= 1:
            return y
        else:
            return np.sign(y)

    def update_errors(self, roll, pitch, yaw, rolld, pitchd):
        '''
        update error, differential error, and integral error
        '''
        self.roll_de  = ((roll  - rolld)  - self.roll_e) /self.dt
        self.pitch_de = ((pitch - pitchd) - self.pitch_e)/self.dt
        self.yaw_de   = ((yaw   - self.yawd) - self.yaw_e)  /self.dt

        self.roll_e = (roll - rolld)
        self.pitch_e = (pitch - pitchd)
        self.yaw_e = (yaw - self.yawd)

        self.roll_ie += self.roll_e
        self.pitch_ie += self.pitch_e
        self.yaw_ie += self.yaw_e

if __name__ == '__main__':
    # controller gains
    kp_x = kp_y = 1.8
    kd_x = kd_y = 2.5
    kp_z = 5.5
    kd_z = 4.5
    M = 25
    ks_roll = ks_pitch = ks_yaw = 1
    k1_roll = 5.5
    k2_roll = 10
    k1_pitch = 5.5
    k2_pitch = 40
    k1_yaw = 8.5
    k2_yaw = 20

    g = 9.8

    # desired pose [x, y, z, yaw]
    dpose = np.array([4., 3., 5., np.pi/4])

    traj = Trajectory(dpose, kp_x, kd_x, kp_y, kd_y, kp_z, kd_z, \
                             ks_roll, k1_roll, k2_roll, \
                             ks_pitch, k1_pitch, k2_pitch, \
                             ks_yaw, k1_yaw, k2_yaw,
                             M, g)
    traj.run()


