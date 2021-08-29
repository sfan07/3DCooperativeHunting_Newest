import matplotlib.pyplot as plt
import numpy as np
#This script used to see if the selected agents and targets are within obstacles in 2D

obs_init = np.zeros((3,7))
obs_init[:3,0] = -1.8, -1.8, 0.3 #[x,y,r]
obs_init[:3,1] =   -1,    0, 0.1
obs_init[:3,2] = -0.5,  1.5, 0.5
obs_init[:3,3] =    0, -0.5, 0.2
obs_init[:3,4] =  0.5,  1.5, 0.4
obs_init[:3,5] =  1.7, -0.5, 0.5
obs_init[:3,6] =    0,   -2, 0.5

agent_init = np.zeros((2,4))
agent_init[:2, 0] =  1.5, 1.5
agent_init[:2, 1] = -1.5, 1.5
agent_init[:2, 2] =  1.5, -1.5
agent_init[:2, 3] = -1.0, -1.0

target = np.array([1.0, 0.5])
l = 0.10
xt, yt = target
target_init = np.zeros((2,4))
target_init[:2,0] = xt + l, yt + l
target_init[:2,1] = xt + l, yt - l
target_init[:2,2] = xt - l, yt + l
target_init[:2,3] = xt - l, yt - l

figure, axes = plt.subplots()
for i in range(obs_init.shape[1]):
    circle = plt.Circle((obs_init[0,i], obs_init[1,i] ), obs_init[2,i] )
    axes.add_patch(circle)

for j in range(agent_init.shape[1]):
    axes.scatter(agent_init[0,j],agent_init[1,j],color='r')
    axes.scatter(target_init[0,j],target_init[1,j],color='g')

axes.set_aspect( 1 )

plt.show()