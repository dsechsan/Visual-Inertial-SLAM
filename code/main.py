import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm 
import scipy
import pr3_utils as utils

t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = utils.load_data('../data/03.npz')

ut = np.vstack((linear_velocity,angular_velocity)).T
dt = t.flatten() - np.append([0],(t.flatten())[:-1])


pose_inc = dt[:,None]*ut
twist = utils.axangle2twist(pose_inc)

pose = np.tile(np.eye(4),[1010,1,1])

for t in range(1,dt.shape[0]):
    pose[t] = pose[t-1]@scipy.linalg.expm(twist[t])

# plt.plot(pose[:,0,3],pose[:,1,3])
# plt.show()

Ks = np.zeros([4,4])
Ks[0:2,0:3] = K[0:2]
Ks[2,0:3] = K[0]
Ks[2,3] = -K[0,0]*b
Ks[3,0:3] = K[1]

def oTw(pixel_coords,Ks):
    uL,uR,vL,vR = pixel_coords[0] , pixel_coords[1], pixel_coords[2] , pixel_coords[3]
    z = -Ks[2,3]/(uL-vL)
    x =  (uL-Ks[0,2])*z/-Ks[2,3]
    y =  (uR-Ks[1,2])*z/-Ks[2,3] 
    return np.stack((x,y,z,np.ones(x.shape[0])))

print(oTw(features[:,0:10,0],Ks).shape)
