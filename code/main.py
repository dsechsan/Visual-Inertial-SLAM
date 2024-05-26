import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
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

#Pixel to world coordinates
def pTo(pixel_coords,Ks):
    uL,uR,vL,vR = pixel_coords[0] , pixel_coords[1], pixel_coords[2] , pixel_coords[3]
    z = -Ks[2,3]/(uL-vL)
    x =  (uL-Ks[0,2])*z/-Ks[2,3]
    y =  (uR-Ks[1,2])*z/-Ks[2,3] 
    return np.stack((x,y,z,np.ones(x.shape[0])))

# print(pTo(features[:,0:10,0],Ks).shape)

#Optical to world frame coordinates
def oTw(Pose, zt):
    return Pose@imu_T_cam@zt
features = features[:,:20,:]
visited = set()
M = features.shape[1]
map = np.zeros([4,M])
sigmaL = 2*np.eye(3*M)
cam_T_imu = utils.inversePose(imu_T_cam)

for i in tqdm(range(5)):
    zt = features[:,:,i]
    valid = np.unique(np.where(zt>0)[1])
    valid_idx = valid.tolist()
    new_idx = set(valid_idx).difference(visited)
    visited.update(valid_idx)
    new_idx_np = np.array(list(new_idx))
    update_idx_np = np.array(list(set(valid_idx).intersection(visited)))

    if(new_idx_np.shape[0] != 0):
        map[:,new_idx_np] = oTw(pose[t], pTo(zt[:,new_idx_np],Ks))

    Nt = update_idx_np.shape[0]
    if(Nt==0):
        continue
    camTrans = cam_T_imu @ utils.inversePose(pose[t])
    testsave = camTrans@map[:,update_idx_np]

    P = np.zeros([3,4])
    P[:3,:3] = np.eye(3)

    H = np.zeros([4*Nt,3*M])
    for i in range(Nt):
        idx = update_idx_np[i]
        # print(utils.projectionJacobian(camTrans@map[:,update_idx_np]).shape)
        H[i*4:(i+1)*4,idx*3:(idx+1)*3] = Ks @ utils.projectionJacobian(camTrans@map[:,idx]) @ camTrans @ P.T

    V = 3*np.eye(4*Nt)
    sig_Ht = sigmaL@H.T
    KG_L = sig_Ht@np.linalg.inv(H@sig_Ht + V)
    sigmaL = (np.eye(3*M) - KG_L@H)@sigmaL

    a = utils.projection(testsave.T)
    z_pred = Ks @ a.T
    innovL = zt[:,update_idx_np] - z_pred
    map_vec = map_vec + KG_L@innovL.flatten('F')
    map = map_vec.


# print(validl)   

    # new_idx = set(list(valid) - visited)
    # old_idx = 
    # visited.add(list(valid))








