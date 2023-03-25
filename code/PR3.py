# %%
import numpy as np
from pr3_utils import *
from scipy.linalg import expm
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
filename = "../data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
t = np.squeeze(t)

Pose = np.zeros([3026,4,4]) # mu_t
Pose[:,0,0], Pose[:,1,1], Pose[:,2,2], Pose[:,3,3] = 1, 1, 1, 1

# PoseVar = np.diag([1e-1,1e-1,1e-1,1e-3,1e-3,1e-3])
PoseVar = np.eye(6)
PoseVar = np.repeat(PoseVar[np.newaxis,:,:], 3026, axis=0) #del_mu

NoiseCoVar = (1e-2)*np.identity(6) # W

tau = t - np.append(t[0],t[0:-1])
ut = np.vstack((linear_velocity,angular_velocity)).T
twist_t = axangle2twist(ut) # twist at time t
pose_increment = tau[:,np.newaxis,np.newaxis]*twist_t #Tau*ut(hat)
pose_increment = pose_increment.astype(np.double)


# %%
for i in range(1,3026):
    Pose[i] = Pose[i-1]@expm(pose_increment[i])
    PoseVar[i] = pose2adpose(expm(-pose_increment[i]))@PoseVar[i-1]@pose2adpose(expm(-pose_increment[i])).T #+ NoiseCoVar

visualize_trajectory_2d(np.moveaxis(Pose,0,-1),show_ori=True)

# %%
Ks = np.zeros([4,4])
Ks[0:2,0:3] = K[0:2]
Ks[2,0:3] = K[0]
Ks[2,3] = -K[0,0]*b
Ks[3,0:3] = K[1]

cu,cv= Ks[0,2],Ks[1,2]
fsu,fsv = Ks[0,0],Ks[1,1]

def initialLandmarks(obs):
    uL,uR,vL,vR = obs[0],obs[1],obs[2],obs[3]
    z = fsu*b/(uL-vL)
    x = (uL-cu)*z/fsu
    y = (uR-cv)*z/fsv
    return np.stack((x,y,z,np.ones(x.shape)))

features20 = features[:,::30,:]

# %%
P = np.zeros([3,4])
P[:3,:3] = np.eye(3)

# %%
features20 = features[:,::30,:]
M = features20.shape[1]
mapDR = np.zeros([M,4])
map_vec = np.reshape(np.delete(mapDR,3,axis=1),[3*M,1])
zt = []
for t in tqdm(range(3026)):
    zt.append(np.squeeze(np.where(features20[0,:,t] != -1)))
    if(t==0):
        mcam = initialLandmarks(np.squeeze(features20[:,zt[0],0]))
        # map[zt[0]] = ((Pose[0]@inversePose(imu_T_cam)@mcam).T)
        mapDR[zt[0]] = ((Pose[0]@(imu_T_cam)@mcam).T)
        sig = 2*np.eye(M*3)
        continue

    new_ele = np.setdiff1d(zt[t],zt[t-1])
    # inter_ele = np.intersect1d(zt[t],zt[t-1])
    mcam = initialLandmarks(np.squeeze(features20[:,new_ele,t]))
    mapDR[new_ele] = ((Pose[t]@(imu_T_cam)@mcam).T)
    
    #Innovation
    obs1 = (Ks@(projection(((inversePose(imu_T_cam)@inversePose(Pose[t]))@(mapDR[zt[t]].T)).T).T))
    innovation = np.squeeze(features20[:,zt[t],t]) - obs1
    # print(innovation)
    Nt = np.size(zt[t])
    if(Nt == 1):
        continue
    # print(innovation)
    #Jacobian
    H = np.zeros([4*Nt,3*M])
    for i in range(Nt):
        index = zt[t][i]
        H[i*4:i*4+4,index*3:index*3+3] = Ks@projectionJacobian(inversePose(imu_T_cam)@inversePose(Pose[t])@(mapDR[zt[t][i]]))@inversePose(imu_T_cam)@inversePose(Pose[t])@(P.T)
    V = np.eye(4*Nt)*3

    #Update Step
    map_vec = np.reshape(mapDR[:,:-1], [3*M, 1])
    KalmanGain = sig@H.T@(np.linalg.inv(H@sig@H.T+V))
    sig = (np.eye(3*M) - KalmanGain@H)@sig
    map_vec = map_vec + np.einsum('ij,jk->ik', KalmanGain, np.reshape(innovation,[KalmanGain.shape[1],1]))
    mapDR[:,:-1] = np.reshape(map_vec, [M,3])

    

# %%
plt.scatter(mapDR[:,0],mapDR[:,1],marker='.',color='royalblue')
plt.plot(Pose[:1008,0,3],Pose[:1008,1,3],color='red')
plt.xlim([-1200,500])
plt.ylim([-800,800])
# plt.xlim([-1100,400])
# plt.ylim([-500,300])

# %% [markdown]
# ### SLAM
# 
# Steps: 
# 1. Prediction:
#     Part a; Modify the noise?
# 2. Update:
# Use the predicted mean and variance as inputs to the update step
# Calculate the Jacobian H wrt T i.e., $$\in$$ R (4N+1)*6
# Use the Jacobian to compute Kalman Gain and then the others

# %%
def circleDot(mulm):
    out = np.zeros([4,6])
    out[:3,:3] = np.eye(3)
    out[:3,3:] = -axangle2skew(mulm[:3].T)
    return out

# %%
features20 = features[:,::20,:]
M = features20.shape[1]

PoseSL = np.zeros([3026,4,4],dtype=np.double)
PoseSL[0] = np.eye(4)

# PoseVarSL = np.diag([0.05,0.05,0.05,0.05,0.05,0.05])
PoseVarSL = 0.05*np.eye(6,dtype=np.double)
PoseVarSL = np.repeat(PoseVarSL[np.newaxis,:,:], 3026, axis=0) 
NoiseSL = (1e-2)*np.eye(6) # W
# NoiseSL = np.diag([1,1,1,1e-1,1e-1,1e-1])
NoiseSL = NoiseSL.astype(np.double)

SigmaSL = 2*np.eye(3*M + 6,dtype=np.double)
map = np.zeros([M,4],dtype=np.double)

#Landmarks at time t=0
zt = []
zt.append(np.squeeze(np.where(features20[0,:,0] != -1)))
mcam = initialLandmarks(np.squeeze(features20[:,zt[0],0]))
mcam = mcam.astype(np.double)
map[zt[0]] = ((PoseSL[0]@(imu_T_cam)@mcam).T)
map_vec = np.reshape(np.delete(map,3,axis=1),[3*M,1]) #flattened world frame coordinates
map_vec = map_vec.astype(np.double)

cam_T_imu = inversePose(imu_T_cam)
cam_T_imu = cam_T_imu.astype(np.double)

for t in tqdm(range(1,1000)):
    #Prediction Step
    PoseSL[t] = PoseSL[t-1]@expm(pose_increment[t])
    F = pose2adpose(expm(-pose_increment[t])) 
    PoseVarSL[t] = F @PoseVarSL[t-1]@ F.T + NoiseSL
    SigmaSL[-6:,-6:] = PoseVarSL[t]
    SigmaSL[:3*M,3*M:3*M+6] = SigmaSL[:3*M,3*M:3*M+6] @ F.T
    SigmaSL[3*M:,:3*M] = SigmaSL[:3*M,3*M:3*M+6].T 
    IPoset = inversePose(PoseSL[t])

    #Read valid indices
    zt.append(np.squeeze(np.where(features20[0,:,t] != -1)))
    # print(zt)
    Nt = np.size(zt[t])
    if(Nt == 1):
        continue

    # Landmark initialization
    new_ele = np.setdiff1d(zt[t],zt[t-1])
    mcam = initialLandmarks(np.squeeze(features20[:,new_ele,t]))
    # print(new_ele)
    map[new_ele] = ((PoseSL[t]@(imu_T_cam)@mcam).T) 

    #Innovation
    # obs1 = (Ks@(projection(((inversePose(imu_T_cam)@inversePose(PoseSL[t]))@(map[zt[t]].T)).T).T))
    obs1 = Ks@(projection(((cam_T_imu @IPoset) @(map[zt[t]].T)).T).T)
    innovation = np.squeeze(features20[:,zt[t],t]) - obs1
    # print(innovation)
    #Jacobian
    H = np.zeros([4*Nt,3*M + 6])
    
    for i in range(Nt):
        index = zt[t][i]
        # print(index)
        H[i*4:i*4+4,index*3:index*3+3] = Ks @projectionJacobian(cam_T_imu @ IPoset @ map[zt[t][i]]) @cam_T_imu @IPoset @(P.T)
        H[i*4:i*4+4,-6:] = -Ks @projectionJacobian(cam_T_imu @IPoset @map[zt[t][i]]) @cam_T_imu @circleDot(IPoset @map[zt[t][i]])

    V = np.eye(4*Nt)*3

    #Update Step
    # print(KalmanGain.shape, SigmaSL.shape, H.shape)
    map_vec = np.reshape(np.delete(map,3,axis=1),[3*M,1])
    if (np.linalg.det(H @ SigmaSL @H.T + V) == 0):
        continue
    KalmanGainSL = SigmaSL @ H.T @ (np.linalg.pinv(H @ SigmaSL @H.T + V)) 

    # KalmanGain_pos = SigmaSL[-6:,-6:] @ H[:,-6:].T @ (np.linalg.inv(H[:,-6:] @ SigmaSL[-6:,-6:] @H[:,-6:].T + V)) 
    
    SigmaSL = (np.eye(3*M + 6) - KalmanGainSL@H)@SigmaSL
    PoseVarSL[t] = SigmaSL[-6:,-6:]
    map_vec = map_vec + KalmanGainSL[:-6,:]@(np.reshape(innovation,[KalmanGainSL.shape[1],1]))
    map = np.concatenate((np.reshape(map_vec,[M,3]),np.ones([M,1])),axis=1)
    # print(map)
    
    # PoseSL[t] = PoseSL[t]@expm(axangle2twist((0.001)*(KalmanGain@np.reshape(innovation,KalmanGain.shape[1]))[-6:]))

# %%
# ### Auxiliary Code

# %%
plt.scatter(map[:,0],map[:,1],marker='.',color='royalblue')
plt.plot(PoseSL[:,0,3],PoseSL[:,1,3],color='red')
plt.plot(Pose[:,0,3],Pose[:,1,3],color='green')
plt.xlim([-1100,400])
plt.ylim([-500,300])



