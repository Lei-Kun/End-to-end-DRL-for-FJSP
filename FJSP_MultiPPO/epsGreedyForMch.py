import numpy as np
import torch
import copy
#from FJSP_Env import FJSP
from permiss import permissibleLeftShift
from updateEndTimeLB import calEndTimeLB
from uniform_instance import uni_instance_gen
def PredictMch(env,actions,eps=0.1):
    #np.random.seed(SEED)
    mch_as = []
    REWARDS = []
    for j,action in enumerate(actions):
        x=np.random.uniform(0,1,1)

        durMch=copy.deepcopy(env.dur_cp[j])

        row = action // durMch.shape[0]  # 取整除
        col = action % durMch.shape[1]  # 取余数
        mchfora=np.where(durMch[row][col]>0)[0]
        if x<=eps:
            mch_a = np.random.choice(mchfora)
            mch_as.append(mch_a)


        else:
            rewards = []
            #print('time',1111111,durMch[row][col],mchfora)
            for i in mchfora:

                mchsStartTimes,opIDsOnMchs,mchsEndTimes,temp1 = copy.deepcopy(env.mchsStartTimes[j]),copy.deepcopy(env.opIDsOnMchs[j]),copy.deepcopy(env.mchsEndTimes[j]),copy.deepcopy(env.temp1[j])
                mchmat = copy.deepcopy(env.m[j])
                startTime_a, _ = permissibleLeftShift(a=action, mch_a=i, durMat=durMch, mchMat=mchmat,
                                                         mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs,
                                                         mchEndTime=mchsEndTimes)
                temp1[row, col] = startTime_a + durMch[row][col][i]
                LBs = calEndTimeLB(temp1, env.input_min[j],env.input_mean[j])
                reward= - (LBs.max() - env.max_endTime)

                rewards.append(reward[j])
            mch_a = mchfora[np.argmax(np.array([rewards]))]
            mch_as.append(mch_a)
    return mch_as

if __name__ == "__main__":
   ''' data=uni_instance_gen(3,3,-10,10)
    env = FJSP(n_j=3,n_m=3)
    PredictMch(env,3,data)'''