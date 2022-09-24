from Params import configs
import numpy as np

import torch

def permissibleLeftShift(a,mch_a, durMat, mchMat, mchsStartTimes, opIDsOnMchs,mchEndTime,row,col,first_col,last_col):#
    #a=action, durMat=self.dur, mchMat=mchaine, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs

    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a,mch_a, mchMat, durMat, mchsStartTimes, opIDsOnMchs,row,col,first_col,last_col)

    dur_a = durMat[row][col][mch_a]

    startTimesForMchOfa = mchsStartTimes[mch_a]#机器mch_a的start数组
    endtineformch0fa=mchEndTime[mch_a]
    #print('starttimesformchofa',startTimesForMchOfa)
    opsIDsForMchOfa = opIDsOnMchs[mch_a]#机器mch_a处理task的数组
    flag = False
    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]

    #machine中以调度的task的开始时间大于job中action的上一个task的完工时间

    if len(possiblePos) == 0:

        startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a)

    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a,mch_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa,first_col,last_col)
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a)
        else:
            flag = True
            startTime_a = putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a)
    return startTime_a, flag


def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')

    index = np.where(startTimesForMchOfa == -configs.high)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)

    startTimesForMchOfa[index] = startTime_a

    opsIDsForMchOfa[index] = a
    endtineformch0fa[index]=startTime_a+dur_a

    return startTime_a


def calLegalPos(dur_a,mch_a,jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa,first_col,last_col):
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]#possiblepos有可能是一个有可能是多个task，找到machine中tasks的starttimefomach
    durOfPossiblePos=[]
    for possiblePo in possiblePos:
        row = np.where(opsIDsForMchOfa[possiblePo] <= last_col)[0][0]
        col = opsIDsForMchOfa[possiblePo] - first_col[row]
        durOfPossiblePos.append(durMat[row,col][mch_a])

    durOfPossiblePos=np.array(durOfPossiblePos)#tasks的加工时间
    if possiblePos[0] != 0:
        row1 = np.where(opsIDsForMchOfa[possiblePos[0] - 1] <= last_col)[0][0]
        col1 = opsIDsForMchOfa[possiblePos[0]-1] - first_col[row1]


        startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0]-1] + durMat[row1,col1][mch_a])
    else:
        startTimeEarlst = max(jobRdyTime_a,0)

    endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care

    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos

    idxLegalPos = np.where(dur_a <= possibleGaps)[0]

    legalPos = np.take(possiblePos, idxLegalPos)

    return idxLegalPos, legalPos, endTimesForPossiblePos

def putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    endtineformch0fa[:]=np.insert(endtineformch0fa, earlstPos, startTime_a+dur_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]

    return startTime_a


def calJobAndMchRdyTimeOfa(a, mch_a,mchMat, durMat, mchsStartTimes, opIDsOnMchs,row,col,first_col,last_col):
    #numpy.take（a，indices，axis = None，out = None，mode ='raise' ）取矩阵中所有元素的第a个元素
    # cal jobRdyTime_a
    if col != 0:
        jobPredecessor = a - 1

    else:
        jobPredecessor = None

    #jobPredecessor = a - 1 if col != 0 else None#if a % mchMat.shape[1] = 0即该job调度完成或为第一个调度的task

    #job中action前一个task
    if jobPredecessor is not None:
        mchJobPredecessor = mchMat[row][col-1]  # 处理该task的机器

        durJobPredecessor = durMat[row,col-1,mchJobPredecessor]#加工时间

        jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)] + durJobPredecessor).item()#opIDsOnMchs->对应mchJobPredecessor----shape（machine,n_job）
        #找到数组opIDsOnMchs[mchJobPredecessor]中等于jobPredecessor的索引值####opIDsOnMchs->shape(machine,job)
    else:
        jobRdyTime_a = 0
    #cal mchRdyTime_a
    mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0 else None

    #machine中action前一个task

    if mchPredecessor is not None:

        row_1 = np.where(mchPredecessor <= last_col)[0][0]
        col_1 = mchPredecessor - first_col[row_1]
        durMchPredecessor = durMat[row_1,col_1,mch_a]

        mchRdyTime_a = (mchsStartTimes[mch_a][np.where(mchsStartTimes[mch_a] >= 0)][-1] + durMchPredecessor).item()

         #np.where()返回一个索引数组，这里返回在该machine中以调度task的索引。最后返回machine中action上一个task的结束时间
    else:

        mchRdyTime_a = 0

    return jobRdyTime_a, mchRdyTime_a



if __name__ == "__main__":
    from FJSP_Env import FJSP
    from uniform_instance import uni_instance_gen, FJSPDataset
    import time
    from torch.utils.data import DataLoader
    import os
    from DataRead import getdata




    def runDRs(rule):
        batch_size = 1
        n_j = 6
        n_m = 6
        num_operation = np.array([[6, 5, 6, 6, 6, 6]]).astype(int)
        num_opt = []
        for i in range(batch_size):
            n = np.full(shape=(n_m),fill_value=n_m)
            num_opt.append(n)
        num_operation = np.array(num_opt)

        low = -99
        high = 99
        SEED = 200
        # np.random.seed(SEED)
        t3 = time.time()
        # ------------------------------------------------------------------
        train_dataset = FJSPDataset(n_j, n_m, low, high,1,seed=SEED)
        np.random.seed(200)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        data_loader = DataLoader(train_dataset, batch_size=batch_size)
        result = []
        for batch_idx, data_set in enumerate(data_loader):
            data_set = data_set.numpy()

            # print(data_set[0])
            # print(t4)
            batch_size = data_set.shape[0]

            env = FJSP(n_j=n_j, n_m=n_m, EachJob_num_operation=num_operation)
            # rollout env random action
            t1 = time.time()
            # data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high,seed=SEED)

            # start time of operations on machines
            mchsStartTimes = -configs.high * np.ones((n_m, n_m * n_j), dtype=np.int32)
            mchsEndtTimes = -configs.high * np.ones((n_m, n_m * n_j), dtype=np.int32)
            # Ops ID on machines
            opIDsOnMchs = -n_j * np.ones([n_m, n_m * n_j], dtype=np.int32)

            # random rollout to test
            # count = 0
            adj, _, omega, mask, mch_mask, _, mch_time, _ = env.reset(data_set, rule)

            # print(env.adj)
            # mch_mask = mch_mask.reshape(batch_size, -1,n_m)
            job = omega
            rewards = []
            flags = []
            # ts = []
            # print(env.mask_mch[0])

            d = 0
            while True:
                action = []
                mch_a = []
                for i in range(batch_size):
                    a = np.random.choice(omega[i][np.where(mask[i] == 0)])
                    # index = np.where(job[i] == a)[0].item()
                    row = np.where(a <= env.last_col[i])[0][0]

                    col = a - env.first_col[i][row]
                    m = np.random.choice(np.where(mch_mask[i][row][col] == 0)[0])

                    action.append(a)
                    mch_a.append(m)
                d += 1
                '''mch_a = np.random.choice()
                mch_a = PredictMch(env,action,1)'''

                '''row = action // n_j  # 取整除
                col = action % n_m  # 取余数
                job_time=data_set[0][row][col]

                mch_a=np.random.choice(np.where(job_time>0)[0])'''

                # dur_a=data[row][col][mch_a]

                # print(mch_a)
                # print('action:', action)
                # t3 = time.time()
                # print('env_opIDOnMchs\n', env.opIDsOnMchs)
                # print('11',env.mchsEndTimes[0])
                adj, _, reward, done, omega, mask, job, mch_mask, mch_time, _ = env.step(action, mch_a)

                # print('33',env.mchsEndTimes[0])
                # print('reward',reward[0],env.dur_a)
                # t4 = time.time()
                # ts.append(t4 - t3)
                # jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a=action,mch_a=mch_a, mchMat=m, durMat=data, mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs)
                # print('mchRdyTime_a:', mchRdyTime_a,"\n",'jobrdytime',jobRdyTime_a)

                # startTime_a, flag = permissibleLeftShift(a=action, mch_a=mch_a,durMat=data.astype(np.single), mchMat=m, mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs,mchEndTime=mchsEndtTimes,dur_a=dur_a)
                # flags.append(flag)

                # print('startTime_a:', startTime_a)
                # print('mchsStartTimes\n', mchsStartTimes)
                # print('NOOOOOOOOOOOOO' if not np.array_equal(env.mchsStartTimes, mchsStartTimes) else '\n')
                # print('opIDsOnMchs\n', opIDsOnMchs)

                # print('LBs\n', env.LBs)
                rewards.append(reward)
                # print('ET after action:\n', env.LBs)
                # print()
                if env.done():
                    break
            t2 = time.time()

            # print(sum(ts))
            # print(np.sum(opIDsOnMchs // n_m, axis=1))
            # print(np.where(mchsStartTimes == mchsStartTimes.max()))
            # print(opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())])
            # print(mchsStartTimes.max() + np.take(data[0], opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())]))
            # np.save('sol', opIDsOnMchs // n_m)
            # np.save('jobSequence', opIDsOnMchs)
            # np.save('testData', data)
            # print(mchsStartTimes)
            # print(data)


            result.append(env.mchsEndTimes.max(-1).max(-1))

            # print(len(np.where(np.array(rewards) == 0)[0]))
            # print(rewards)
            '''print()

            print(env.mchsStartTimes)
            print('reward---------------', env.mchsEndTimes, env.mchsEndTimes.max(-1).max(-1))
            print()
            print(env.opIDsOnMchs[0])
            print(env.adj[0])
            # print(sum(flags))
            # data = np.load('data.npy')
            t4 = time.time() - t3
            print(t4)'''


        print(np.array(result).mean())
        return np.array(result).mean()


    DRs = ['FIFO_SPT', 'FIFO_EET', 'MOPNR_SPT', 'MOPNR_EET', 'LWKR_SPT', 'LWKR_EET', 'MWKR_SPT', 'MWKR_EET']
    results = []
    a = runDRs(DRs[0])