import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance import override
from updateEndTimeLB import calEndTimeLB,calEndTimeLBm
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs
from copy import deepcopy
import torch
import random
import matplotlib.pyplot as plt
import time
from min_job_machine_time import min_job_mch,min_mch_job,min_job_mch1
class FJSP(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = []
        self.last_col = []
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs

    def done(self):
        if np.all(self.partial_sol_sequeence[0] >=0):
            return True
        return False

    @override
    def step(self, action,mch_a,gantt_plt=None):
        # action is a int 0 - 224 for 15x15 for example
        time1 = time.time()
        feas, rewards, dones,masks,mch_masks = [],[], [], [],[]
        mch_spaces, mchForJobSpaces = [],[]
        for i in range(self.batch_sie):
            # redundant action makes no effect 多余的动作无效
            if action[i] not in self.partial_sol_sequeence[i]:

                # UPDATE BASIC INFO:
                row = action[i] // self.number_of_machines#取整除
                col = action[i] % self.number_of_machines#取余数
                if i == 0:
                    self.step_count += 1
                self.finished_mark[i,row, col] = 1

                self.dur_a = self.dur[i,row, col,mch_a[i]]
                #action time
                self.partial_sol_sequeence[i][np.where(self.partial_sol_sequeence[i]<0)[0][0]] = action[i]

                self.m[i][row][col]=mch_a[i]
                # UPDATE STATE:
                # permissible left shift 允许向左移动

                startTime_a, flag = permissibleLeftShift(a=action[i], mch_a=mch_a[i], durMat=self.dur_cp[i], mchMat=self.m[i],
                                                         mchsStartTimes=self.mchsStartTimes[i], opIDsOnMchs=self.opIDsOnMchs[i],mchEndTime=self.mchsEndTimes[i])
                self.flags.append(flag)
                if gantt_plt != None:
                    gantt_plt.gantt_plt(row, col, mch_a.cpu().numpy(), startTime_a, self.dur_a,
                                    self.number_of_jobs)
                # update omega or mask
                if action[i] not in self.last_col[i]:
                    self.omega[i,action[i] // self.number_of_machines] += 1
                else:
                    self.mask[i,action[i] // self.number_of_machines] = 1

                self.temp1[i,row, col] = startTime_a + self.dur_a#完工时间

                #temp1.shape()
                self.LBs[i] = calEndTimeLB(self.temp1[i], self.input_min[i],self.input_mean[i])

                self.LBm[i] = calEndTimeLBm(self.temp1[i],self.input_min[i])


                #self.LBs为所有task最快的完工时间
                # adj matrix
                precd, succd = self.getNghbs(action[i], self.opIDsOnMchs[i])

                self.adj[i, action[i]] = 0
                self.adj[i, action[i], action[i]] = 1
                if action[i] not in self.first_col[i]:
                    self.adj[i, action[i], action[i] - 1] = 1
                self.adj[i, action[i], precd] = 1
                self.adj[i, succd, action[i]] = 1

                '''if action[i] not in self.first_col[i]:
                    self.adj[i,action[i]-1, action[i]] = 0
                self.adj[i, precd,action[i]] = 0
                self.adj[i, action[i],succd] = 0'''
                done = self.done()
                #min_job_mch(mch_time, mchsEndTimes, number_of_machines, dur, temp, first_col)
                mch_space,mchForJobSpace,mask1,mch_mask = min_job_mch(self.mch_time[i],self.job_time[i],self.mchsEndTimes[i],self.number_of_machines,self.dur_cp[i],self.temp1[i],self.omega[i],self.mask[i],done,self.mask_mch[i])

                mch_spaces.append(mch_space)
                mchForJobSpaces.append(mchForJobSpace)
                masks.append(mask1)
                mch_masks.append(mch_mask)
                #print('action_space',mchForJobSpaces,'mchspace',mch_space)

            # prepare for return
            #-------------------------------------------------------------------------------------
            '''fea = np.concatenate((self.LBs[i].reshape(-1, 2)/configs.et_normalize_coef,
                                  self.finished_mark[i].reshape(-1, 1)), axis=-1)'''
            #----------------------------------------------------------------------------------------

            '''fea = np.concatenate((self.dur[i].reshape( -1, self.number_of_machines)/configs.et_normalize_coef,
                                  self.finished_mark[i].reshape( -1, 1)), axis=-1)'''
#--------------------------------------------------------------------------------------------------------------------

            '''fea = self.LBm[i].reshape(-1, 1) / configs.et_normalize_coef'''
            fea = np.concatenate((self.LBm[i].reshape(-1, 1) / configs.et_normalize_coef,
                                  #np.expand_dims(self.job_time[i], 1).repeat(self.number_of_machines, axis=1).reshape(
                                      #self.number_of_tasks, 1)/configs.et_normalize_coef,
                                  self.finished_mark[i].reshape( -1, 1)), axis=-1)

            feas.append(fea)


            '''reward = self.mchsEndTimes[i][mch_a[i]].max()-self.up_mchendtime[i][mch_a[i]].max()-self.dur_a


            if reward < 0.00001:
                reward = 0
            self.up_mchendtime = np.copy(self.mchsEndTimes)
            for b,c in zip(self.up_mchendtime[i],range(self.number_of_machines)):
                self.up_mchendtime[i][c] = [0 if i < 0 else i for i in b]
            rewards.append(reward)'''
            reward = -(self.LBm[i].max() - self.max_endTime[i])
            if reward == 0:
                reward = configs.rewardscale
                self.posRewards[i] += reward
            rewards.append(reward)
            self.max_endTime[i] = self.LBm[i].max()

            dones.append(done)


        t2 = time.time()
        mch_masks = np.array(mch_masks)

        #print('t2',t2-t1)
        return self.adj, np.array(feas), rewards, dones, self.omega, masks,mchForJobSpaces,self.mask_mch,self.mch_time,self.job_time

    @override
    def reset(self, data):
        #data (batch_size,n_job,n_mch,n_mch)

        self.batch_sie = data.shape[0]
        for i in range(self.batch_sie):

            first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
            self.first_col.append(first_col)
        # the task id for last column
            last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
            self.last_col.append(last_col)
        self.first_col = np.array(self.first_col)
        self.last_col = np.array(self.last_col)

        self.step_count = 0
        self.m = -1 * np.ones((self.batch_sie,self.number_of_jobs,self.number_of_machines), dtype=np.int)

        self.dur = data.astype(np.single)#single单精度浮点数
        self.dur_cp = deepcopy(self.dur)
        # record action history
        self.partial_sol_sequeence = -1 * np.ones((self.batch_sie,self.number_of_jobs*self.number_of_machines),dtype=np.int)

        self.flags = []
        self.posRewards = np.zeros(self.batch_sie)
        self.adj = []
        # initialize adj matrix
        for i in range(self.batch_sie):
            conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
            conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
            # first column does not have upper stream conj_nei
            conj_nei_up_stream[self.first_col] = 0
            # last column does not have lower stream conj_nei
            conj_nei_low_stream[self.last_col] = 0
            self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
            adj = self_as_nei + conj_nei_up_stream
            self.adj.append(adj)
        self.adj = torch.tensor(self.adj)

        '''for i in range(self.batch_sie):
            dat = torch.from_numpy(data[i].reshape(-1, self.number_of_machines)).permute(1, 0)
            adj = np.eye(self.number_of_tasks)
            conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
            conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
            # first column does not have upper stream conj_nei
            conj_nei_up_stream[self.first_col] = 0
            # last column does not have lower stream conj_nei
            conj_nei_low_stream[self.last_col] = 0

            one = np.where(dat > 0)

            for i in range(self.number_of_machines):
                index = np.where(one[0] == i)[0]
                for j in one[1][index]:
                    for z in one[1][index]:
                        adj[j][z] = 1
            for i in range(1,self.number_of_tasks):
                adj[i][i - 1] = conj_nei_up_stream[i][i - 1]
                adj[i - 1][i] = 0
            self.adj.append(adj)

        self.adj = torch.tensor(self.adj).type(torch.float)'''


        # initialize features
        self.mask_mch = np.full(shape=(self.batch_sie, self.number_of_jobs,self.number_of_machines, self.number_of_machines), fill_value=0,
                            dtype=bool)
        input_min=[]
        input_mean=[]
        start = time.time()
        for t in range(self.batch_sie):
            min = []
            mean = []
            for i in range(self.number_of_jobs):
                dur_min = []
                dur_mean = []
                for j in range(self.number_of_machines):
                    durmch = self.dur[t][i][j][np.where(self.dur[t][i][j] > 0)]

                    self.mask_mch[t][i][j] = [1 if i <= 0 else 0 for i in self.dur_cp[t][i][j]]
                    self.dur[t][i][j] = [durmch.mean() if i <= 0 else i for i in self.dur[t][i][j]]
                    dur_min.append(durmch.min().tolist())
                    dur_mean.append(durmch.mean().tolist())
                min.append(dur_min)
                mean.append(dur_mean)
            input_min.append(min)
            input_mean.append(mean)
        end = time.time()-start

        self.input_min = np.array(input_min)
        self.input_mean =  np.array(input_mean)
        self.input_2d = np.concatenate([self.input_min.reshape((self.batch_sie,self.number_of_jobs,self.number_of_machines,1)),
                                        self.input_mean.reshape((self.batch_sie,self.number_of_jobs,self.number_of_machines,1))],-1)

        self.LBs = np.cumsum(self.input_2d,-2)
        self.LBm = np.cumsum(self.input_min,-1)

        self.initQuality = np.ones(self.batch_sie)
        for i in range(self.batch_sie):
            self.initQuality[i] = self.LBm[i].max() if not configs.init_quality_flag else 0

        self.max_endTime = self.initQuality

        self.job_time = np.zeros((self.batch_sie, self.number_of_jobs))
        self.finished_mark = np.zeros_like(self.m)
#--------------------------------------------------------------------------------------------------------------------------
        '''fea = self.LBm.reshape(self.batch_sie,-1, 1) / configs.et_normalize_coef'''
        fea = np.concatenate((self.LBm.reshape(self.batch_sie,-1, 1) / configs.et_normalize_coef
                              #,np.expand_dims(self.job_time,2).repeat(self.number_of_machines,axis=2).reshape(self.batch_sie,self.number_of_tasks,1)/ configs.et_normalize_coef
                              ,self.finished_mark.reshape(self.batch_sie,-1, 1)), axis=-1)
#--------------------------------------------------------------------------------------------------------------------------
        '''fea = self.dur.reshape(self.batch_sie, -1, self.number_of_machines)/configs.et_normalize_coef'''

        '''fea = np.concatenate((self.LBs.reshape(self.batch_sie,-1, 2)/configs.et_normalize_coef,
                                #self.dur.reshape(self.batch_sie,-1,self.number_of_machines)/configs.high,
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(self.batch_sie,-1, 1)), axis=-1)'''
        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)

        # initialize mask
        self.mask = np.full(shape=(self.batch_sie,self.number_of_jobs), fill_value=0, dtype=bool)

        self.mch_time = np.zeros((self.batch_sie,self.number_of_machines))
        # start time of operations on machines
        self.mchsStartTimes = -configs.high * np.ones((self.batch_sie,self.number_of_machines,self.number_of_tasks))
        self.mchsEndTimes=-configs.high * np.ones((self.batch_sie,self.number_of_machines,self.number_of_tasks))
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones((self.batch_sie,self.number_of_machines,self.number_of_tasks), dtype=np.int32)
        self.up_mchendtime = np.zeros_like(self.mchsEndTimes)
#用number_of_jobs填充数组的形状

        self.temp1 = np.zeros((self.batch_sie,self.number_of_jobs,self.number_of_machines))
        dur = self.dur.reshape(self.batch_sie,-1,self.number_of_machines)


        self.mask_mch = self.mask_mch.reshape(self.batch_sie,-1,self.mask_mch.shape[-1])
        return self.adj, fea, self.omega, self.mask,self.mask_mch,dur,self.mch_time,self.job_time
class DFJSP_GANTT_CHART():
    def __init__(self,total_n_job,number_of_machines):
        super(DFJSP_GANTT_CHART, self).__init__()

        self.total_n_job = total_n_job
        self.number_of_machines = number_of_machines
        self.initialize_plt()
    def colour_gen(self,n):
        '''
        为工件生成随机颜色
        :param n: 工件数
        :return: 颜色列表
        '''
        color_bits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        colours = []
        random.seed(234)
        for i in range(n):
            colour_bits = ['#']
            colour_bits.extend(random.sample(color_bits, 6))
            colours.append(''.join(colour_bits))
        return colours
    def initialize_plt(self):
        plt.figure(figsize=((self.total_n_job * 1.5, self.number_of_machines)))
        y_value = list(range(1, 21))

        plt.xlabel('Makespan', size=20, fontdict={'family': 'SimSun'})
        plt.ylabel('机器号', size=20, fontdict={'family': 'SimSun'})
        plt.yticks(y_value, fontproperties='Times New Roman', size=20)
        plt.xticks(fontproperties='Times New Roman', size=20)


    def gantt_plt(self,job, operation, mach_a, start_time, dur_a,number_of_jobs):
        '''
        绘制甘特图
        :param job: 工件号
        :param operation: 工序号
        :param mach_a: 机器号
        :param start_time: 开始时间
        :param dur_a: 加工时间
        :param colors: 颜色列表
        '''
        colors = self.colour_gen(number_of_jobs)
        plt.barh(mach_a + 1, dur_a, 0.5, left=start_time, color=colors[job])
        plt.text(start_time + dur_a / 10, mach_a + 0.9, 'J%s\nO%s' % (job + 1, operation + 1), size=6)