from Params import configs
import numpy as np

def FIFO_EET(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done):
    temp = np.copy(temp)
    for machine, j in zip(mchsEndTimes, range(number_of_machines)):

        if np.all(machine == -configs.high):

            mch_time[j] = 0
        else:

            mch_time[j] = machine[np.where(machine >= 0)][-1]


    for job, j in zip(temp, range(temp.shape[0])):
        if np.all(job == 0):
            job_time[j] = 0
        else:
            job_time[j] = job[np.where(job != 0)][-1]

    job_time1 = np.copy(job_time)
    job_time_mean = job_time.mean()
    while True:
        mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)

        # 返回最先完成加工的工件
        min_job_time = np.where(job_time1 == job_time.min())[0]

        min_task = first_col[min_job_time]

        dur = np.copy(dur)
        dur = dur.reshape(-1, number_of_machines)

        mchFor_minTask = []
        for z in min_task:
            mch_for_job = np.where(dur[z] > 0)[0]
            mchFor_minTask.append(mch_for_job)

            # 计算加工该task的机器
        minMch_For_minTask = []
        mch_mask = []
        for i in range(len(min_task)):
            m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
            mchtask = np.array(mchFor_minTask[i])
            mch_action_space = mchtask[np.where(mch_time[mchtask] <= job_time.min())].tolist() if len(
                    np.where(mch_time[mchtask] <= job_time.min())[0]) != 0 else [mchtask[np.argmin(mch_time[mchtask])]]
            for z in mch_action_space:
                m_mask[z] = 0

            mch_mask.append(m_mask)

            minMch_For_minTask.append(mch_action_space)

        for i in min_task:
            mask[np.where(first_col == i)] = 0

        mask = mask+mask_last

        if done:
            break
        elif np.all(mask) == True:
            job_time = np.delete(job_time, np.where(job_time == job_time.min())[0])
        else:
            break


    mch_space = minMch_For_minTask

    mchForActionSpace = min_task
    return mch_space, mchForActionSpace,mask,mch_mask