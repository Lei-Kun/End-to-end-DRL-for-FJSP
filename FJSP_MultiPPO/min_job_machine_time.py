from Params import configs
import numpy as np
from copy import deepcopy

def min_mch_job(mch_time,mchsEndTimes,number_of_machines,dur,temp,first_col):

    temp = np.copy(temp)
    for machine, j in zip(mchsEndTimes, range(number_of_machines)):
        if np.all(machine == -configs.high):
            mch_time[j] = 0
        else:
            mch_time[j] = machine[np.where(machine >= 0)][-1]
    while True:

        min_mch_time = np.where(mch_time == mch_time.min())[0]
        #dur(job,op,machine)->(machine,number_of_tasks)
        min_mchForJob = []
        dur = np.copy(dur)
        dur = dur.reshape(-1,number_of_machines).T
        for z in min_mch_time:
            mch_for_job = np.where(dur[z] > 0)[0]
            min_mchForJob.append(mch_for_job)


        min_mchFortask = []
        for z in range(len(min_mch_time)):

            task = np.intersect1d(min_mchForJob[z],first_col)
            if len(task) == 0:
                min_mch_time = np.delete(min_mch_time,z)
            else:
                min_mchFortask.append(task)
        if len(min_mchFortask) == 0:
            np.delete(mch_time,min_mch_time)
        else:
            break

    #min_mchFortask = np.array(min_mchFortask)
    job_time = np.zeros(temp.shape[0])
    for job, j in zip(temp, range(temp.shape[0])):
        if np.all(job == 0):
            job_time[j] = 0
        else:
            job_time[j] = job[np.where(job != 0)][-1]

    task_time = []

    mchForActionSpace = []
    for z in range(len(min_mch_time)):
        min = np.array(min_mchFortask[z])
        time = job_time[min//number_of_machines]

        job_action_space = min[np.where(time <= mch_time.min())] if len(np.where(time <= mch_time.min())[0]) != 0 else min[np.argmin(time)]
        mchForActionSpace.append(job_action_space)
        task_time.append(time)

    mch_space = min_mch_time
    mchForActionSpace = mchForActionSpace
    return mch_space,mchForActionSpace


def min_job_mch(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done,mask_mch):
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
    '''if np.all(job_time) != 0:
        job_time = np.delete(job_time, np.where(job_time == job_time.min())[0])'''
    while True:
        mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)


        min_job_time = np.where(job_time1 <= job_time.min())[0]

        min_task = first_col[min_job_time]

        dur = deepcopy(dur)
        dur = dur.reshape(-1, number_of_machines)

        mchFor_minTask = []
        for z in min_task:
            mch_for_job = np.where(dur[z] > 0)[0]
            mchFor_minTask.append(mch_for_job)


        minMch_For_minTask = []
        mch_mask = []
        m_masks = np.copy(mask_mch)

        '''for i in range(len(min_task)):
            m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
            mchtask = np.array(mchFor_minTask)[i]

            if len(np.where(mch_time[mchtask] <= job_time.min())[0]) != 0:
                mch_action_space = mchtask[np.where(mch_time[mchtask] <= job_time.min())].tolist()
                for z in mch_action_space:
                    m_mask[z] = 0

                mm = m_masks[min_task[i]] + m_mask

                if np.any(~mm):
                    m_masks[min_task[i]] = mm'''

            #minMch_For_minTask.append(mch_action_space)

        for i in min_task:
            mask[np.where(first_col == i)] = 0

        mask = mask_last + mask

        if done:
            break
        elif np.all(mask) == True:
            job_time = np.delete(job_time, np.where(job_time == job_time.min())[0])
        else:
            break


    mch_space = minMch_For_minTask

    mchForActionSpace = min_task
    return mch_space, mchForActionSpace,mask,m_masks

def min_job_mch1(mch_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done):
    temp = np.copy(temp)
    for machine, j in zip(mchsEndTimes, range(number_of_machines)):

        if np.all(machine == -configs.high):

            mch_time[j] = 0
        else:

            mch_time[j] = machine[np.where(machine >= 0)][-1]

    job_time = np.zeros(temp.shape[0])
    for job, j in zip(temp, range(temp.shape[0])):
        if np.all(job == 0):
            job_time[j] = 0
        else:
            job_time[j] = job[np.where(job != 0)][-1]

    job_time1 = np.copy(job_time)
    job_time_mean = job_time.mean()
    while True:
        mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)


        min_job_time = np.where(job_time1 <= job_time_mean)[0]

        min_task = first_col[min_job_time]

        dur = np.copy(dur)
        dur = dur.reshape(-1, number_of_machines)

        mchFor_minTask = []
        for z in min_task:
            mch_for_job = np.where(dur[z] > 0)[0]
            mchFor_minTask.append(mch_for_job)


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
