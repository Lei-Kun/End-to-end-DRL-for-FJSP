from Params import configs
import numpy as np

def DRs(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, omega,mask_last,done,mask_mch,num_operation,dispatched_num_opera,input_min,job_col,input_max,rule,last_col,first_col):
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

    remain_num = num_operation - dispatched_num_opera

    remain_num1 = np.copy(remain_num)

    if rule == 'FIFO_EET':
        while True:
            mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)

            # 返回最先完成加工的工件
            min_job_time = np.where(job_time1 == job_time.min())[0]

            min_task = omega[min_job_time]
            mchFor_minTask = []
            for z in min_task:
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]
                mch_for_job = np.where(dur[row, col] > 0)[0]
                mchFor_minTask.append(mch_for_job)

                # 计算加工该task的机器
            minMch_For_minTask = []
            mch_mask = []
            m_masks = np.copy(mask_mch)
            for i, z in enumerate(min_task):
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
                mchtask = np.array(mchFor_minTask[i])
                mch_action_space = [mchtask[np.argmin(mch_time[mchtask])]]
                for z in mch_action_space:
                    m_mask[z] = 0
                mm = m_masks[row][col] + m_mask
                if np.any(~mm):
                    m_masks[row][col] = mm
                minMch_For_minTask.append(mch_action_space)
            for i in min_task:
                mask[np.where(omega == i)] = 0
            mask = mask+mask_last
            if done:
                break
            elif np.all(mask) == True:
                job_time = np.delete(job_time, np.where(job_time == job_time.min())[0])
            else:
                break
    elif rule == 'FIFO_SPT':
        while True:
            mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)

            # 返回最先完成加工的工件
            min_job_time = np.where(job_time1 == job_time.min())[0]
            min_task = omega[min_job_time]

            mchFor_minTask = []
            for z in min_task:
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]
                mch_for_job = np.where(dur[row, col] > 0)[0]
                mchFor_minTask.append(mch_for_job)

                # 计算加工该task的机器
            minMch_For_minTask = []
            mch_mask = []
            m_masks = np.copy(mask_mch)
            for i, z in enumerate(min_task):
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]
                m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
                mchtask = np.array(mchFor_minTask[i])
                mchtimeforMINtask = dur[row][col][mchtask]
                Mintime = np.min(mchtimeforMINtask)
                mch = np.where(dur[row][col] == Mintime)[0]

                for z in mch:
                    m_mask[z] = 0
                mm = m_masks[row][col] + m_mask

                if np.any(~mm):
                    m_masks[row][col] = mm

            for i in min_task:
                mask[np.where(omega == i)] = 0
            mask = mask + mask_last

            if done:
                break
            elif np.all(mask) == True:
                job_time = np.delete(job_time, np.where(job_time == job_time.min())[0])
            else:
                break

    elif rule == 'MOPNR_SPT':
        while True:
            min_job_time = np.where(remain_num == remain_num1.max())[0]
            mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
            # 返回最先完成加工的工件
            min_task = omega[min_job_time]
            mchFor_minTask = []
            for z in min_task:

                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                mch_for_job = np.where(dur[row,col] > 0)[0]
                mchFor_minTask.append(mch_for_job)
                # 计算加工该task的机器

            m_masks = np.copy(mask_mch)
            for i,z in enumerate(min_task):
                row = np.where(z <= last_col)[0][0]

                col = z - first_col[row]
                m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
                mchtask = np.array(mchFor_minTask[i])
                mchtimeforMINtask = dur[row][col][mchtask]
                Mintime = np.min(mchtimeforMINtask)
                mch = np.where(dur[row][col] == Mintime)[0]

                for z in mch:
                    m_mask[z] = 0

                mm = m_masks[row][col] + m_mask

                if np.any(~mm):
                    m_masks[row][col] = mm
            for i in min_task:
                mask[np.where(omega == i)] = 0
            mask = mask + mask_last

            if done:
                break
            elif np.all(mask) == True:
                remain_num1 = np.delete(remain_num1, np.where(remain_num1 == remain_num1.max())[0])
            else:
                break

    elif rule == 'MOPNR_EET':
        while True:
            min_job_time = np.where(remain_num == remain_num1.max())[0]
            mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
            # 返回最先完成加工的工件
            min_task = omega[min_job_time]
            mchFor_minTask = []
            for z in min_task:
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                mch_for_job = np.where(dur[row, col] > 0)[0]
                mchFor_minTask.append(mch_for_job)
                # 计算加工该task的机器


            minMch_For_minTask = []
            m_masks = np.copy(mask_mch)
            for i, z in enumerate(min_task):
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
                mchtask = np.array(mchFor_minTask[i])
                mch_action_space = [mchtask[np.argmin(mch_time[mchtask])]]
                for z in mch_action_space:
                    m_mask[z] = 0
                mm = m_masks[row][col] + m_mask
                if np.any(~mm):
                    m_masks[row][col] = mm
                minMch_For_minTask.append(mch_action_space)
            for i in min_task:
                mask[np.where(omega == i)] = 0
            mask = mask + mask_last

            if done:
                break
            elif np.all(mask) == True:
                remain_num1 = np.delete(remain_num1, np.where(remain_num1 == remain_num1.min())[0])
            else:
                break

    elif rule == 'LWKR_SPT':
        while True:
            reverse = []
            for j in range(input_min.shape[0]):
                a = []
                for i in reversed(input_min[j]):
                    a.append(i)
                reverse.append(a)
            min = []

            for i in range(input_min.shape[0]):
                b = np.array(reverse[i][input_min.shape[1]-num_operation[i]:])
                b = b.cumsum(axis=-1)
                if job_col[i] < num_operation[i] - 1:
                    min.append(b[num_operation[i]-job_col[i]-1])
                else:
                    min.append(9999)
            min = np.array(min)

            min_job_time = np.where(min == min.min())[0]

            mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
            # 返回最先完成加工的工件
            min_task = omega[min_job_time]
            mchFor_minTask = []
            for z in min_task:
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                mch_for_job = np.where(dur[row, col] > 0)[0]
                mchFor_minTask.append(mch_for_job)

                # 计算加工该task的机器
            minMch_For_minTask = []
            mch_mask = []
            m_masks = np.copy(mask_mch)

            for i, z in enumerate(min_task):
                row = np.where(z <= last_col)[0][0]

                col = z - first_col[row]
                m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
                mchtask = np.array(mchFor_minTask[i])
                mchtimeforMINtask = dur[row][col][mchtask]
                Mintime = np.min(mchtimeforMINtask)
                mch = np.where(dur[row][col] == Mintime)[0]

                for z in mch:
                    m_mask[z] = 0

                mm = m_masks[row][col] + m_mask

                if np.any(~mm):
                    m_masks[row][col] = mm
            for i in min_task:
                mask[np.where(omega == i)] = 0

            mask = mask + mask_last

            if done:
                break

            else:
                break

    elif rule == 'LWKR_EET':
        while True:
            reverse = []
            for j in range(input_min.shape[0]):
                a = []
                for i in reversed(input_min[j]):
                    a.append(i)
                reverse.append(a)
            min = []

            for i in range(input_min.shape[0]):
                b = np.array(reverse[i][input_min.shape[1] - num_operation[i]:])
                b = b.cumsum(axis=-1)
                if job_col[i] < num_operation[i] - 1:
                    min.append(b[num_operation[i] - job_col[i] - 1])
                else:
                    min.append(9999)
            min = np.array(min)

            min_job_time = np.where(min == min.min())[0]

            mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
            # 返回最先完成加工的工件
            min_task = omega[min_job_time]
            mchFor_minTask = []
            for z in min_task:
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                mch_for_job = np.where(dur[row, col] > 0)[0]
                mchFor_minTask.append(mch_for_job)

                # 计算加工该task的机器
            minMch_For_minTask = []
            mch_mask = []
            m_masks = np.copy(mask_mch)
            for i, z in enumerate(min_task):
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
                mchtask = np.array(mchFor_minTask[i])
                mch_action_space = [mchtask[np.argmin(mch_time[mchtask])]]
                for z in mch_action_space:
                    m_mask[z] = 0
                mm = m_masks[row][col] + m_mask
                if np.any(~mm):
                    m_masks[row][col] = mm
                minMch_For_minTask.append(mch_action_space)
            for i in min_task:
                mask[np.where(omega == i)] = 0
            mask = mask + mask_last

            if done:
                break

            else:
                break

    elif rule == 'MWKR_SPT':
        while True:
            reverse = []
            for j in range(input_min.shape[0]):
                a = []
                for i in reversed(input_min[j]):
                    a.append(i)
                reverse.append(a)
            min = []

            for i in range(input_min.shape[0]):
                b = np.array(reverse[i][input_min.shape[1] - num_operation[i]:])
                b = b.cumsum(axis=-1)
                if job_col[i] < num_operation[i] - 1:
                    min.append(b[num_operation[i] - job_col[i] - 1])
                else:
                    min.append(-1)
            min = np.array(min)

            min_job_time = np.where(min == min.max())[0]

            mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
            # 返回最先完成加工的工件
            min_task = omega[min_job_time]
            mchFor_minTask = []
            for z in min_task:
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                mch_for_job = np.where(dur[row, col] > 0)[0]
                mchFor_minTask.append(mch_for_job)

                # 计算加工该task的机器
            minMch_For_minTask = []
            mch_mask = []
            m_masks = np.copy(mask_mch)

            for i, z in enumerate(min_task):
                row = np.where(z <= last_col)[0][0]

                col = z - first_col[row]
                m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
                mchtask = np.array(mchFor_minTask[i])
                mchtimeforMINtask = dur[row][col][mchtask]
                Mintime = np.min(mchtimeforMINtask)
                mch = np.where(dur[row][col] == Mintime)[0]

                for z in mch:
                    m_mask[z] = 0

                mm = m_masks[row][col] + m_mask

                if np.any(~mm):
                    m_masks[row][col] = mm
            for i in min_task:
                mask[np.where(omega == i)] = 0

            mask = mask + mask_last

            if done:
                break

            else:
                break

    elif rule == 'MWKR_EET':
        while True:
            reverse = []
            for j in range(input_min.shape[0]):
                a = []
                for i in reversed(input_min[j]):
                    a.append(i)
                reverse.append(a)
            min = []

            for i in range(input_min.shape[0]):
                b = np.array(reverse[i][input_min.shape[1] - num_operation[i]:])
                b = b.cumsum(axis=-1)
                if job_col[i] < num_operation[i] - 1:
                    min.append(b[num_operation[i] - job_col[i] - 1])
                else:
                    min.append(-1)
            min = np.array(min)

            min_job_time = np.where(min == min.max())[0]

            mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
            # 返回最先完成加工的工件
            min_task = omega[min_job_time]
            mchFor_minTask = []
            for z in min_task:
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                mch_for_job = np.where(dur[row, col] > 0)[0]
                mchFor_minTask.append(mch_for_job)

                # 计算加工该task的机器
            minMch_For_minTask = []
            mch_mask = []

            m_masks = np.copy(mask_mch)
            for i, z in enumerate(min_task):
                row = np.where(z <= last_col)[0][0]
                col = z - first_col[row]

                m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
                mchtask = np.array(mchFor_minTask[i])
                mch_action_space = [mchtask[np.argmin(mch_time[mchtask])]]
                for z in mch_action_space:
                    m_mask[z] = 0
                mm = m_masks[row][col] + m_mask
                if np.any(~mm):
                    m_masks[row][col] = mm
                minMch_For_minTask.append(mch_action_space)
            for i in min_task:
                mask[np.where(omega == i)] = 0
            mask = mask + mask_last
            if done:
                break
            else:
                break

    return mask,m_masks

def FIFO_SPT(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done,mask_mch,num_operation,dispatched_num_opera,input_min,job_col,input_max):
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
        m_masks = np.copy(mask_mch)
        for i in range(len(min_task)):
            m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
            mchtask = np.array(mchFor_minTask[i])
            mchtimeforMINtask = dur[min_task[i]][mchtask]
            Mintime = np.min(mchtimeforMINtask)
            mch = np.where(dur[min_task[i]]==Mintime)[0]

            for z in mch:
                m_mask[z] = 0
            mm = m_masks[min_task[i]]+ m_mask

            if np.any(~mm):
                m_masks[min_task[i]] = mm
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
    return mch_space, mchForActionSpace,mask,m_masks

def MOPNR_SPT(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done,mask_mch,num_operation,dispatched_num_opera,input_min,job_col,input_max):

    remain_num = num_operation - dispatched_num_opera
    remain_num1 = np.copy(remain_num)
    while True:
        min_job_time = np.where(remain_num == remain_num1.min())[0]

        mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
        # 返回最先完成加工的工件
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
        m_masks = np.copy(mask_mch)
        for i in range(len(min_task)):
            m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
            mchtask = np.array(mchFor_minTask[i])
            mchtimeforMINtask = dur[min_task[i]][mchtask]
            Mintime = np.min(mchtimeforMINtask)
            mch = np.where(dur[min_task[i]]==Mintime)[0]

            for z in mch:
                m_mask[z] = 0
            mm = m_masks[min_task[i]]+ m_mask

            if np.any(~mm):
                m_masks[min_task[i]] = mm
        for i in min_task:
            mask[np.where(first_col == i)] = 0

        mask = mask+mask_last

        if done:
            break
        elif np.all(mask) == True:
            remain_num1 = np.delete(remain_num1, np.where(remain_num1 == remain_num1.min())[0])
        else:
            break
    mch_space = minMch_For_minTask
    mchForActionSpace = min_task
    return mch_space, mchForActionSpace,mask,m_masks

def MOPNR_EET(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done,mask_mch,num_operation,dispatched_num_opera,input_min,job_col,input_max):

    remain_num = num_operation - dispatched_num_opera
    remain_num1 = np.copy(remain_num)
    while True:
        min_job_time = np.where(remain_num == remain_num1.min())[0]

        mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
        # 返回最先完成加工的工件
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
        m_masks = np.copy(mask_mch)
        for i in range(len(min_task)):
            m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
            mchtask = np.array(mchFor_minTask[i])
            mch_action_space = [mchtask[np.argmin(mch_time[mchtask])]]
            for z in mch_action_space:
                m_mask[z] = 0
            mm = m_masks[min_task[i]] + m_mask
            if np.any(~mm):
                m_masks[min_task[i]] = mm
            minMch_For_minTask.append(mch_action_space)
        for i in min_task:
            mask[np.where(first_col == i)] = 0
        mask = mask + mask_last

        if done:
            break
        elif np.all(mask) == True:
            remain_num1 = np.delete(remain_num1, np.where(remain_num1 == remain_num1.min())[0])
        else:
            break
    mch_space = minMch_For_minTask
    mchForActionSpace = min_task
    return mch_space, mchForActionSpace,mask,m_masks

def LWKR_SPT(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done,mask_mch,num_operation,dispatched_num_opera,input_min,job_col,input_max):

    while True:
        reverse = []
        for j in range(input_min.shape[0]):
            a = []
            for i in reversed(input_min[j]):
                a.append(i)
            reverse.append(a)

        reverse_sum = np.cumsum(np.array(reverse), axis=1)
        remain_num = num_operation - dispatched_num_opera

        min = []

        for i in range(reverse_sum.shape[0]):

            if job_col[i] < num_operation[i] - 1:
                min.append(reverse_sum[i][job_col[i]])
            else:
                min.append(9999)

        min = np.array(min)

        min_job_time = np.where(min == min.min())[0]


        mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
        # 返回最先完成加工的工件
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
        m_masks = np.copy(mask_mch)
        for i in range(len(min_task)):
            m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
            mchtask = np.array(mchFor_minTask[i])
            mchtimeforMINtask = dur[min_task[i]][mchtask]
            Mintime = np.min(mchtimeforMINtask)
            mch = np.where(dur[min_task[i]]==Mintime)[0]

            for z in mch:
                m_mask[z] = 0
            mm = m_masks[min_task[i]]+ m_mask

            if np.any(~mm):
                m_masks[min_task[i]] = mm
        for i in min_task:
            mask[np.where(first_col == i)] = 0

        mask = mask+mask_last

        if done:
            break

        else:
            break

    mch_space = minMch_For_minTask
    mchForActionSpace = min_task
    return mch_space, mchForActionSpace,mask,m_masks
def LWKR_EET(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done,mask_mch,num_operation,dispatched_num_opera,input_min,job_col,input_max):

    while True:
        reverse = []
        for j in range(input_min.shape[0]):
            a = []
            for i in reversed(input_min[j]):
                a.append(i)
            reverse.append(a)

        reverse_sum = np.cumsum(np.array(reverse), axis=1)
        remain_num = num_operation - dispatched_num_opera

        min = []

        for i in range(reverse_sum.shape[0]):

            if job_col[i] < num_operation[i] - 1:
                min.append(reverse_sum[i][job_col[i]])
            else:
                min.append(9999)

        min = np.array(min)

        min_job_time = np.where(min == min.min())[0]


        mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
        # 返回最先完成加工的工件
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
        m_masks = np.copy(mask_mch)
        for i in range(len(min_task)):
            m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
            mchtask = np.array(mchFor_minTask[i])
            mch_action_space = [mchtask[np.argmin(mch_time[mchtask])]]
            for z in mch_action_space:
                m_mask[z] = 0
            mm = m_masks[min_task[i]] + m_mask
            if np.any(~mm):
                m_masks[min_task[i]] = mm
            minMch_For_minTask.append(mch_action_space)
        for i in min_task:
            mask[np.where(first_col == i)] = 0

        mask = mask+mask_last

        if done:
            break

        else:
            break

    mch_space = minMch_For_minTask
    mchForActionSpace = min_task
    return mch_space, mchForActionSpace,mask,m_masks
def MWKR_SPT(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done,mask_mch,num_operation,dispatched_num_opera,input_min,job_col,input_max):

    while True:
        reverse = []
        for j in range(input_max.shape[0]):
            a = []
            for i in reversed(input_max[j]):
                a.append(i)
            reverse.append(a)

        reverse_sum = np.cumsum(np.array(reverse), axis=1)
        remain_num = num_operation - dispatched_num_opera
        min = []
        for i in range(reverse_sum.shape[0]):

            if job_col[i] < num_operation[i] - 1:
                min.append(reverse_sum[i][job_col[i]])
            else:
                min.append(-1)

        min = np.array(min)

        min_job_time = np.where(min == min.max())[0]


        mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
        # 返回最先完成加工的工件
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
        m_masks = np.copy(mask_mch)
        for i in range(len(min_task)):
            m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
            mchtask = np.array(mchFor_minTask[i])
            mchtimeforMINtask = dur[min_task[i]][mchtask]
            Mintime = np.min(mchtimeforMINtask)
            mch = np.where(dur[min_task[i]]==Mintime)[0]
            for z in mch:
                m_mask[z] = 0
            mm = m_masks[min_task[i]]+ m_mask
            if np.any(~mm):
                m_masks[min_task[i]] = mm
        for i in min_task:
            mask[np.where(first_col == i)] = 0
        mask = mask+mask_last
        if done:
            break
        else:
            break

    mch_space = minMch_For_minTask
    mchForActionSpace = min_task
    return mch_space, mchForActionSpace,mask,m_masks

def MWKR_EET(mch_time,job_time, mchsEndTimes, number_of_machines, dur, temp, first_col,mask_last,done,mask_mch,num_operation,dispatched_num_opera,input_min,job_col,input_max):

    while True:
        reverse = []
        for j in range(input_max.shape[0]):
            a = []
            for i in reversed(input_max[j]):
                a.append(i)
            reverse.append(a)

        reverse_sum = np.cumsum(np.array(reverse), axis=1)
        remain_num = num_operation - dispatched_num_opera
        min = []
        for i in range(reverse_sum.shape[0]):

            if job_col[i] < num_operation[i] - 1:
                min.append(reverse_sum[i][job_col[i]])
            else:
                min.append(-1)

        min = np.array(min)

        min_job_time = np.where(min == min.max())[0]


        mask = np.full(shape=(temp.shape[0]), fill_value=1, dtype=bool)
        # 返回最先完成加工的工件
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
        m_masks = np.copy(mask_mch)
        for i in range(len(min_task)):
            m_mask = np.full(shape=(number_of_machines), fill_value=1, dtype=bool)
            mchtask = np.array(mchFor_minTask[i])
            mch_action_space = [mchtask[np.argmin(mch_time[mchtask])]]
            for z in mch_action_space:
                m_mask[z] = 0
            mm = m_masks[min_task[i]] + m_mask
            if np.any(~mm):
                m_masks[min_task[i]] = mm
            minMch_For_minTask.append(mch_action_space)
        for i in min_task:
            mask[np.where(first_col == i)] = 0
        mask = mask+mask_last
        if done:
            break
        else:
            break

    mch_space = minMch_For_minTask
    mchForActionSpace = min_task
    return mch_space, mchForActionSpace,mask,m_masks