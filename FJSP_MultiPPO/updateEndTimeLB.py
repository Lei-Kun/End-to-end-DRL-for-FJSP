import numpy as np
def lastNonZero(arr, axis, invalid_val=-1):
    mask = arr != 0#返回arr.shape大小的bool数组
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1#按指定轴翻转数组中的元素
    #如果最后一个元素不为0或全为0则np.flip(mask, axis=axis).argmax(axis=axis)等于0
    yAxis = np.where(mask.any(axis=axis), val, invalid_val)#np.any()任意元素为true即输出true#三个参数np.where(cond,x,y)：满足条件（cond）输出x，不满足输出y;cond为bool矩阵
    #数组全为0则输出-1，不全为0 则输出val，则yaxis为该数组中最后一个不为0的位置索引

    xAxis = np.arange(arr.shape[0], dtype=np.int64)

    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet


def calEndTimeLB(temp1, min,mean):

    #(action的完工时间，时间矩阵)

    x, y = lastNonZero(temp1, 1, invalid_val=-1)

    min[np.where(temp1 != 0)] = 0
    mean[np.where(temp1 != 0)] = 0

    #(temp1 != 0)->已经调度过的工序，dur_cp[np.where(temp1 != 0)] = 0将dur_cp中已经调度过的工序的时间设置为0

    min[x, y] = temp1[x, y]
    mean[x, y] = temp1[x, y]
    temp20 = np.cumsum(min, axis=1)
    temp21 = np.cumsum(mean, axis=1)#cumsum按轴依次向前累加
    temp20[np.where(temp1 != 0)] = 0
    temp21[np.where(temp1 != 0)] = 0
    temp2=np.concatenate((temp20.reshape(temp20.shape[0],temp20.shape[1],1),temp21.reshape(temp20.shape[0],temp20.shape[1],1)),-1)
    #print('temp',temp20,temp1)
    ret = temp1.reshape(temp1.shape[0],temp1.shape[1],1)+temp2

    return ret
def calEndTimeLBm(temp1, min):

    #(action的完工时间，时间矩阵)

    x, y = lastNonZero(temp1, 1, invalid_val=-1)

    min[np.where(temp1 != 0)] = 0


    #(temp1 != 0)->已经调度过的工序，dur_cp[np.where(temp1 != 0)] = 0将dur_cp中已经调度过的工序的时间设置为0

    min[x, y] = temp1[x, y]

    temp20 = np.cumsum(min, axis=1)

    temp20[np.where(temp1 != 0)] = 0


    #print('temp',temp20,temp1)
    ret = temp1+temp20

    return ret

if __name__ == '__main__':
    #dur = np.array([[1, 2], [3, 4]])
    dur=np.random.randint(1,10,(3,3))
    temp1 = np.zeros((3,3))

    temp1[0, 0] = 1
    temp1[1, 0] = 3
    temp1[1, 1] = 5


    ret = calEndTimeLB(temp1, dur,dur)