from Params import configs
import numpy as np

def getActionNbghs(action, opIDsOnMchs):

    coordAction = np.where(opIDsOnMchs == action)#action位于矩阵中的位置

    precd = opIDsOnMchs[coordAction[0], coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]].item()#位于该machine中的前一个task（除action为第一个task外）

    succdTemp = opIDsOnMchs[coordAction[0], coordAction[1] + 1 if coordAction[1].item() + 1 < opIDsOnMchs.shape[-1] else coordAction[1]].item()
    succd = action if succdTemp < 0 else succdTemp#位于该machine中的后一个task（除action为第一个task和下一个task为负外）
    # precedX = coordAction[0]
    # precedY = coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]
    # succdX = coordAction[0]
    # succdY = coordAction[1] + 1 if coordAction[1].item()+1 < opIDsOnMchs.shape[-1] else coordAction[1]
    return precd, succd

if __name__ == '__main__':
    opIDsOnMchs = np.array([[7, 29, 33, 16, -6, -6],#machine1
                            [6, 18, 28, 34, 2, -6],#machine2
                            [26, 31, 14, 21, 11, 1],
                            [30, 19, 27, 13, 10, -6],
                            [25, 20, 9, 15, -6, -6],
                            [24, 12, 8, 32, 0, -6]])
    print(opIDsOnMchs.shape[-1])

    action = 29
    precd, succd = getActionNbghs(action, opIDsOnMchs)
    print(precd, succd)
    print(opIDsOnMchs)