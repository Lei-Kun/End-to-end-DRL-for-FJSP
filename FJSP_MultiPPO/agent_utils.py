from torch.distributions.categorical import Categorical

import torch
def select_action(p, cadidate, memory,log_prob):
    dist = Categorical(p.squeeze())
    s = dist.sample()

    if memory is not None: log_prob.append(dist.log_prob(s).cpu().tolist())
    action = []
    for i in range(s.size(0)):
        a = cadidate[i][s[i]].cpu().tolist()
        action.append(a)

    return action, s
def select_action1(p, cadidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    action = []
    log_a = dist.log_prob(s)

    for i in range(s.size(0)):
        a = cadidate[i][s[i]]
        action.append(a)
    action = torch.stack(action,0)
    return action, s,log_a


def select_action2(p):
    dist = Categorical(p.squeeze())
    s = dist.sample()

    #if memory is not None: log_prob.append(dist.log_prob(s).cpu().tolist())

    log_a = dist.log_prob(s)

    return s,log_a
# evaluate the actions
def eval_actions(p, actions):
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


# select action method for test
def greedy_select_action(p,cadidate):

    _, index = p.squeeze(-1).max(1)
    action = []
    for i in range(index.size(0)):
        a = cadidate[i][index[i]]
        action.append(a)
    action = torch.stack(action, 0)
    return action

# select action method for test
def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return candidate[s]
