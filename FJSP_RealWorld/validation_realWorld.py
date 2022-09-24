from mb_agg import *
from Params import configs
from copy import deepcopy
from FJSP_Env import FJSP
from mb_agg import g_pool_cal
import copy
from DataRead import getdata
from agent_utils import sample_select_action
from agent_utils import greedy_select_action
import numpy as np
import torch
import os
from Params import configs


def validate(vali_set,batch_size, policy_job,policy_mch,num_operation,number_of_task,Data):
    policy_job = copy.deepcopy(policy_job)
    policy_mch = copy.deepcopy(policy_mch)
    policy_job.eval()
    policy_mch.eval()
    def eval_model_bat(bat):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()

            env = FJSP(n_j=Data['n'], n_m=configs.n_m,EachJob_num_operation=num_operation)
            device = torch.device(configs.device)
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [batch_size, number_of_task, number_of_task]),
                                     n_nodes=number_of_task,
                                     device=device)

            adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)
            first_task = []
            pretask = []
            j = 0

            ep_rewards = - env.initQuality
            rewards = []
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            pool=None
            while True:
                env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), number_of_task)

                env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
                env_mask = torch.from_numpy(np.copy(mask)).to(device)
                env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)
                # env_job_time = torch.from_numpy(np.copy(job_time)).float().to(device)
                action, a_idx, log_a, action_node, _, mask_mch_action, hx = policy_job(x=env_fea,
                                                                                               graph_pool=g_pool_step,
                                                                                               padded_nei=None,
                                                                                               adj=env_adj,
                                                                                               candidate=env_candidate
                                                                                               , mask=env_mask

                                                                                               , mask_mch=env_mask_mch
                                                                                               , dur=env_dur
                                                                                               , a_index=0
                                                                                               , old_action=0
                                                                                       , mch_pool=pool
                                                                                               ,old_policy=True,
                                                                                                T=1
                                                                                               ,greedy=True
                                                                                               )

                pi_mch,_,pool = policy_mch(action_node, hx, mask_mch_action, env_mch_time)

                _, mch_a = pi_mch.squeeze(-1).max(1)

                if j == 0:
                    first_task = action.type(torch.long).to(device)

                pretask = action.type(torch.long).to(device)

                adj, fea, reward, done, candidate, mask,job,_,mch_time,job_time = env.step(action.cpu().numpy(), mch_a)
                #rewards += reward

                j += 1
                print(j)
                if env.done():
                    break

            '''print(env.mchsStartTimes[0])
            print('reward---------------', env.mchsEndTimes[0], env.mchsEndTimes.max(-1).max(-1))
            print()
            print(env.opIDsOnMchs[0])
            print(env.adj[0])'''

            cost = env.mchsEndTimes.max(-1).max(-1)
            C_max.append(cost)
        return torch.tensor(cost)
    #make_spans.append(rewards - env.posRewards)
    # print(env.mchsStartTimes,env.mchsEndTimes,env.opIDsOnMchs)
    # print('REWARD',rewards - env.posRewards)
    totall_cost = torch.cat([eval_model_bat(bat) for bat in vali_set], 0)

    return totall_cost

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

if __name__ == '__main__':

    from uniform_instance import uni_instance_gen,FJSPDataset
    import numpy as np
    import time
    import argparse
    from Params import configs

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=6, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=6, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=6, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=6, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=-1, help='LB of duration')
    parser.add_argument('--high', type=int, default=1, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Cap seed for validate set generation')
    parser.add_argument('--n_vali', type=int, default=100, help='validation set size')
    params = parser.parse_args()
    def test(filepath,datafile):
        N_JOBS_P = params.Pn_j
        N_MACHINES_P = params.Pn_m
        LOW = params.low
        HIGH = params.high
        N_JOBS_N = params.Nn_j
        N_MACHINES_N = params.Nn_m
        from torch.utils.data import DataLoader
        from PPOwithValue import PPO
        import torch
        import os

        #filename11 = './FJSSPinstances/0_BehnkeGeiger/Behnke6.fjs'

        Data = getdata(data_file)

        n_j = Data['n']
        n_m = Data['m']

        ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
                  n_j=n_j,
                  n_m=n_m,
                  num_layers=configs.num_layers,
                  neighbor_pooling_type=configs.neighbor_pooling_type,
                  input_dim=configs.input_dim,
                  hidden_dim=configs.hidden_dim,
                  num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                  num_mlp_layers_actor=configs.num_mlp_layers_actor,
                  hidden_dim_actor=configs.hidden_dim_actor,
                  num_mlp_layers_critic=configs.num_mlp_layers_critic,
                  hidden_dim_critic=configs.hidden_dim_critic)

        job_path = './{}.pth'.format('policy_job')
        mch_path = './{}.pth'.format('policy_mch')

        job_path = os.path.join(filepath,job_path)
        mch_path = os.path.join(filepath, mch_path)

        ppo.policy_job.load_state_dict(torch.load(job_path))
        ppo.policy_mch.load_state_dict(torch.load(mch_path))

        num_val = 2
        batch_size = 2
        SEEDs = [200]
        num_operations = []
        num_operation = []
        for i in Data['J']:
            num_operation.append(Data['OJ'][i][-1])
        num_operation_max = np.array(num_operation).max()

        time_window = np.zeros(shape=(Data['n'], num_operation_max, Data['m']))

        data_set = []
        for i in range(Data['n']):

            for j in Data['OJ'][i+1]:
                mchForJob = Data['operations_machines'][(i + 1, j)]
                for k in mchForJob:
                    time_window[i][j-1][k - 1] = Data['operations_times'][(i + 1, j, k)]


        for i in range(batch_size):
            num_operations.append(num_operation)
            data_set.append(time_window)
        data_set = np.array(data_set)

        num_operation = np.array(num_operations)
        number_of_tasks = num_operation.sum(axis=1)[0]
        number_of_tasks = int(number_of_tasks)

        for SEED in SEEDs:

            mean_makespan = []
            #np.random.seed(SEED)
            validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, num_val, SEED)
            valid_loader = DataLoader(data_set, batch_size=batch_size)
            vali_result = validate(valid_loader,batch_size, ppo.policy_job, ppo.policy_mch,num_operation,number_of_tasks,Data).mean()
            mean_makespan.append(vali_result)
            print(np.array(mean_makespan).mean())
        return np.array(mean_makespan).mean()

        # print(min(result))


    filename = './FJSSPinstances/M15'
    filename = get_imlist(filename)
    print(filename)
    filepath = 'saved_network'
    filepath = os.path.join(filepath, 'FJSP_J%sM%s' % (15,15))
    filepaths = get_imlist(filepath)
    print(filepaths)
    #---------------------------------------------------------------------------------------------
    '''data_file = './FJSSPinstances/0_BehnkeGeiger/Behnke13.fjs'
    result = []
    for filepath in filepaths:
        a = test(filepath, data_file)
        result.append(a)
    min = np.array(result).min()
    print('min', min)'''

    #---------------------------------------------------------------------------------------------

    results = []
    for data_file in filename:
        result = []

        for filepath in filepaths:
            a = test(filepath,data_file)
            result.append(a)
        min = np.array(result).min()
        print('min',min)
        results.append(min)
    print('mins',results)


