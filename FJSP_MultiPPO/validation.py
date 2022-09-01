from epsGreedyForMch import PredictMch
from mb_agg import *
from Params import configs
from copy import deepcopy
from FJSP_Env import FJSP,DFJSP_GANTT_CHART
from mb_agg import g_pool_cal
import copy
from agent_utils import sample_select_action
from agent_utils import greedy_select_action
import numpy as np
import torch
import matplotlib.pyplot as plt
from Params import configs
def validate(vali_set,batch_size, policy_jo,policy_mc):
    policy_job = copy.deepcopy(policy_jo)
    policy_mch = copy.deepcopy(policy_mc)
    policy_job.eval()
    policy_mch.eval()
    def eval_model_bat(bat,i):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()

            env = FJSP(n_j=configs.n_j, n_m=configs.n_m)
            gantt_chart = DFJSP_GANTT_CHART( configs.n_j, configs.n_m)
            device = torch.device(configs.device)
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                     n_nodes=configs.n_j * configs.n_m,
                                     device=device)

            adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)

            j = 0

            ep_rewards = - env.initQuality
            rewards = []
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            pool=None
            while True:
                env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), configs.n_j * configs.n_m)
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
                                                                                                ,mch_pool=pool
                                                                                               ,old_policy=True,
                                                                                                T=1
                                                                                               ,greedy=True
                                                                                               )

                pi_mch,pool = policy_mch(action_node, hx, mask_mch_action, env_mch_time)

                _, mch_a = pi_mch.squeeze(-1).max(1)

                adj, fea, reward, done, candidate, mask,job,_,mch_time,job_time = env.step(action.cpu().numpy(), mch_a,gantt_chart)
                #rewards += reward

                j += 1
                if env.done():
                    plt.savefig("./3020_%s.svg"%i, format='svg',dpi=300, bbox_inches='tight')
                    #plt.show()
                    break
            cost = env.mchsEndTimes.max(-1).max(-1)
            C_max.append(cost)
        return torch.tensor(cost)
    #make_spans.append(rewards - env.posRewards)
    #print(env.mchsStartTimes,env.mchsEndTimes,env.opIDsOnMchs)
    #print('REWARD',rewards - env.posRewards)
    totall_cost = torch.cat([eval_model_bat(bat,i) for i,bat in enumerate(vali_set)], 0)

    return totall_cost



if __name__ == '__main__':

    from uniform_instance import uni_instance_gen,FJSPDataset
    import numpy as np
    import time
    import argparse
    from Params import configs

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=30, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=20, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=30, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=20, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=-99, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Cap seed for validate set generation')
    parser.add_argument('--n_vali', type=int, default=100, help='validation set size')
    params = parser.parse_args()

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
    from torch.utils.data import Dataset
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_JOBS_P,
              n_m=N_MACHINES_P,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    filepath = 'saved_network'
    filepath = os.path.join(filepath, 'FJSP_J%sM%s' % (30,configs.n_m))
    #filepath = os.path.join(filepath, '%s_%s' % (0,239))
    filepath = os.path.join(filepath, 'best_value0')

    job_path = './{}.pth'.format('policy_job')
    mch_path = './{}.pth'.format('policy_mch')



    '''filepath = 'saved_network'
    filepath = os.path.join(filepath,'%s'%19)
    job_path = './{}.pth'.format('policy_job'+str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
    mch_path = './{}.pth'.format('policy_mch'+ str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))'''

    job_path = os.path.join(filepath,job_path)
    mch_path = os.path.join(filepath, mch_path)

    ppo.policy_job.load_state_dict(torch.load(job_path))
    ppo.policy_mch.load_state_dict(torch.load(mch_path))
    num_val = 10
    batch_size = 1
    SEEDs = [200]
    result = []
    loade = False


    for SEED in SEEDs:

        mean_makespan = []
        #np.random.seed(SEED)
        if loade:
            validat_dataset = np.load(file="FJSP_J%sM%s_unew_test_data.npy" % (configs.n_j, configs.n_m))
            print(validat_dataset.shape[0])
        else:
            validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, num_val, SEED)
        valid_loader = DataLoader(validat_dataset, batch_size=batch_size)
        vali_result = validate(valid_loader,batch_size, ppo.policy_job, ppo.policy_mch)
        #mean_makespan.append(vali_result)
        print(vali_result,np.array(vali_result).mean())

    # print(min(result))

