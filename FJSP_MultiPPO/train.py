from torch.optim.lr_scheduler import LambdaLR
from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action1,select_action2
from models.actor_critic import Job_Actor,Mch_Actor
from copy import deepcopy
import torch
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
from epsGreedyForMch import PredictMch
device = torch.device(configs.device)
from torch.utils.data import DataLoader
from uniform_instance import FJSPDataset
from FJSP_Env import FJSP
from rolloutbaseline import RolloutBaseline
import os
def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs
def train(epochs):

    folder = 'FJSP-N_J-{}-N_M-{}'.format(configs.n_j, configs.n_m)
    filename = 'rollout'
    filepath = os.path.join(folder, filename)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size(
                                 [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                             n_nodes=configs.n_j * configs.n_m,
                             device=device)

    job_actor = Job_Actor(n_j=configs.n_j,
                        n_m=configs.n_m,
                        num_layers=configs.num_layers,
                        learn_eps=False,
                        neighbor_pooling_type=configs.neighbor_pooling_type,
                        input_dim=configs.input_dim,
                        hidden_dim=configs.hidden_dim,
                        num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                        device=device)
    mch_actor = Mch_Actor(n_j=configs.n_j,
                        n_m=configs.n_m,
                        num_layers=configs.num_layers,
                        learn_eps=False,
                        neighbor_pooling_type=configs.neighbor_pooling_type,
                        input_dim=configs.input_dim,
                        hidden_dim=configs.hidden_dim,
                        num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                        device=device).to(device)


    validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high,1280,200)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

    rol_baseline = RolloutBaseline(job_actor,mch_actor, valid_loader, g_pool_step)


    job_actor_optim = optim.Adam(job_actor.parameters(), lr=configs.lr)
    mch_actor_optim = optim.Adam(mch_actor.parameters(), lr=configs.lr)

    times, losses, rewards2, critic_rewards = [], [], [], []
    start = time.time()
    train_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, configs.num_ins,200)
    plt = []
    data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    for epoch in range(epochs):
        job_actor.train()
        mch_actor.train()
        print("epoch:", epoch, "------------------------------------------------")



        job_losses,mch_losses, rewards, critic_loss = [],[], [], []

        for batch_idx, batch in enumerate(data_loader):

            job_scheduler = LambdaLR(job_actor_optim, lr_lambda=lambda f: 0.98 ** batch_idx)

            mch_scheduler = LambdaLR(mch_actor_optim, lr_lambda=lambda f: 0.98 ** batch_idx)
            env = FJSP(configs.n_j, configs.n_m)
            data = batch.numpy()

            adj, fea, candidate, mask,mask_mch,dur,mch_time,job_time = env.reset(data)

            job_log_prob = []
            mch_log_prob = []
            first_task = []
            pretask = []
            rewardssss = []
            j = 0
            job = candidate
            hx = []
            mch_node=[]
            R_eward = []
            actions = []
            #env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            while True:
                #print(adj[0])
                env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), configs.n_j * configs.n_m)
                env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)

                env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
                env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
                env_mask = torch.from_numpy(np.copy(mask)).to(device)
                env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)
                env_job_time = torch.from_numpy(np.copy(job_time)).float().to(device)
                action,log_a,action_node,_,mask_mch_action,hx= job_actor(x=env_fea,
                                               graph_pool=g_pool_step,
                                               padded_nei=None,
                                               adj=env_adj,
                                               candidate=env_candidate
                                               ,mask=env_mask
                                               ,pretask=pretask
                                               ,firsttask=first_task
                                               ,j=j
                                               ,mask_mch=env_mask_mch
                                               ,hx=hx
                                               ,mch_node=mch_node
                                               ,dur=env_dur
                                               ,job=job
                                               ,job_time=env_job_time
                )

                log_mch,mch_a,mch_node = mch_actor(action_node,hx,mask_mch_action,env_mch_time)
                #print(action,mch_a)

                job_log_prob.append(log_a.unsqueeze(1))

                #print(action[0].item(),mch_a[0].item())
                mch_log_prob.append(log_mch.unsqueeze(1))
                if j == 0:
                    first_task = action.type(torch.long).to(device)

                pretask =action.type(torch.long).to(device)

                adj, fea, reward, done, candidate, mask,job,mask_mch,mch_time,job_time = env.step(action.cpu().numpy(), mch_a)


                R_eward.append(reward)
                if env.done():
                    break

            job_log_p = torch.cat(job_log_prob, dim=1).sum(dim=1)
            mch_log_p = torch.cat(mch_log_prob, dim=1).sum(dim=1)
            reward = env.mchsEndTimes.max(-1).max(-1)

            r_eward = torch.tensor(R_eward).permute(1,0).sum(-1)
            base_reward = rol_baseline.eval(batch)
            advantage = torch.tensor(reward - base_reward)
            if (batch_idx + 1) % 1 == 0:
                print('reward',reward.mean(),base_reward.mean())
                print('advantage',torch.mean(advantage))
                print('log_p',torch.mean(job_log_p).item())
                print('log_p', torch.mean(mch_log_p).item())
            advantage = adv_normalize(advantage)


            job_actor_loss = torch.mean(advantage * job_log_p.cpu())
            mch_actor_loss = torch.mean(advantage * mch_log_p.cpu())

            job_actor_optim.zero_grad()
            job_actor_loss.backward(retain_graph=True)
            #grad_norms = clip_grad_norms(actor_optim.param_groups, 1)
            #torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
            job_actor_optim.step()

            job_scheduler.step()

            mch_actor_optim.zero_grad()
            mch_actor_loss.backward(retain_graph=True)
            # grad_norms = clip_grad_norms(actor_optim.param_groups, 1)
            # torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
            mch_actor_optim.step()

            mch_scheduler.step()


            rewards.append(np.mean(reward).item())
            job_losses.append(torch.mean(job_actor_loss).item())
            mch_losses.append(torch.mean(mch_actor_loss).item())
            step = 10
            if (batch_idx + 1) % step == 0:
                end = time.time()
                times.append(end - start)
                start = end

                job_mean_loss = np.mean(job_losses[-step:])
                mch_mean_loss = np.mean(mch_losses[-step:])
                mean_reward = np.mean(rewards[-step:])

                print('  Batch %d/%d, reward: %2.3f, job_loss: %2.4f,mch_loss:%2.4f, took: %2.4fs' %
                      (batch_idx, len(data_loader), mean_reward, job_mean_loss,mch_mean_loss,
                       times[-1]))
            if (batch_idx + 1) % 10 == 0:
                rol_baseline.epoch_callback(job_actor, mch_actor, batch_idx)
        print(plt)
        rol_baseline.epoch_callback(job_actor,mch_actor, epoch)
        epoch_dir = os.path.join(filepath, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        job_save_path = os.path.join(epoch_dir, 'job_actor.pt')
        mch_save_path = os.path.join(epoch_dir, 'mch_actor.pt')
        torch.save(job_actor.state_dict(), job_save_path)
        torch.save(mch_actor.state_dict(), mch_save_path)
        '''cost = rollout(actor, valid_loder, batch_size, n_nodes)
        cost = cost.mean()
        costs.append(cost.item())
        print('Problem:TSP''%s' % n_nodes, '/ Average distance:', cost.item())
        print(costs)'''

train(100)