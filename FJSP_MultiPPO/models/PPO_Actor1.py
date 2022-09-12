import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic,MLP
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
from torch.distributions.categorical import Categorical
import torch
from Params import configs
from Mhattention import ProbAttention
from agent_utils import select_action1,greedy_select_action,select_action2
from models.Pointer import Pointer
INIT = configs.Init
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        return u_i

class Encoder(nn.Module):
    def __init__(self,num_layers, num_mlp_layers, input_dim,  hidden_dim, learn_eps, neighbor_pooling_type, device):
        super(Encoder,self).__init__()
        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
    def forward(self,x,graph_pool, padded_nei, adj,):
        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)

        return h_pooled,h_nodes

class Job_Actor(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device
                 ):
        super(Job_Actor, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        self.device=device
        self.bn = torch.nn.BatchNorm1d(input_dim).to(device)
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        #self.fc = nn.Linear(hidden_dim * 2, hidden_dim, bias=False).to(device)
        #self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        #self.fc2 = nn.Linear(1, hidden_dim, bias=False).to(device)
        '''self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)'''
        self.encoder = Encoder(num_layers=num_layers,
                               num_mlp_layers=num_mlp_layers_feature_extract,
                               input_dim=input_dim,
                               hidden_dim=hidden_dim,
                               learn_eps=learn_eps,
                               neighbor_pooling_type=neighbor_pooling_type,
                               device=device).to(device)
        self._input = nn.Parameter(torch.Tensor(hidden_dim))
        self._input.data.uniform_(-1, 1).to(device)
        #self.actor = ProbAttention(8,hidden_dim,hidden_dim).to(device)
        self.actor1 = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)

        #self.MCH_actor = ProbAttention(8, hidden_dim, hidden_dim).to(device)
        #self.attn = Attention(hidden_dim).to(device)
        #self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2, hidden_dim_actor, 1).to(device)

        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,
                mask,
                mask_mch,
                dur,
                a_index,
                old_action,
                mch_pool,
                old_policy=True,
                T=1,
                greedy=False
                ):
        #print('sssssssssssssssssssssss',x.size(),graph_pool.size(),padded_nei,adj.size(),candidate.size(),mask.size())
        h_pooled, h_nodes = self.encoder(x=x,
                                         graph_pool=graph_pool,
                                         padded_nei=padded_nei,
                                         adj=adj)

        if old_policy:
            dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))

            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)

            candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)

            # -----------------------------------------------------------------------------------------------------------
            #candidate_scores = self.actor(decoder_input, candidate_feature,0)

            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(candidate_feature)
            if mch_pool==None:
                mch_pooled_repeated = self._input[None,None, :].expand_as(candidate_feature).to(self.device)
            else:
                mch_pooled_repeated = mch_pool.unsqueeze(-2).expand_as(candidate_feature).to(self.device)
            concateFea = torch.cat((candidate_feature, h_pooled_repeated,mch_pooled_repeated), dim=-1)
            candidate_scores = self.actor1(concateFea)

            #candidate_scores = self.attn(decoder_input, candidate_feature)
            candidate_scores = candidate_scores * 10
            mask_reshape = mask.reshape(candidate_scores.size())
            candidate_scores[mask_reshape] = float('-inf')

            pi = F.softmax(candidate_scores, dim=1)
            if greedy:
                action = greedy_select_action(pi,candidate)
                log_a = 0
                index = 0
            else:
                action, index, log_a = select_action1(pi, candidate)
            action1 = action.type(torch.long).to(self.device)
            batch_x = dur.reshape(dummy.size(0), self.n_j * self.n_m, -1).to(self.device)
            mask_mch = mask_mch.reshape(dummy.size(0), -1, self.n_m)
            mask_mch_action = torch.gather(mask_mch, 1,
                                           action1.unsqueeze(-1).unsqueeze(-1).expand(mask_mch.size(0), -1,
                                                                                      mask_mch.size(2)))
            # --------------------------------------------------------------------------------------------------------------------
            action_feature = torch.gather(batch_node, 1,
                                          action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1,
                                                                                     batch_node.size(2))).squeeze(1)
            action_node = torch.gather(batch_x, 1,
                                       action1.unsqueeze(-1).unsqueeze(-1).expand(batch_x.size(0), -1,
                                                                                  batch_x.size(2))).squeeze(1)  # [:,:-2]

            return action,index, log_a, action_node.detach(), action_feature.detach(), mask_mch_action.detach(), h_pooled.detach()

        else:
            dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)

            # -----------------------------------------------------------------------------------------------------------
            #candidate_scores = self.actor(decoder_input, candidate_feature, 0)
            #candidate_scores = self.attn(h_pooled, candidate_feature)


            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(candidate_feature)
            if mch_pool == None:
                mch_pooled_repeated = self._input[None, None, :].expand_as(candidate_feature).to(self.device)
            else:
                mch_pooled_repeated = mch_pool.unsqueeze(-2).expand_as(candidate_feature).to(self.device)
            concateFea = torch.cat((candidate_feature, h_pooled_repeated, mch_pooled_repeated), dim=-1)
            candidate_scores = self.actor1(concateFea)

            candidate_scores = candidate_scores.squeeze(-1) * 10
            mask_reshape = mask.reshape(candidate_scores.size())
            candidate_scores[mask_reshape] = float('-inf')

            pi = F.softmax(candidate_scores, dim=1)
            dist = Categorical(pi)

            log_a = dist.log_prob(a_index.to(self.device))
            entropy = dist.entropy()
            action1 = old_action.type(torch.long).cuda()
            batch_x = dur.reshape(dummy.size(0), self.n_j*self.n_m, -1).to(self.device)
            mask_mch = mask_mch.reshape(dummy.size(0), -1, self.n_m)
            mask_mch_action = torch.gather(mask_mch, 1,
                                           action1.unsqueeze(-1).unsqueeze(-1).expand(mask_mch.size(0), -1,
                                                                                      mask_mch.size(2)))
            # --------------------------------------------------------------------------------------------------------------------
            action_feature = torch.gather(batch_node, 1,
                                          action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1,
                                                                                     batch_node.size(2))).squeeze(1)
            action_node = torch.gather(batch_x, 1,
                                       action1.unsqueeze(-1).unsqueeze(-1).expand(batch_x.size(0), -1,
                                                                                  batch_x.size(2))).squeeze(1)  # [:,:-2]
            v = self.critic(h_pooled)

            return entropy, v, log_a, action_node.detach(), action_feature.detach(), mask_mch_action.detach(), h_pooled.detach()


class Mch_Actor(nn.Module):
    def __init__(self,n_j,
                 n_m,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 device):
        super(Mch_Actor,self).__init__()
        self.n_j = n_j
        self.bn = torch.nn.BatchNorm1d(hidden_dim).to(device)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim).to(device)
        # machine size for problems, no business with network
        self.n_m = n_m
        self.hidden_size=hidden_dim
        self.n_ops_perjob = n_m
        self.device = device

        self.fc2 = nn.Linear(2, hidden_dim, bias=False).to(device)
        self.actor = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)
        #self.critic = MLPCritic(3, hidden_dim*2, hidden_dim, 1).to(device)
    def forward(self,action_node,hx,mask_mch_action,mch_time,mch_a=None,last_hh=None,policy=False):
        mch_time = mch_time/configs.et_normalize_coef
        action_node = action_node/configs.et_normalize_coef


        feature = torch.cat([mch_time.unsqueeze(-1), action_node.unsqueeze(-1)], -1)
        action_node = self.bn(self.fc2(feature).reshape(-1, self.hidden_size)).reshape(-1,self.n_m,self.hidden_size)
        pool = action_node.mean(dim=1)
        #action_node = self.fc2(feature)
        h_pooled_repeated = pool.unsqueeze(1).expand_as(action_node)
        pooled_repeated = hx.unsqueeze(1).expand_as(action_node)
        concateFea = torch.cat((action_node, h_pooled_repeated,pooled_repeated), dim=-1)
        mch_scores = self.actor(concateFea)
        #mch_scores = self.actor(action_feature, action_node, mask_mch_action, True)
        #mch_scores = self.attn(hx, action_node)
        mch_scores = mch_scores.squeeze(-1) * 10
        # mask_reshape = mask_mch_action.reshape(candidate_scores.size())
        mch_scores = mch_scores.masked_fill(mask_mch_action.squeeze(1).bool(), float("-inf"))
        pi_mch = F.softmax(mch_scores, dim=1)
        '''if policy:
            pools = torch.cat([pool,hx],-1)
            v = self.critic(pools)
        else:
            v = 0'''

        return pi_mch,pool




if __name__ == '__main__':
    print('Go home')
